"""CHARKHA optimizers — Muon, NormM, LR schedules, gradient utilities."""

from __future__ import annotations
import math
import torch


def _zeropower_ns5(G, steps=5):
    # Adapted from Keller Jordan's Muon reference implementation:
    # https://github.com/KellerJordan/Muon (MIT, copyright 2024 Keller Jordan).
    # The upstream license notice is reproduced in NOTICE.md.
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_shape = G.shape
    # Newton-Schulz needs a 2D matrix; flatten if >2D
    if G.ndim > 2:
        G = G.reshape(G.size(0), -1)
    dt = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(dt) / (G.norm() + 1e-7)
    flip = G.size(0) > G.size(1)
    if flip:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    if flip:
        X = X.T
    return X.reshape(orig_shape).to(G.dtype)


# --------------------------------------------------------------------------
# Grad-release (train.py --grad-release): LOMO-style optimizer-memory elimination for the big
# 2D matrices. A post-accumulate-grad hook moves each parameter's gradient to a pinned CPU
# accumulator the moment backward finishes producing it and frees the GPU copy — so the full
# fp32 gradient set (~1.6GB at 0.42B) never coexists on the GPU. Composes with gradient
# accumulation (the CPU buffer sums micro-steps) and with the CPU-offloaded Muon/NormM steps,
# which stream one matrix at a time back to the GPU (_fetch_grad) to compute the update.
# --------------------------------------------------------------------------


def _fetch_grad(p):
    """Return (grad, from_release_buffer). Optimizer-side accessor: p.grad if present, else the
    grad-release CPU accumulator (moved to p's device as a one-matrix transient)."""
    if p.grad is not None:
        return p.grad.float() if p.grad.dtype != torch.float32 else p.grad, False
    cg = getattr(p, "_cpu_grad", None)
    if cg is not None and getattr(p, "_cpu_grad_ready", False):
        return cg.to(p.device, non_blocking=True), True
    return None, False


def _consume_cpu_grad(p):
    p._cpu_grad.zero_()
    p._cpu_grad_ready = False


def install_grad_release(params):
    """Register the release hooks on `params`. Returns the count installed (0 if this torch has
    no post_accumulate_grad_hook — the caller should then keep the standard path)."""
    if not hasattr(torch.Tensor, "register_post_accumulate_grad_hook"):
        return 0

    def _release(param):
        g = param.grad
        if g is None:
            return
        param._cpu_grad.add_(
            g.detach().to(device="cpu", dtype=torch.float32)
        )  # sync D2H: g is freed right after
        param._cpu_grad_ready = True
        param.grad = None

    n = 0
    for p in params:
        buf = torch.zeros(p.shape, dtype=torch.float32, device="cpu")
        if p.is_cuda:
            try:
                buf = buf.pin_memory()
            except RuntimeError:
                pass
        p._cpu_grad = buf
        p._cpu_grad_ready = False
        p.register_post_accumulate_grad_hook(_release)
        n += 1
    return n


def clip_grads_mixed(model, max_norm):
    """clip_grad_norm_ equivalent that sees BOTH on-GPU grads and grad-release CPU accumulators.
    Returns the total norm (like torch.nn.utils.clip_grad_norm_)."""
    gpu_g, cpu_g = [], []
    for p in model.parameters():
        if p.grad is not None:
            gpu_g.append(p.grad)
        elif getattr(p, "_cpu_grad_ready", False):
            cpu_g.append(p._cpu_grad)
    sq = sum(float(g.norm()) ** 2 for g in cpu_g)
    if gpu_g:
        sq += float(torch.stack([g.norm() for g in gpu_g]).norm()) ** 2
    total = sq**0.5
    if max_norm > 0 and total > max_norm and math.isfinite(total):
        scale = max_norm / (total + 1e-6)
        for g in gpu_g:
            g.mul_(scale)
        for g in cpu_g:
            g.mul_(scale)
    return total


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, wd=0.1, cpu_offload=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, wd=wd))
        # cpu_offload: keep the momentum buffer in host RAM, stream one matrix to the GPU at a
        # time during step(). Frees the full momentum (~1.3GB at 0.42B) from VRAM during the
        # backward - identical math, only a per-step PCIe transfer (cheap vs the step itself).
        self.cpu_offload = cpu_offload

    @torch.no_grad()
    def step(self):
        for g_ in self.param_groups:
            for p in g_["params"]:
                grad, released = _fetch_grad(p)  # p.grad, or the grad-release CPU buffer
                if grad is None:
                    continue
                # fused MuonClip: Newton-Schulz orthogonalization of momentum
                st = self.state.setdefault(p, {})
                if self.cpu_offload:
                    buf = st.get("mom")
                    if buf is None:
                        buf = torch.zeros(p.shape, dtype=torch.float32, device="cpu")
                        st["mom"] = buf
                    elif (
                        buf.device.type != "cpu" or buf.dtype != torch.float32
                    ):  # re-home after a load_state_dict
                        buf = buf.to(device="cpu", dtype=torch.float32)
                        st["mom"] = buf
                    bg = buf.to(grad.device, non_blocking=True)  # only THIS matrix on GPU
                    bg.mul_(g_["momentum"]).add_(grad)
                    buf.copy_(bg)  # SYNCHRONOUS write-back: bg is mutated in place right
                    # after, so an async D2H here can capture half-mutated
                    # data -> silently poisoned momentum (intermittent!)
                    if g_["nesterov"]:
                        # in-place nesterov (bg -> update input): avoids a second full-size
                        # GPU temporary per matrix at opt.step (see NormM for the same fix)
                        bg.mul_(g_["momentum"]).add_(grad)
                    u = _zeropower_ns5(bg)
                else:
                    buf = st.setdefault("mom", torch.zeros_like(grad, dtype=torch.float32))
                    if buf.dtype != torch.float32:
                        buf = buf.float()
                        st["mom"] = buf
                    buf.mul_(g_["momentum"]).add_(grad)
                    g = grad.add(buf, alpha=g_["momentum"]) if g_["nesterov"] else buf
                    u = _zeropower_ns5(g)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                # weight decay (Moonlight/KimiK2: Muon needs explicit decay, not only AdamW)
                if g_["wd"] > 0:
                    p.mul_(1 - g_["lr"] * g_["wd"])
                p.add_(u, alpha=-g_["lr"] * scale)
                if released:
                    _consume_cpu_grad(p)


class NormM(torch.optim.Optimizer):
    """Symmetry-compatible momentum optimizer (arXiv:2605.18106). Like Muon but replaces NS5
    orthogonalization with cheap row/column normalization of the (momentum) update, matching the
    layer's symmetry group instead of imposing bi-orthogonality on everything:
      - mode='row'  -> embeddings / LM head + SwiGLU gate&up  (output / intermediate-neuron perm)
      - mode='col'  -> SwiGLU down                            (input-intermediate-neuron perm)
    Cheaper than Muon (no Newton-Schulz) and the *correct* update for these matrices per the paper."""

    def __init__(
        self, params, lr=0.02, momentum=0.95, nesterov=True, wd=0.1, mode="row", cpu_offload=False
    ):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, wd=wd))
        if mode not in ("row", "col"):
            raise ValueError(f"NormM mode must be 'row' or 'col', got {mode!r}")
        self.mode = mode
        self.cpu_offload = cpu_offload

    @torch.no_grad()
    def step(self):
        dim = 1 if self.mode == "row" else 0
        for g_ in self.param_groups:
            for p in g_["params"]:
                grad, released = _fetch_grad(p)  # p.grad, or the grad-release CPU buffer
                if grad is None:
                    continue
                st = self.state.setdefault(p, {})
                if self.cpu_offload:
                    buf = st.get("mom")
                    if buf is None:
                        buf = torch.zeros(p.shape, dtype=torch.float32, device="cpu")
                        st["mom"] = buf
                    elif buf.device.type != "cpu" or buf.dtype != torch.float32:
                        buf = buf.to(device="cpu", dtype=torch.float32)
                        st["mom"] = buf
                    bg = buf.to(grad.device, non_blocking=True)
                    bg.mul_(g_["momentum"]).add_(grad)
                    buf.copy_(bg)  # synchronous: bg is mutated next (see Muon)
                    if g_["nesterov"]:
                        # in-place: bg becomes the nesterov update. The old `g = grad.add(bg)`
                        # allocated a SECOND full-size GPU tensor — on the 131072x1280 embedding
                        # (0.67GB fp32) that transient alone blew the 8GB budget at opt.step.
                        bg.mul_(g_["momentum"]).add_(grad)
                    g = bg
                else:
                    buf = st.setdefault("mom", torch.zeros_like(grad, dtype=torch.float32))
                    if buf.dtype != torch.float32:
                        buf = buf.float()
                        st["mom"] = buf
                    buf.mul_(g_["momentum"]).add_(grad)
                    g = grad.add(buf, alpha=g_["momentum"]) if g_["nesterov"] else buf
                if self.cpu_offload:
                    g.div_(g.norm(dim=dim, keepdim=True) + 1e-7)  # in-place: no extra 0.67GB
                    u = g
                else:
                    u = g / (g.norm(dim=dim, keepdim=True) + 1e-7)  # row/col-normalized update
                if g_["wd"] > 0:
                    p.mul_(1 - g_["lr"] * g_["wd"])
                p.add_(u, alpha=-g_["lr"])
                if released:
                    _consume_cpu_grad(p)


def build_symmetry_optimizers(
    model, muon_lr=0.02, norm_lr=0.02, adam_lr=3e-3, wd=0.1, offload=False, use_8bit_adam=False
):
    """Symmetry-compatible split (arXiv:2605.18106). Returns a LIST of optimizers:
       [muon, row, col, adam]. Attention/GDN projections -> Muon (bi-orthogonal, correct);
       embeddings + SwiGLU gate/up -> RowNormM; SwiGLU down -> ColNormM; 1D/heads -> AdamW.
    NOTE: train.py currently assumes the 2-tuple from build_optimizers(); wiring this list into the
    train loop (checkpoint/LR-mult/step over N opts) is the remaining integration step."""
    muon_p, row_p, col_p, adam_p = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            p.ndim != 2
            or "k_precond" in n
            or "sngp" in n
            or n in ("halt_head.weight", "conf_head.weight")
        ):
            adam_p.append(p)  # norms, biases, gates, heads, precond, sngp
        elif "embed" in n or n.endswith("mlp.gate.weight") or n.endswith("mlp.up.weight"):
            row_p.append(p)  # row-perm symmetry
        elif n.endswith("mlp.down.weight"):
            col_p.append(p)  # column-perm symmetry
        else:
            muon_p.append(p)  # attn/GDN/adapter: Muon
    if use_8bit_adam:
        import bitsandbytes as bnb

        adam = bnb.optim.AdamW8bit(adam_p, lr=adam_lr, betas=(0.9, 0.95), weight_decay=wd)
    else:
        adam = torch.optim.AdamW(adam_p, lr=adam_lr, betas=(0.9, 0.95), weight_decay=wd)

    opts = [
        Muon(muon_p, lr=muon_lr, momentum=0.95, nesterov=True, wd=wd, cpu_offload=offload),
        NormM(row_p, lr=norm_lr, wd=wd, mode="row", cpu_offload=offload),
        NormM(col_p, lr=norm_lr, wd=wd, mode="col", cpu_offload=offload),
        adam,
    ]
    return opts


def build_optimizers(model, muon_lr=0.02, adam_lr=3e-3, wd=0.1, offload=False, use_8bit_adam=False):
    # Muon orthogonalizes 2D *hidden* weight matrices (attn/GDN/MLP/adapter projections).
    # Everything else goes to AdamW, matching modded-nanogpt/K2 practice:
    #   - the tied embedding / LM head: Muon's NS5 orthogonalization actively hurts token
    #     embeddings (they want per-row adaptive scaling, which is Adam's job);
    #   - depthwise conv kernels (ndim==3) and 1D gates/norms/biases: not matrices Muon targets;
    #   - the tiny (d,1) halt/conf heads: column vectors, Adam is the right tool.
    muon_p, adam_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_hidden_matrix = (
            p.ndim == 2
            and "embed" not in n
            and "k_precond" not in n
            and "sngp" not in n
            and n not in ("halt_head.weight", "conf_head.weight")
        )
        (muon_p if is_hidden_matrix else adam_p).append(p)
    muon = Muon(muon_p, lr=muon_lr, momentum=0.95, nesterov=True, wd=wd, cpu_offload=offload)
    if use_8bit_adam:
        import bitsandbytes as bnb

        adam = bnb.optim.AdamW8bit(adam_p, lr=adam_lr, betas=(0.9, 0.95), weight_decay=wd)
    else:
        adam = torch.optim.AdamW(adam_p, lr=adam_lr, betas=(0.9, 0.95), weight_decay=wd)
    return (muon, adam)


def wsd_lr_mult(step, warmup, decay_start=None, decay_steps=None):
    """Warmup-Stable-Decay: constant LR forever; branch a decay run for releases."""
    if step < warmup:
        return step / max(warmup, 1)
    if decay_start is not None and step >= decay_start:
        return max(0.0, 1 - (step - decay_start) / max(decay_steps, 1))
    return 1.0


# --------------------------------------------------------------------------
# Toy data: byte-level over any text file (real run: Comma BPE + Common Pile)
# --------------------------------------------------------------------------

FALLBACK_TEXT = (
    "We the People, in order to form a more perfect union, establish "
    "justice, insure domestic tranquility, provide for the common defence, promote "
    "the general welfare, and secure the blessings of liberty to ourselves and our "
    "posterity, do ordain and establish this model of the people, by the people, "
    "for the people, whose weights shall not perish from the earth. "
) * 400
