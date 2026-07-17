"""CHARKHA model — the Charkha class.  Sections: Aux heads (L13-97) · Init (L100-684) · Forward (L686-780) · Aux losses (L783-824) · Task RL (L826-897) · Generation (L900-1267)"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt
from .config import CharkhaConfig
from ._modules import *
from ._modules import _rev_run_blocks, _two_stream_blocks, make_stack
from contextlib import nullcontext as _nullcontext
from ._loss import _ce_chunk, _ce_argmax_chunk, _ce_stream


class SNGPHead(nn.Module):
    """SNGP (arXiv:2006.10108): a distance-aware GP classifier head built on Random Fourier
    Features. RFF maps the (frozen-random) hidden -> Φ(h)=sqrt(2/D)·cos(W h + b); a learnable
    weight β gives the mean logit (trained with BCE on the same is-correct target as conf_head),
    and a running precision matrix M = ridge·I + Σ Φ Φᵀ (accumulated under no_grad over training
    batches) yields the Laplace predictive VARIANCE diag(Φ M⁻¹ Φᵀ) at inference. Variance grows
    for inputs far from the training feature manifold — the epistemic signal a point-estimate
    conf_head cannot see. W,b are fixed buffers (no grad); only β learns."""

    def __init__(self, d_model, rff_dim=256, scale=1.0, ridge=1.0):
        super().__init__()
        self.rff_dim, self.ridge = rff_dim, ridge
        # fixed random projection (RBF-kernel RFF): W ~ N(0, scale), b ~ U[0, 2π)
        self.register_buffer("rff_W", torch.randn(rff_dim, d_model) * scale)
        self.register_buffer("rff_b", torch.rand(rff_dim) * (2 * math.pi))
        self.beta = nn.Linear(rff_dim, 1, bias=False)  # GP mean (the only learnable part)
        # running precision over RFF features; reset before a calibration pass, accumulated in no_grad
        self.register_buffer("precision", torch.eye(rff_dim) * ridge)

    def features(self, h):
        # h: (.., d_model) -> Φ: (.., rff_dim)
        proj = F.linear(h, self.rff_W) + self.rff_b
        return math.sqrt(2.0 / self.rff_dim) * torch.cos(proj)

    def mean_logit(self, h):
        return self.beta(self.features(h)).squeeze(-1)  # (..,) GP mean (BCE-trainable)

    @torch.no_grad()
    def accumulate_precision(self, h):
        # M += Φᵀ Φ over the flattened batch (call during a calibration pass, eval mode)
        phi = self.features(h).reshape(-1, self.rff_dim)
        self.precision += phi.t() @ phi

    @torch.no_grad()
    def reset_precision(self):
        self.precision.copy_(torch.eye(self.rff_dim, device=self.precision.device) * self.ridge)

    @torch.no_grad()
    def variance(self, h):
        # predictive variance diag(Φ M⁻¹ Φᵀ): high => far from training manifold => uncertain
        phi = self.features(h)
        cov = torch.linalg.solve(self.precision.to(phi.dtype), phi.unsqueeze(-1)).squeeze(-1)
        return (phi * cov).sum(-1)  # (..,)


class LaplaceConf:
    """Laplace-Redux (arXiv:2106.14806): POST-HOC last-layer Laplace approximation of conf_head.
    Zero training change — fit() runs after training on a small calibration loader, accumulating a
    diagonal Gauss-Newton (GGN) precision over the conf_head weights; predict() returns the
    Bayesian (mean, var) of the confidence via the linearized/probit predictive. Pairs with the
    conformal gate. Standalone (not a Module) so it never touches the training graph."""

    def __init__(self, conf_head, prior_precision=1.0):
        self.W = conf_head.weight.detach().clone()  # (1, d_model)
        self.b = conf_head.bias.detach().clone() if conf_head.bias is not None else None
        self.d = self.W.numel()
        self.prior = prior_precision
        self.post_var = None  # diagonal posterior variance over weights

    @torch.no_grad()
    def fit(self, hidden_iter):
        """hidden_iter yields post-norm hidden states h (.., d_model). Accumulates the diagonal GGN
        precision  P = prior·I + Σ p(1-p)·hᵢ²  and stores the posterior variance 1/P."""
        ggn = torch.zeros(self.d, device=self.W.device)
        for h in hidden_iter:
            h = h.reshape(-1, self.W.size(-1))
            logit = F.linear(h, self.W, self.b).squeeze(-1)
            p = torch.sigmoid(logit)
            w = (p * (1 - p)).unsqueeze(-1)  # GGN weight per sample
            ggn += (w * h.pow(2)).sum(0)  # diagonal over input dims
        self.post_var = 1.0 / (self.prior + ggn)
        return self

    @torch.no_grad()
    def predict(self, h):
        """Return (mean_conf, var_conf). Linearized predictive: logit variance = Σ var_w·h²,
        squashed through the probit approximation of the sigmoid for the mean."""
        if self.post_var is None:
            raise RuntimeError("LaplaceConf.predict before fit")
        h2 = h.reshape(-1, self.W.size(-1))
        logit = F.linear(h2, self.W, self.b).squeeze(-1)
        logit_var = (self.post_var.to(h2.dtype) * h2.pow(2)).sum(-1)
        kappa = 1.0 / torch.sqrt(1.0 + math.pi / 8.0 * logit_var)  # MacKay probit approx
        mean = torch.sigmoid(kappa * logit)
        return mean.reshape(h.shape[:-1]), logit_var.reshape(h.shape[:-1])


# --------------------------------------------------------------------------
# The model
class Charkha(nn.Module):
    def __init__(self, cfg: CharkhaConfig):
        # Lazy imports for optional staged modules
        from granary import ProductKeyMemory
        from latent_memory import LatentMemoryConfig, LatentMemorySpine

        super().__init__()
        self.cfg = cfg
        self.embed = (
            FactorizedEmbedding(cfg.vocab_size, cfg.d_model, cfg.embed_factor)
            if getattr(cfg, "embed_factor", 0)
            else nn.Embedding(cfg.vocab_size, cfg.d_model)
        )
        self.prelude = make_stack(cfg, cfg.n_prelude, final_global_attn=True)
        self.core = make_stack(cfg, cfg.n_core, final_global_attn=False)
        self.adapter = nn.Linear(
            3 * cfg.d_model if cfg.use_mtp_routing else 2 * cfg.d_model, cfg.d_model, bias=False
        )
        self.coda = make_stack(cfg, cfg.n_coda, final_global_attn=True)
        self.norm_f = RMSNorm(cfg.d_model)
        self.mtp_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.halt_head = nn.Linear(cfg.d_model, 1)  # PonderNet halting (per-token effort)
        self.conf_head = nn.Linear(cfg.d_model, 1)  # calibration: P(top-1 is correct)
        self.value_head = nn.Linear(cfg.d_model, 1)  # RL critic: predict future reward from latent
        if cfg.use_process_head:
            self.process_head = nn.Linear(cfg.d_model, 1)  # process verifier over recurrent states
        # EXPERIMENTAL low-dim recurrent scratchpad gated into the core loop (off by default)
        self.subconscious = (
            Subconscious(cfg.d_model, cfg.subconscious_dim) if cfg.use_subconscious else None
        )
        self.loop_adapters = (
            LoopAdapters(cfg.d_model, cfg.loop_adapter_rank, cfg.loop_adapter_max)
            if cfg.use_loop_adapters
            else None
        )
        self.latent_memory = (
            LatentMemorySpine(
                LatentMemoryConfig(
                    d_model=cfg.d_model,
                    slots=cfg.latent_memory_slots,
                    mem_dim=cfg.latent_memory_dim,
                    decay=cfg.latent_memory_decay,
                    temp=cfg.latent_memory_temp,
                    commit_weight=cfg.latent_memory_commit_weight,
                    pred_weight=cfg.latent_memory_pred_weight,
                    balance_weight=cfg.latent_memory_balance_weight,
                )
            )
            if cfg.use_latent_memory
            else None
        )
        # GRANARY: staged product-key memory. gate_init=0.0 => exact no-op until a
        # grow point deliberately opens it; query_bn off so eval/decode paths are exact.
        self.granary = (
            ProductKeyMemory(
                cfg.d_model,
                n_slots=cfg.granary_slots,
                d_key=cfg.granary_d_key,
                n_heads=cfg.granary_heads,
                topk=cfg.granary_topk,
                knn=cfg.granary_knn,
                gate_init=0.0,
                query_bn=False,
            )
            if cfg.use_granary
            else None
        )
        if cfg.use_nitp:  # NITP: deep hidden -> next-token shallow rep
            self.nitp_head = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        if cfg.sngp_enabled:  # SNGP: distance-aware epistemic head (off=0 cost)
            self.sngp_head = SNGPHead(cfg.d_model, cfg.sngp_rff_dim, cfg.sngp_scale, cfg.sngp_ridge)
            if cfg.sngp_spectral_norm:  # bi-Lipschitz coda -> distance-preserving features
                from torch.nn.utils.parametrizations import spectral_norm as _sn

                for blk in self.coda:
                    if hasattr(blk.mixer, "wo"):
                        blk.mixer.wo = _sn(blk.mixer.wo)
                    if hasattr(blk.mlp, "down"):
                        blk.mlp.down = _sn(blk.mlp.down)
        self._n_core_steps = 0  # core-pass counter (toy self-test hook)
        self._core_traj = None  # grad-carrying core states this forward (deep-sup)
        self._value_states = None  # value-head predictions for each core state (RL critic)
        self._last_convergence = None  # per-token extrapolation-error signal (arXiv:2606.05346)
        self._last_memory_info = None  # latent-memory telemetry (when enabled)
        if cfg.use_loop_embed:
            # sinusoidal table over loop index (like positional encoding, but over DEPTH not
            # sequence). Added to the state each recurrent pass so the shared block knows which
            # iteration it is on. A buffer (not a param): fixed, free, no grad.
            n_loops = max(cfg.max_recurrence_train, cfg.max_recurrence_infer) + 2
            self.register_buffer(
                "loop_embed", self._build_loop_embed(n_loops, cfg.d_model), persistent=False
            )
        self.apply(self._init)
        # Residual-growth control: downscale every block's *output* projection (attn/GDN wo,
        # MLP down) by 1/sqrt(2*n_layers). Standard GPT-2/modded-nanogpt trick - keeps residual
        # variance from compounding across a stack that is already deep and gets *re-run* by the
        # recurrent core (effective depth n_prelude + r*n_core + n_coda), the regime where
        # un-scaled 0.02 init drifts. Use max_recurrence_train to ensure correct scale at
        # worst-case depth; at lower r the residuals are slightly conservative — safe.
        effective_n_res = (
            cfg.n_prelude + cfg.max_recurrence_train * cfg.n_core + cfg.n_coda
            if cfg.effective_depth_scale
            else cfg.n_prelude + cfg.n_core + cfg.n_coda
        )
        res_scale = (2 * effective_n_res) ** -0.5
        for name, p in self.named_parameters():
            if name.endswith(("mixer.wo.weight", "mlp.down.weight")):
                p.data.mul_(res_scale)
        # Factorized embedding: the global _init gave up.weight std 0.02, making the composed
        # embedding output std ≈ 0.02·0.02·√f — orders too small (and the tied logits likewise).
        # Re-init up at f^-0.5 so composed output variance ≈ the dense table's 0.02² exactly.
        if getattr(cfg, "embed_factor", 0):
            nn.init.normal_(self.embed.up.weight, std=cfg.embed_factor**-0.5)
        # Bias halting head toward continuation (prevent immediate-halt collapse)
        # Initial λ ≈ σ(-2.0) ≈ 0.12 => ~88% probability of continuing to next loop
        nn.init.constant_(self.halt_head.bias, -2.0)
        # The global _init pass above re-randomized every Linear; re-establish the subconscious as an
        # exact no-op (zero write, closed gate) so it starts inert and must EARN its influence — and so
        # the equal-FLOP ablation stays a clean control.
        if self.subconscious is not None:
            self.subconscious.reset_noop()
        if self.latent_memory is not None:
            self.latent_memory.reset_noop()
        if self.loop_adapters is not None:
            self.loop_adapters.reset_noop()
        # Same hazard for the self-gated reasoning modules (math / logic playground / calculus /
        # modules own no mixer.wo/mlp.down weights, so this is independent of the res_scale pass above.

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    @staticmethod
    def _build_loop_embed(n_loops, dim):
        pos = torch.arange(n_loops).float().unsqueeze(1)
        inv = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        emb = torch.zeros(n_loops, dim)
        emb[:, 0::2] = torch.sin(pos * inv)
        emb[:, 1::2] = torch.cos(pos * inv)
        return emb

    def _tok_embed(self, idx):
        """Token embedding — the SOLE entry point for idx -> embedding space, so every forward
        path (forward/hidden/shallow) gets identical treatment."""
        return self.embed(idx)

    def _sample_r(self, batch=None):
        if not self.cfg.use_recurrence:
            return 1
        lam = float(self.cfg.mean_recurrence - 1)
        if batch is not None and self.cfg.per_seq_recurrence and self.training and batch > 1:
            r = 1 + torch.poisson(torch.full((batch,), lam)).long()  # (B,) per-sequence (LoopWM)
            return r.clamp(1, self.cfg.max_recurrence_train)
        r = 1 + int(torch.poisson(torch.tensor(lam)))
        return min(max(r, 1), self.cfg.max_recurrence_train)

    def _initial_core_state(self, e):
        # Training uses noisy initial states (regularization); eval is deterministic zero UNLESS
        # _seed_noise is set (generate(seed_noise=...)): latent-seed sampling draws DIFFERENT
        # in-distribution "trains of thought" from the recurrence at identical temperature —
        # the diversity source for best-of-N reranking that token-level temperature cannot give.
        noise = (
            float(self.cfg.recurrent_state_noise)
            if self.training
            else float(getattr(self, "_seed_noise", 0.0))
        )
        if noise > 0:
            return torch.randn_like(e) * noise
        return torch.zeros_like(e)

    def _run_core_sampled(self, x, batch, mtp_future=None):
        # route the fixed (non-halting) recurrent core: per-sequence depth when enabled (batch>1),
        # else the scalar fast path. Keeps batch==1 / per_seq_recurrence=False byte-identical.
        r = self._sample_r(batch)
        if torch.is_tensor(r):
            return self._run_core_fixed_perseq(x, r, mtp_future=mtp_future)
        return self._run_core_fixed(x, r, mtp_future=mtp_future)

    def _core_step(self, s, e, n=0, err=None, mtp_future=None, caches=None):
        # one pass of the shared recurrent core: inject e via the concat-adapter, run blocks.
        # The loop-index signal differentiates this iteration from the others (see use_loop_embed).
        # err: per-token prediction error from previous loop (active inference feedback, T1-A).
        # mtp_future: predicted next-token representation from MTP head (multi-scale reasoning).
        inputs = [s, e]
        if mtp_future is not None and self.cfg.use_mtp_routing:
            inputs.append(mtp_future)
        h = self.adapter(torch.cat(inputs, -1))
        if self.cfg.use_loop_embed:
            idx = min(int(n), self.loop_embed.size(0) - 1)
            h = h + self.loop_embed[idx].to(h.dtype)
        if getattr(self.cfg, "reversible", False):
            if caches is None and self.training and torch.is_grad_enabled():
                h = _rev_run_blocks(self.core, h)
            else:  # eval / streaming decode: same two-stream function, plain compute
                h = _two_stream_blocks(
                    self.core, h, caches=list(caches) if caches is not None else None
                )
        elif caches is None:
            for blk in self.core:
                h = blk(h)
        else:
            for blk, c in zip(self.core, caches):
                h = blk(h, cache=c)
        if self.loop_adapters is not None:
            h = self.loop_adapters(h, n)  # per-loop specialization (no-op at init)
        return h

    def _step(self, s, e, n=0, err=None, mtp_future=None):
        # gradient-checkpoint the core pass so unrolling N loops keeps only O(1) sets of
        # activations live (recompute on backward) - the 8GB-budget enabler for the 0.4B run.
        self._n_core_steps += 1  # instrumentation (asserted in the toy self-test)
        if self.cfg.grad_checkpoint and self.training and torch.is_grad_enabled():
            return _ckpt.checkpoint(self._core_step, s, e, n, err, mtp_future, use_reentrant=False)
        return self._core_step(s, e, n, err, mtp_future)

    def _run_blocks(self, blocks, x):
        # Run a (prelude/core/coda) block stack, gradient-checkpointing each block when enabled.
        # These run once per forward (unlike the looped core), so checkpointing trades one cheap
        # recompute for ~all their activation memory - without it the uncheckpointed prelude+coda
        # activations at T=4096 exhaust 8GB during the forward, before backward even starts.
        if getattr(self.cfg, "reversible", False) and len(blocks) > 0:
            # Two-stream reversible coupling. Training: O(1) activation memory over the whole
            # stack via the inversion backward (subsumes per-block checkpointing). Eval/no-grad:
            # the same function computed plainly — train/serve consistency.
            if self.training and torch.is_grad_enabled():
                return _rev_run_blocks(blocks, x)
            return _two_stream_blocks(blocks, x)
        ckpt = self.cfg.grad_checkpoint and self.training and torch.is_grad_enabled()
        for blk in blocks:
            x = _ckpt.checkpoint(blk, x, use_reentrant=False) if ckpt else blk(x)
        return x

    def _run_core_fixed(self, e, r, mtp_future=None):
        # Huginn recipe (manual effort dial): random state init; inject e every loop via
        # concat-adapter; truncated backprop through the last `backprop_depth` iterations.
        # Total core passes are always exactly r: the first r-k run under no_grad, the last k
        # carry gradients (k=0 at inference -> all r run no_grad, none repeated).
        s = self._initial_core_state(e)
        z = self.subconscious.init_state(e) if self.subconscious is not None else None
        k = min(r, self.cfg.backprop_depth) if self.training else 0
        # A2 convergence: keep a short rolling buffer of recent states (inference-only, cheap)
        track = self.cfg.track_convergence and not self.training
        cbuf = [] if track else None
        win = max(2, self.cfg.convergence_window)
        s_prev = None
        err = None
        with torch.no_grad():
            for n in range(r - k):
                s = self._step(s, e, n, err, mtp_future)
                if self.subconscious is not None:  # scratchpad update BETWEEN loops (no_grad here)
                    s, z = self.subconscious(s, z)
                if track:
                    cbuf.append(s)
                    if len(cbuf) > win + 1:
                        cbuf.pop(0)
        traj = []
        for n in range(r - k, r):
            s = self._step(s, e, n, err, mtp_future)
            if self.subconscious is not None:  # scratchpad update BETWEEN loops (grad-carrying)
                s, z = self.subconscious(s, z)
            traj.append(s)
        # stash the gradient-carrying loop states for anytime deep supervision (training only)
        self._core_traj = traj if (self.training and len(traj) >= 2) else None
        self._value_states = [self.value_head(sj).squeeze(-1) for sj in traj] if traj else None
        if track:
            self._last_convergence = self._convergence_signal(cbuf)
        return s

    def _run_core_fixed_perseq(self, e, r_vec, mtp_future=None):
        # LoopWM per-sequence depth (arXiv:2606.18208 §3.3): each row of the batch runs its OWN
        # loop count r_b (sampled per sequence), freezing once it hits r_b; truncated BPTT keeps
        # grad only on each row's last k_b=min(r_b,backprop_depth) loops. Because the no-grad/grad
        # split is per-row, we can't use a global torch.no_grad() prefix (a shallow row would then
        # carry zero gradient) — instead we run every loop with grad enabled and DETACH the rows
        # that should not propagate this loop. That keeps the full activation graph for the batch,
        # so this path is for batch>1 on big-VRAM (the cloud config); batch==1 uses the scalar path.
        B = e.size(0)
        r_vec = r_vec.to(e.device).long().clamp(min=1, max=self.cfg.max_recurrence_train)
        Rmax = int(r_vec.max().item())
        bd = self.cfg.backprop_depth if self.training else 0
        view = (B,) + (1,) * (e.dim() - 1)  # broadcast (B,1,1)
        rb = r_vec.view(view)
        kb = r_vec.clamp(max=bd) if self.training else torch.zeros_like(r_vec)
        gstart = (r_vec - kb).view(view)  # grad window starts at r_b - k_b
        s = self._initial_core_state(e)
        s_prev = None
        err = None
        traj = []
        for n in range(Rmax):
            # Per-row loop counts make the "final loop" row-dependent, so (like the halting core)
            # run the self-gated reasoning modules on every loop in both modes.
            s_new = self._step(s, e, n, err, mtp_future)
            active = n < rb  # row still looping
            grad_on = active & (n >= gstart)  # row propagates grad this loop
            s_upd = torch.where(grad_on, s_new, s_new.detach())
            s = torch.where(active, s_upd, s)  # frozen rows keep their state
            if self.training and bd > 0 and n >= Rmax - bd:
                traj.append(s)
        self._core_traj = traj if (self.training and len(traj) >= 2) else None
        self._value_states = [self.value_head(sj).squeeze(-1) for sj in traj] if traj else None
        return s

    @torch.no_grad()
    def _run_core_converge(self, e, mtp_future=None):
        """Equilibrium effort mode (inference-only, zero params): run mean_recurrence loops with
        their real loop indices (the trained regime), then FREEZE the loop-index signal — making
        the core a stationary map s <- f(s) — and iterate with Anderson acceleration until the
        relative state change drops below converge_tol (or converge_max_iter caps it). The effort
        dial becomes "think until settled": depth is unbounded but principled, a deep-equilibrium
        readout of the PonderNet-trained looped core. Falls back to plain iteration whenever the
        accelerated candidate is degenerate (rank-deficient lstsq / non-finite), so the mode can
        never be WORSE-behaved than plain unrolling."""
        if self.training:
            raise RuntimeError("converge mode is inference-only")
        cfg = self.cfg
        s = torch.zeros_like(e)
        n_warm = max(1, cfg.mean_recurrence)
        for n in range(n_warm):
            s = self._step(s, e, n, None, mtp_future)
        n_star = n_warm - 1  # frozen loop index -> stationary f
        hist_s, hist_f = [], []
        iters = 0
        for iters in range(1, cfg.converge_max_iter + 1):
            f = self._step(s, e, n_star, None, mtp_future)
            rel = ((f - s).norm() / s.norm().clamp_min(1e-6)).item()
            self._last_convergence = (f - s).norm(dim=-1)  # per-token settledness signal
            hist_s.append(s.reshape(-1))
            hist_f.append(f.reshape(-1))
            if len(hist_s) > max(1, cfg.converge_anderson_m):
                hist_s.pop(0)
                hist_f.pop(0)
            if rel < cfg.converge_tol or not math.isfinite(rel):
                s = f
                break
            if len(hist_s) >= 2:
                # Anderson: pick mixing weights over the history that minimize the linearized
                # residual ||Σ a_i g_i||, Σ a_i = 1 (g_i = f_i - s_i), then combine the f_i.
                G = torch.stack([hf - hs for hf, hs in zip(hist_f, hist_s)], 1).float()  # (N, m)
                Fm = torch.stack(hist_f, 1).float()  # (N, m)
                dG, g_last = G[:, :-1] - G[:, -1:], G[:, -1:]
                try:
                    w = torch.linalg.lstsq(dG, -g_last).solution  # (m-1, 1)
                    cand = (Fm[:, -1:] + (Fm[:, :-1] - Fm[:, -1:]) @ w).reshape_as(e)
                    s = cand.to(e.dtype) if torch.isfinite(cand).all() else f
                except Exception:
                    s = f
            else:
                s = f
        self._last_converge_iters = n_warm + iters
        return s

    def _convergence_signal(self, buf):
        # Linear-extrapolation error of the recurrent-core trajectory (arXiv:2606.05346).
        # Predict the final state from a constant-velocity fit over the prior states, return the
        # per-token residual norm ||s_n - s_pred|| (B,T). Falls back to last-step displacement when
        # there are too few loops to extrapolate. None if no states ran. Small => converged.
        if not buf:
            return None
        if len(buf) < 3:  # not enough history for either 3-pt signal
            d = buf[-1] - buf[-2] if len(buf) >= 2 else torch.zeros_like(buf[-1])
            return d.norm(dim=-1)
        if (
            self.cfg.convergence_mode == "acceleration"
        ):  # 2nd difference (curvature); small => settled
            return (buf[-1] - 2 * buf[-2] + buf[-3]).norm(dim=-1)
        v = buf[-2] - buf[-3]  # velocity from the two states before last
        s_pred = buf[-2] + v  # extrapolate to the final state
        return (buf[-1] - s_pred).norm(dim=-1)  # extrapolation residual (default)

    def _run_core_halting(self, e, mtp_future=None):
        # PonderNet/ACT: loop the shared core, learn a per-token halting probability at
        # each step. The state passed forward is the halt-weighted expectation E[s]; the
        # halting distribution is regularized toward a geometric prior (mean recurrence).
        B, T, _ = e.shape
        N = self.cfg.max_recurrence_train if self.training else self.cfg.max_recurrence_infer
        s = self._initial_core_state(e)
        z = self.subconscious.init_state(e) if self.subconscious is not None else None
        remainder = e.new_ones(B, T)
        out = torch.zeros_like(e)
        steps = e.new_zeros(B, T)
        halts = []
        traj = [] if self.training else None  # save last states for A1/A3
        k_bptt = self.cfg.backprop_depth if self.training else 0
        accel_exit = self.cfg.use_accel_exit and not self.training
        s_hist = [] if accel_exit else None
        s_prev = None
        err = None
        for n in range(1, N + 1):
            grad_on = (not self.training) or (n > N - max(k_bptt, 1))
            with _nullcontext() if grad_on else torch.no_grad():
                # Halting's final loop is DYNAMIC at inference (early break) — run modules every loop.
                s = self._step(s, e, n - 1, err, mtp_future)
                if self.subconscious is not None:  # refine state with the scratchpad, in-context
                    s, z = self.subconscious(s, z)
                if n < N:
                    lam = torch.sigmoid(self.halt_head(s)).squeeze(-1)
                else:
                    lam = torch.ones(B, T, device=e.device, dtype=e.dtype)
                lam = self._pool_halt_prob(lam)
                p = remainder * lam
                out = out + p.unsqueeze(-1) * s
                steps = steps + p * n
            # track halting confidence for thermostat
            if not self.training and self.cfg.use_thermostat:
                if not hasattr(self, "_halting_conf_hist"):
                    self._halting_conf_hist = []
                self._halting_conf_hist.append(lam.detach().mean())  # 0-dim scalar, NOT the (B,T)
                if len(self._halting_conf_hist) > 5:
                    self._halting_conf_hist.pop(0)  # tensor — see below
            halts.append(p)
            remainder = remainder * (1 - lam)
            if self.training and grad_on and traj is not None:
                traj.append(s)  # gradient-carrying
            # Epistemic thermostat: force continue if model is confused or oscillating
            if not self.training and self.cfg.use_thermostat and n >= 3 and n < N:
                halt_signal = remainder.max().item() < (1 - self.cfg.halt_threshold)
                oscillating = False
                if accel_exit and len(s_hist) >= 3:
                    curv = (s_hist[-1] - 2 * s_hist[-2] + s_hist[-3]).norm(dim=-1)
                    oscillating = curv.mean().item() > self.cfg.accel_exit_threshold * 1.5
                confused = False
                if hasattr(self, "_halting_conf_hist") and len(self._halting_conf_hist) >= 3:
                    # stack of 0-dim scalars (per-step mean halt-prob). Storing the raw (B,T) tensors
                    # instead crashed here during generation: T grows each token, so the cached tensors
                    # had mismatched shapes and torch.stack raised. Scalars are shape-stable.
                    conf_var = torch.stack(self._halting_conf_hist[-3:]).var().item()
                    confused = conf_var > self.cfg.thermostat_conf_threshold
                if halt_signal and (oscillating or confused):
                    remainder = remainder + p  # restore remainder mass, force continue
                    continue
            if not self.training and remainder.max().item() < (1 - self.cfg.halt_threshold):
                break
            if accel_exit:
                s_hist.append(s)
                if len(s_hist) > 3:
                    s_hist.pop(0)
                if len(s_hist) == 3:
                    accel = (s_hist[-1] - 2 * s_hist[-2] + s_hist[-3]).norm(dim=-1).mean().item()
                    self._last_convergence = (s_hist[-1] - 2 * s_hist[-2] + s_hist[-3]).norm(dim=-1)
                    if accel < self.cfg.accel_exit_threshold:
                        break
        out = out + remainder.unsqueeze(-1) * s  # assign any leftover prob mass
        self._core_traj = traj if (self.training and traj and len(traj) >= 2) else None
        self._value_states = [self.value_head(sj).squeeze(-1) for sj in traj] if traj else None
        return out, steps, torch.stack(halts, -1)

    def _pool_halt_prob(self, lam):
        """Optional effort sharing across spans. Token-level ACT remains the default."""
        mode = getattr(self.cfg, "halt_granularity", "token")
        if mode == "token":
            return lam
        B, T = lam.shape
        if mode == "sequence":
            return lam.mean(dim=1, keepdim=True).expand(B, T)
        if mode != "segment":
            return lam
        seg = max(1, int(getattr(self.cfg, "halt_segment_len", 16)))
        if seg <= 1:
            return lam
        pad = (-T) % seg
        if pad:
            lp = F.pad(lam, (0, pad), value=0.0)
            mask = F.pad(torch.ones_like(lam), (0, pad), value=0.0)
        else:
            lp = lam
            mask = torch.ones_like(lam)
        lp = lp.view(B, -1, seg)
        mask = mask.view(B, -1, seg)
        pooled = (lp * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)
        return pooled.unsqueeze(-1).expand(-1, -1, seg).reshape(B, -1)[:, :T]

    def _ponder_loss(self, halt_dist):  # KL(halt || geometric prior)
        N = halt_dist.size(-1)
        g = 1.0 / max(self.cfg.mean_recurrence, 1)
        n = torch.arange(1, N + 1, device=halt_dist.device, dtype=halt_dist.dtype)
        prior = (1 - g) ** (n - 1) * g
        prior = prior / prior.sum()
        # clamp the prior away from 0 before log: at mean_recurrence=1, g=1 makes the geometric
        # prior [1,0,0,…] and log(0)=-inf → the KL is +inf, which poisoned the whole loss (the
        # Phase-7 curriculum spends its first steps at r=1, hence the run of `non-finite loss (inf)`
        # there). Clamped, the penalty stays finite and still strongly favors halting at step 1.
        p = halt_dist.clamp_min(1e-9)
        kl = p * (p.log() - prior.clamp_min(1e-9).log())  # (..., N) per-step KL
        # Mask the FORCED truncation step: at n=N we set lam=1 and dump all leftover probability mass
        # there, but the geometric prior is ~0 at N, so the KL on that step is huge and purely an
        # artifact of truncation. Penalizing it teaches "halting anxiety" — the model collapses to
        # r=1 to avoid ever reaching N. Zero it so the prior only shapes the genuinely-learned halts.
        mask = torch.ones_like(kl)
        mask[..., -1] = 0.0
        return (kl * mask).sum(-1).mean()

    def _head_hw(self, h):
        """Factorization-aware tied head: returns (h', W') with logits = h' @ W'ᵀ.
        Dense: (h, embed.weight). Factorized: (h @ up (N, f), codes.weight (V, f)) — the (V, d)
        table is never materialized, and the f-dim projection also shrinks every CE chunk."""
        if getattr(self.cfg, "embed_factor", 0):
            return h.matmul(self.embed.up.weight), self.embed.codes.weight
        return h, self.embed.weight

    def _fused_ce(self, h, tgt, want_correct=False, prior=None):
        # Mean CE over (B*T) without ever materializing the full logits tensor: split the
        # rows into cfg.ce_chunk-sized pieces, checkpoint each (linear -> CE) so the chunk's
        # logits/softmax are recomputed in backward, not kept alive. This is what lets the
        # backward fit in 8GB at T=4096. Identical result to one F.cross_entropy(mean).
        # prior: optional (hp, Wp, weight) — a FROZEN fluency model's hidden states + head; its
        # logits are added chunk-by-chunk (never a full (B,T,V) tensor) so gradient only flows
        # into what the prior cannot already predict (train.py --logit-prior).
        h, W = self._head_hw(h)
        Hf, Tf = h.reshape(-1, h.size(-1)), tgt.reshape(-1)
        hp, Wp, pw = (None, None, 0.0)
        if prior is not None:
            hp, Wp, pw = prior
            hp = hp.reshape(-1, hp.size(-1))
        chunk = max(1, self.cfg.ce_chunk)
        ckpt = self.training and torch.is_grad_enabled()
        cap = float(getattr(self.cfg, "logit_softcap", 0.0))
        zl = float(getattr(self.cfg, "z_loss", 0.0))
        ce = h.new_zeros(())
        correct = [] if want_correct else None
        vchunk = int(getattr(self.cfg, "ce_vchunk", 0) or 0)
        for i in range(0, Hf.size(0), chunk):
            hc, tc = Hf[i : i + chunk], Tf[i : i + chunk]
            hpc = hp[i : i + chunk] if hp is not None else None
            if vchunk > 0:
                # CCE-style vocab streaming: the (chunk, V) logit block never materializes.
                c, cor = _ce_stream(
                    hc, W, tc, vchunk, ckpt, hpc, Wp, pw, cap, zl, want_correct=want_correct
                )
                if want_correct:
                    correct.append(cor)
                ce = ce + c
                continue
            if want_correct:
                c, cor = (
                    _ckpt.checkpoint(
                        _ce_argmax_chunk, hc, W, tc, hpc, Wp, pw, cap, zl, use_reentrant=False
                    )
                    if ckpt
                    else _ce_argmax_chunk(hc, W, tc, hpc, Wp, pw, cap, zl)
                )
                correct.append(cor)
            else:
                c = (
                    _ckpt.checkpoint(
                        _ce_chunk, hc, W, tc, hpc, Wp, pw, cap, zl, use_reentrant=False
                    )
                    if ckpt
                    else _ce_chunk(hc, W, tc, hpc, Wp, pw, cap, zl)
                )
            ce = ce + c
        ce = ce / max(Tf.numel(), 1)
        return (ce, torch.cat(correct)) if want_correct else ce

    def _kd_topk(self, h, t_idx, t_prob, temp):
        # Memory-frugal soft-label distillation: KL(teacher || student) over the teacher's top-k
        # tokens only. Gathering k vocab columns per position keeps peak at (chunk,k,d) - we never
        # build the (B,T,V) logits the fused-CE design avoids. Assumes a SHARED tokenizer with the
        # teacher (CHARKHA uses GPT-NeoX BPE; pair with any GPT-NeoX-tokenizer teacher — frontier
        # KD is in scope, the artifact is non-distributable either way).
        h, W = self._head_hw(h)  # (V, d) tied head (or factorized (V, f))
        Hf = h.reshape(-1, h.size(-1))  # (N, d) or (N, f)
        If = t_idx.reshape(-1, t_idx.size(-1))  # (N, k)
        Pf = t_prob.reshape(-1, t_prob.size(-1))  # (N, k)
        chunk = max(1, self.cfg.ce_chunk)
        kd = h.new_zeros(())
        for i in range(0, Hf.size(0), chunk):
            hc, ic, pc = Hf[i : i + chunk], If[i : i + chunk], Pf[i : i + chunk]
            lk = (hc.unsqueeze(1) * W[ic]).sum(-1)  # (chunk, k) student logits over top-k
            slp = F.log_softmax(lk / temp, dim=-1)
            kd = kd + (pc * (pc.clamp_min(1e-9).log() - slp)).sum(-1).sum()
        return (kd / max(Hf.size(0), 1)) * (temp * temp)  # KD temp^2 convention

    def shallow(self, idx):
        """Post-prelude shallow features (B,T,d_model) — the representation the recurrent CORE
        consumes (forward order: embed -> prelude -> [shallow] -> core -> coda -> norm_f). This is
        the correct input space for replaying states back through core/coda (e.g. somatic wake-sleep
        consolidation in continual.py), UNLIKE hidden(), which returns the post-coda, post-norm_f
        representation that feeds the heads. Grad-carrying (respects training mode)."""
        return self._run_blocks(self.prelude, self._tok_embed(idx))

    def hidden(self, idx, r=None, soft_prefix=None):
        """Post-norm hidden states h (B,T,d_model) — the representation feeding the LM/conf/SNGP
        heads — without building any head. Grad-carrying (respects training mode); used by RLCM
        margin training (conf_margin_loss over good/bad prefixes) and as a generic probe hook.
        Routes the recurrent core exactly like forward: fixed-r if given, else halting/sample.
        soft_prefix (B,G,d_model): optional learned pseudo-embeddings prepended BEFORE the
        prelude (fold.py context compression). None → bit-identical to the original path;
        when given, the returned h covers prefix+idx positions (caller slices [:, G:])."""
        self._core_traj = None  # clear stale trajectory from a prior call
        self._last_convergence = None
        self._last_memory_info = None
        x = self._tok_embed(idx)
        if soft_prefix is not None:
            x = torch.cat([soft_prefix.to(x.dtype), x], dim=1)
        x = self._run_blocks(self.prelude, x)
        mtp_future = (
            F.silu(self.mtp_proj(x))
            if (self.cfg.use_mtp_routing and self.cfg.mtp_weight > 0)
            else None
        )
        if self.cfg.use_recurrence:
            if r == "converge":
                x = self._run_core_converge(x, mtp_future=mtp_future)
            elif r is not None:
                x = self._run_core_fixed(x, r, mtp_future=mtp_future)
            elif self.cfg.use_halting:
                x, _s, _hd = self._run_core_halting(x, mtp_future=mtp_future)
            else:
                x = self._run_core_sampled(x, idx.size(0), mtp_future=mtp_future)
        else:
            x = self._run_blocks(self.core, x)
        x = self._run_blocks(self.coda, x)
        if self.granary is not None:
            x = x + self.granary(x)  # residual knowledge read (no-op at gate=0)
        x = self.norm_f(x)
        if self.latent_memory is not None:
            x, mem_info = self.latent_memory(x)
            self._last_memory_info = self.latent_memory.telemetry(mem_info)
        return x

    # ── Forward pass ──
    def forward(self, idx, targets=None, r=None, kd=None, prior=None):
        self._core_traj = None  # clear stale trajectory from a prior call
        self._last_convergence = None  # clear stale convergence signal
        self._last_memory_info = None
        x = self._tok_embed(idx)
        x = self._run_blocks(self.prelude, x)
        # NITP: stash the shallow (post-prelude) contextual rep as the self-supervised target.
        shallow = (
            x
            if (self.cfg.use_nitp and self.training and targets is not None and x.size(1) > 1)
            else None
        )
        # MTP future: predicted next-token representation for multi-scale core routing
        mtp_future = (
            F.silu(self.mtp_proj(x))
            if (self.cfg.use_mtp_routing and self.cfg.mtp_weight > 0)
            else None
        )
        ponder_loss = x.new_zeros(())
        if self.cfg.use_recurrence:
            if r == "converge":  # equilibrium effort (inference-only)
                x = self._run_core_converge(x, mtp_future=mtp_future)
            elif r is not None:  # manual effort override (fixed loops)
                x = self._run_core_fixed(x, r, mtp_future=mtp_future)
            elif self.cfg.use_halting:  # adaptive per-token halting
                x, _steps, halt_dist = self._run_core_halting(x, mtp_future=mtp_future)
                if targets is not None and self.training:
                    ponder_loss = self._ponder_loss(halt_dist)
            else:
                x = self._run_core_sampled(x, idx.size(0), mtp_future=mtp_future)
        else:
            x = self._run_blocks(self.core, x)
        x = self._run_blocks(self.coda, x)
        if self.granary is not None:
            x = x + self.granary(x)  # residual knowledge read (no-op at gate=0)
        h = self.norm_f(x)
        mem_info = None
        if self.latent_memory is not None:
            h, mem_info = self.latent_memory(h)
            self._last_memory_info = self.latent_memory.telemetry(mem_info)
        if targets is None:  # inference/eval: full logits, no backward
            hh, Wh = self._head_hw(h)
            logits = self._softcap_logits(F.linear(hh, Wh))  # tied head (factorization-aware)
            conf = torch.sigmoid(self.conf_head(h)).squeeze(-1)  # per-token confidence
            if self.cfg.sngp_enabled:  # distance-aware epistemic variance (A2 sibling)
                self._last_sngp_var = self.sngp_head.variance(h)
            return logits, conf
        # training: fused/chunked heads - never materialize the full (B,T,vocab) logits.
        loss, correct = self._fused_ce(h[:, :-1], targets[:, :-1], want_correct=True, prior=prior)
        # per-component telemetry: record each term's *weighted* contribution (detached, no sync
        # until train.py reads it on a log step) so we can see if any aux loss is starving the main
        # CE on the shared trunk — the cheap first-order interference probe (vs full PCGrad).
        parts = {"ce": loss.detach()}
        if self.cfg.mtp_weight > 0 and idx.size(1) > 2:  # MTP: predict t+2
            hm = F.silu(self.mtp_proj(h[:, :-2]))
            mtp = self.cfg.mtp_weight * self._fused_ce(hm, targets[:, 1:-1])
            loss = loss + mtp
            parts["mtp"] = mtp.detach()
        # confidence/calibration head: predict whether the model's own top-1 is correct.
        # `correct` came (detached) from the chunked main head - a *signal* to trigger the
        # "decompose & delegate" path at inference, not a hard gate.
        conf_logit = self.conf_head(h[:, :-1]).squeeze(-1).reshape(-1)
        conf_l = self.cfg.conf_weight * F.binary_cross_entropy_with_logits(conf_logit, correct)
        loss = loss + conf_l
        parts["conf"] = conf_l.detach()
        if self.cfg.sngp_enabled:  # SNGP GP-mean head shares the is-correct target
            sngp_logit = self.sngp_head.mean_logit(h[:, :-1]).reshape(-1)
            sngp_l = self.cfg.conf_weight * F.binary_cross_entropy_with_logits(sngp_logit, correct)
            loss = loss + sngp_l
            parts["sngp"] = sngp_l.detach()
            if self.cfg.sngp_accumulate_train:
                self.sngp_head.accumulate_precision(h[:, :-1].detach())  # explicit calibration path
        if shallow is not None:  # NITP: cosine-match deep h[t] -> shallow[t+1]
            pred = F.normalize(self.nitp_head(h[:, :-1]), dim=-1)
            tgt_rep = F.normalize(shallow[:, 1:].detach(), dim=-1)
            nitp_l = self.cfg.nitp_weight * (1.0 - (pred * tgt_rep).sum(-1)).mean()
            loss = loss + nitp_l
            parts["nitp"] = nitp_l.detach()
        if kd is not None:  # soft-label distillation from a teacher
            t_idx, t_prob, kd_temp, kd_weight = kd
            kd_h = h[:, :-1]
            if t_idx.size(1) < kd_h.size(1):  # ESR: teacher budget over prefix only
                kd_h = kd_h[:, : t_idx.size(1)]
            kd_l = kd_weight * self._kd_topk(kd_h, t_idx, t_prob, kd_temp)
            loss = loss + kd_l
            parts["kd"] = kd_l.detach()
        if self.cfg.use_deep_supervision and self._core_traj is not None:
            ds_l = self.cfg.deepsup_weight * self._deepsup_loss(self._core_traj, targets)
            loss = loss + ds_l
            parts["deepsup"] = ds_l.detach()
        if self.cfg.use_process_head and self._core_traj is not None:
            ph_l = self.cfg.process_weight * self._process_loss(self._core_traj, correct)
            loss = loss + ph_l
            parts["process"] = ph_l.detach()
        if self.cfg.cross_loop_consistency and self._core_traj is not None:
            cl_l = self.cfg.cross_loop_weight * self._cross_loop_loss(self._core_traj)
            loss = loss + cl_l
            parts["cos_loop"] = cl_l.detach()
        if self.latent_memory is not None and mem_info is not None:
            mem_l, mem_parts = self.latent_memory.loss(mem_info)
            loss = loss + mem_l
            for mk, mv in mem_parts.items():
                parts[mk] = mv.detach()
        ponder_l = self.cfg.ponder_cost * ponder_loss  # PonderNet regularizer
        loss = loss + ponder_l
        parts["ponder"] = ponder_l.detach()
        self._last_loss_parts = parts
        return None, loss  # logits unused in training (saves (B,T,V))

    # ── Auxiliary losses ──
    def _deepsup_loss(self, traj, targets):
        # Anytime supervision: decode the EARLIER grad-carrying core states through the real
        # coda+head and CE them against the target, weighted toward later loops. The final state
        # (traj[-1]) is already the main loss, so we only pay coda on traj[:-1]. This trains every
        # effort level to be a valid answer and makes the loop trajectory converge to the target.
        ds = traj[0].new_zeros(())
        wsum = 0.0
        k = len(traj)
        for j, sj in enumerate(traj[:-1]):
            w = (j + 1) / k  # earlier loops matter less
            hj = self.norm_f(self._run_blocks(self.coda, sj))
            ds = ds + w * self._fused_ce(hj[:, :-1], targets[:, :-1])
            wsum += w
        return ds / max(wsum, 1e-9)

    def _process_loss(self, traj, correct):
        """Train every stored recurrent state to predict final next-token correctness."""
        losses = []
        tgt = correct.view(traj[-1].size(0), traj[-1].size(1) - 1).detach()
        for sj in traj:
            logit = self.process_head(sj[:, :-1]).squeeze(-1)
            losses.append(F.binary_cross_entropy_with_logits(logit.reshape(-1), tgt.reshape(-1)))
        return torch.stack(losses).mean()

    def _cross_loop_loss(self, traj):
        """A1: aux loss maximizing cosine similarity between consecutive core states. Pulls the
        recurrence toward a fixed direction — the trajectory contracts instead of wandering."""
        sims = []
        for i in range(1, len(traj)):
            sim = F.cosine_similarity(traj[i - 1].float(), traj[i].float(), dim=-1).mean()
            sims.append(sim)
        return 1.0 - torch.stack(sims).mean()

    def conf_margin_loss(self, h_good, h_bad, margin=None):
        """RLCM (arXiv:2604.23333): margin-based confidence training for RL stability. Instead of
        plain BCE score-matching (brittle under RL), push conf(correct-prefix) above
        conf(incorrect-prefix) by `margin`. h_good/h_bad: post-norm hidden states (.., d_model) from
        verified-correct vs incorrect prefixes (paired by selfteach/RLVR at the same budget). 0
        inference cost; the conf_head is reused."""
        if margin is None:
            margin = self.cfg.conf_margin
        cg = torch.sigmoid(self.conf_head(h_good).squeeze(-1))
        cb = torch.sigmoid(self.conf_head(h_bad).squeeze(-1))
        return F.relu(margin - (cg - cb)).mean()

    def task_loss(self, reward, targets=None):
        """Advantage-weighted regression from a stored core trajectory + reward signal.
        Call AFTER a task-episode forward pass (which stored _core_traj, _value_states,
        and produced an answer). reward is a scalar tensor (task outcome).

        Returns (total_loss, parts_dict) where total_loss is the sum of:
          - value_loss: MSE(value_pred, reward) for each core state (critic training)
          - awr_policy: advantage-weighted deepsup + cross-loop (actor training)
        The caller calls .backward() on total_loss to flow gradients through the core trajectory."""
        traj = self._core_traj
        values = self._value_states
        if not traj or not values or len(traj) < 2:
            return torch.zeros(()), {}

        # ── Critic: value head predicts reward from each core state ──
        if len(values) != len(traj):
            raise RuntimeError(f"value/traj mismatch: {len(values)} vs {len(traj)}")
        last_value = values[-1].detach()  # baseline, no grad through this
        advantage = reward - last_value.mean()

        value_loss = torch.zeros(())
        for v in values:
            value_loss = value_loss + F.mse_loss(v, reward.expand_as(v))
        value_loss = self.cfg.value_weight * value_loss / len(values)

        # ── Actor: AWR weight on existing trajectory-shaping losses ──
        awr_w = torch.exp(advantage / self.cfg.awr_temperature).clamp(0.1, 10.0)
        policy_loss = torch.zeros(())

        if targets is not None and targets.shape[1] == traj[0].shape[1]:
            # Deep supervision: earlier core states → next-token targets. _deepsup_loss shifts
            # internally (hj[:,:-1] vs targets[:,:-1]), so targets must be the UNSHIFTED sequence
            # whose length matches the core trajectory (ts). The caller already satisfies this:
            # task_forward(full[:,:-1]) makes ts=T-1 and targets=full[:,1:] is also T-1. Do not
            # re-slice targets here: a second [:,1:] makes the length check below always fail and
            # silently disables this branch, leaving only the critic learning.
            ds = self._deepsup_loss(traj, targets)
            policy_loss = policy_loss + awr_w * ds

        if self.cfg.cross_loop_consistency:
            cl = self._cross_loop_loss(traj)
            policy_loss = policy_loss + self.cfg.cross_loop_weight * cl

        total = value_loss + policy_loss
        parts = {
            "value": value_loss.detach(),
            "awr_policy": policy_loss.detach(),
            "advantage": advantage.detach(),
        }
        return total, parts

    def task_forward(self, idx):
        """Forward pass for a task episode: produce answer logits + store trajectory for RL.
        No teacher-forcing CE - the model reads the task and generates an answer.
        Identical to inference except we keep gradients on _core_traj for task_loss()."""
        was_training = self.training
        self.train()  # keep gradients flowing through core
        with torch.enable_grad():
            logits, _conf = self.forward(idx)
        if not was_training:
            self.eval()
        return logits

    # ── Streaming decode engine ────────────────────────────────────────────
    # Every mixer's memory is either a KV window (attention) or a fixed-size state matrix
    # (GDN/GDN-2), so autoregressive decode never needs the prefix again: prefill builds the
    # caches once, then each new token costs one trunk pass over ONE position — O(1)/token
    # instead of the O(T) full-prefix reforward the uncached path pays. The recurrent core gets
    # one cache set PER LOOP (loop n's sequence state differs from loop m's).

    def _decode_supported(self, effort):
        return isinstance(effort, int)

    # ── Decode and generation ──
    def _decode_trunk(self, idx, cache):
        """Run embed -> prelude -> core loops -> coda -> norm_f over the NEW tokens `idx`,
        advancing the streaming caches. Returns post-norm h for the new positions."""
        r = cache["r"]
        rev = getattr(self.cfg, "reversible", False)
        x = self._tok_embed(idx)
        if rev:
            x = _two_stream_blocks(self.prelude, x, caches=list(cache["prelude"]))
        else:
            for blk, c in zip(self.prelude, cache["prelude"]):
                x = blk(x, cache=c)
        if not self.cfg.use_recurrence:
            if rev:
                x = _two_stream_blocks(self.core, x, caches=list(cache["core"][0]))
            else:
                for blk, c in zip(self.core, cache["core"][0]):
                    x = blk(x, cache=c)
        else:
            e = x
            mtp_future = (
                F.silu(self.mtp_proj(e))
                if (self.cfg.use_mtp_routing and self.cfg.mtp_weight > 0)
                else None
            )
            s = torch.zeros_like(e)  # eval init (deterministic zero state)
            z = self.subconscious.init_state(e) if self.subconscious is not None else None
            s_prev = None
            err = None
            for n in range(r):
                self._n_core_steps += 1
                s = self._core_step(s, e, n, err, mtp_future, caches=cache["core"][n])
                if self.subconscious is not None:
                    s, z = self.subconscious(s, z)
            x = s
        if rev:
            x = _two_stream_blocks(self.coda, x, caches=list(cache["coda"]))
        else:
            for blk, c in zip(self.coda, cache["coda"]):
                x = blk(x, cache=c)
        x = self.norm_f(x)
        if self.latent_memory is not None:
            x, _ = self.latent_memory(x, cache=cache.setdefault("latent_memory", {}))
        return x

    @torch.no_grad()
    def decode_prefill(self, idx, effort):
        """Build streaming caches over the prompt. Returns (cache, h_last (B,1,d))."""
        if self.training or not self._decode_supported(effort):
            raise RuntimeError("decode_prefill requires eval mode and fixed-effort cache support")
        r = int(effort) if self.cfg.use_recurrence else 1
        cache = {
            "r": r,
            "prelude": [{} for _ in self.prelude],
            "core": [[{} for _ in self.core] for _ in range(max(r, 1))],
            "coda": [{} for _ in self.coda],
        }
        h = self._decode_trunk(idx, cache)
        return cache, h[:, -1:]

    @torch.no_grad()
    def decode_step(self, tok, cache):
        """Advance the streaming decode by the new token(s) `tok` (B, t_new). O(1) per token."""
        return self._decode_trunk(tok, cache)[:, -1:]

    def _gen_logits(self, ctx, effort, positions=1):
        """Decode-path head read: run the trunk once, build logits ONLY for the last `positions`
        positions instead of the full (B, T, V) tensor forward() returns — at V=131072 the full
        tensor costs ~0.5GB per generated token at T=2048, all but one row discarded. Populates
        _last_sngp_var (serve's epistemic fusion reads it after generate) and returns
        (logits (B, positions, V), conf (B, positions))."""
        h = self.hidden(ctx, r=effort)
        if self.cfg.sngp_enabled:
            self._last_sngp_var = self.sngp_head.variance(h)
        hp = h[:, -positions:]
        hh, W = self._head_hw(hp)
        return self._softcap_logits(F.linear(hh, W)), torch.sigmoid(self.conf_head(hp)).squeeze(-1)

    def _softcap_logits(self, logits):
        cap = float(getattr(self.cfg, "logit_softcap", 0.0))
        return cap * torch.tanh(logits / cap) if cap > 0 else logits

    @staticmethod
    def _warp_logits(logits, temp, top_k, top_p=0.0, min_p=0.0):
        """Shared sampling warp: temperature + top-k/top-p/min-p + non-finite sanitization,
        returning probabilities. temp < 1e-4 means greedy (a one-hot distribution — dividing by
        ~0 instead blows finite logits to ~1e6, softmax overflows to nan, and torch.multinomial
        fires a CUDA device-side assert that corrupts the whole CUDA context). Non-finite logits
        (early training / a hot module) are sanitized for the same reason: sampling degrades to
        "pick the best real token" instead of crashing preflight/serve.
        min_p (keep tokens with p >= min_p * p_max) adapts the candidate set to the model's own
        certainty — measurably better than fixed top-k for small models. top_p is nucleus."""
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
        if temp < 1e-4:
            return F.one_hot(logits.argmax(-1), logits.size(-1)).to(logits.dtype)
        logits = logits / temp
        if top_k:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, -float("inf"))
        if min_p > 0:
            # p/p_max >= min_p  <=>  logit >= logit_max + log(min_p): no softmax needed
            logits = logits.masked_fill(
                logits < logits.max(-1, keepdim=True).values + math.log(min_p), -float("inf")
            )
        if top_p > 0:
            srt, idx_srt = logits.sort(-1, descending=True)
            cum = F.softmax(srt, -1).cumsum(-1)
            drop = cum - F.softmax(srt, -1) >= top_p  # keep the token that crosses top_p
            logits = logits.masked_fill(drop.gather(-1, idx_srt.argsort(-1)), -float("inf"))
        return F.softmax(logits, -1)

    @staticmethod
    def _penalize_repeats(logits, prefix, rep_penalty=1.0, no_repeat_ngram=0, window=256):
        """In-place repetition control on next-token logits (B, V) given the sequence so far.
        rep_penalty (CTRL-style) divides positive / multiplies negative logits of recently seen
        tokens; no_repeat_ngram bans any token that would complete an n-gram already present.
        Small models degenerate into loops far more than large ones —
        repetition control is critical at this scale."""
        if rep_penalty and rep_penalty != 1.0:
            for b in range(prefix.size(0)):
                toks = prefix[b, -window:].unique()
                lb = logits[b, toks]
                logits[b, toks] = torch.where(lb > 0, lb / rep_penalty, lb * rep_penalty)
        n = no_repeat_ngram
        if n and prefix.size(1) >= n:
            for b in range(prefix.size(0)):
                seq = prefix[b, -1024:].tolist()
                tail = tuple(seq[len(seq) - (n - 1) :]) if n > 1 else ()
                banned = [
                    seq[i + n - 1]
                    for i in range(len(seq) - n + 1)
                    if tuple(seq[i : i + n - 1]) == tail
                ]
                if banned:
                    logits[b, banned] = -float("inf")
        return logits

    def _echo_gate(self, logits, prefix, effort, n=3, ctx_len=64, scale=2.0):
        """CONDITIONAL repetition control (B,V next-token logits, in place-ish).
        no_repeat_ngram is task-blind: it bans loop-extending tokens even when the
        task IS repetition (quoting, 'repeat X three times'). This gate instead asks
        WHY the model wants the repeat: it re-scores the same next token from only
        the recent ctx_len-token tail. A token whose full-context log-prob does not
        exceed its tail-only log-prob is driven by local momentum (the signature of
        degenerate loops) and is penalized by scale x the excess; a token the
        DISTANT context promotes above local momentum (an instruction to repeat, a
        quoted span) keeps its logit untouched. Costs one short forward per step,
        and only on steps where a loop-extending candidate actually exists."""
        T = prefix.size(1)
        if T <= ctx_len + n:  # tail == context: gate is undefined
            return logits
        cands, any_hit = [], False
        for b in range(prefix.size(0)):
            seq = prefix[b, -1024:].tolist()
            tail = tuple(seq[len(seq) - (n - 1) :]) if n > 1 else ()
            c = sorted(
                {
                    seq[i + n - 1]
                    for i in range(len(seq) - n + 1)
                    if tuple(seq[i : i + n - 1]) == tail
                }
            )
            cands.append(c)
            any_hit |= bool(c)
        if not any_hit:
            return logits
        r_tail = effort if isinstance(effort, int) else 1
        t_logits, _ = self._gen_logits(prefix[:, -ctx_len:], r_tail)
        lp_full = F.log_softmax(logits.float(), -1)
        lp_tail = F.log_softmax(self._softcap_logits(t_logits[:, -1]).float(), -1)
        for b, c in enumerate(cands):
            if not c:
                continue
            ci = torch.as_tensor(c, device=logits.device)
            support = lp_full[b, ci] - lp_tail[b, ci]  # >0: distant context earns it
            logits[b, ci] -= (scale * (-support).clamp(min=0.0)).to(logits.dtype)
        return logits

    def _contrast_scores(self, logits, mini_logits, lam, alpha=0.1):
        """Contrastive decoding: promote what the strong model believes BEYOND what a small
        fluency 'amateur' already predicts (score = logp_big - lam*logp_mini), restricted to
        the plausibility set p_big >= alpha * max p_big so the contrast can never surface a
        token the big model itself rejects. Measurably improves small-model generation quality
        (coherence/factuality) at the cost of one cheap mini forward per token."""
        lp = F.log_softmax(logits.float(), -1)
        lp_m = F.log_softmax(mini_logits.float(), -1)
        keep = lp >= lp.max(-1, keepdim=True).values + math.log(alpha)
        return (lp - lam * lp_m).masked_fill(~keep, -float("inf"))

    @torch.no_grad()
    def generate(
        self,
        idx,
        n_new,
        effort=None,
        temp=0.8,
        top_k=50,
        return_conf=False,
        draft_effort=None,
        draft_len=4,
        use_cache=True,
        top_p=0.0,
        min_p=0.0,
        rep_penalty=1.0,
        no_repeat_ngram=0,
        contrast=None,
        seed_noise=0.0,
        loop_contrast=0.0,
        loop_contrast_effort=1,
        loop_contrast_alpha=0.1,
        echo_gate=0.0,
        echo_ngram=3,
        echo_ctx=64,
    ):
        # effort: None -> adaptive halting picks loops per token; int -> fixed manual loops;
        #         'converge' -> equilibrium mode (iterate the core to a fixed point).
        # return_conf: additionally return the conf-head value at each generated position
        # (B, n_new) — serve's escalation loop reuses these instead of paying a SECOND full
        # forward pass over the finished sequence just to re-read the same head.
        # draft_effort: opt-in SELF-SPECULATIVE decoding via the effort dial — see _generate_spec.
        # use_cache: streaming O(1)/token decode (fixed-effort only); falls back to full
        # reforwards for adaptive halting / converge / overflow past max_seq_len.
        # top_p/min_p/rep_penalty/no_repeat_ngram: sampling quality controls (see the warps).
        # contrast: (mini_model, lam[, alpha]) — contrastive decoding vs a small fluency model.
        # loop_contrast: DoLa-style contrast, but across RECURRENT effort instead of transformer
        # layers: score = logp(effort) - lam*logp(low_effort), on the plausible set. This amplifies
        # signal that appears only after extra thinking, without another model checkpoint.
        self.eval()
        self._seed_noise = seed_noise  # latent-seed diversity (see _initial_core_state)
        try:
            if draft_effort is not None:
                return self._generate_spec(
                    idx,
                    n_new,
                    effort,
                    temp,
                    top_k,
                    return_conf,
                    draft_effort,
                    draft_len,
                    top_p=top_p,
                    min_p=min_p,
                    rep_penalty=rep_penalty,
                    no_repeat_ngram=no_repeat_ngram,
                )
            c_model, c_lam, c_alpha = (None, 0.0, 0.1)
            if contrast is not None:
                c_model = contrast[0].eval()
                c_lam = contrast[1]
                c_alpha = contrast[2] if len(contrast) > 2 else 0.1
            lc_lam = float(loop_contrast or 0.0)
            lc_effort = max(1, int(loop_contrast_effort or 1))
            lc_alpha = float(loop_contrast_alpha or 0.1)

            def pick(logits, mini_logits=None, loop_logits=None):
                logits = self._softcap_logits(logits)
                if lc_lam > 0 and loop_logits is not None:
                    logits = self._contrast_scores(logits, loop_logits, lc_lam, lc_alpha)
                if c_model is not None and mini_logits is not None:
                    logits = self._contrast_scores(logits, mini_logits, c_lam, c_alpha)
                if echo_gate > 0:  # conditional anti-loop (see _echo_gate)
                    logits = self._echo_gate(
                        logits, idx, effort, n=echo_ngram, ctx_len=echo_ctx, scale=echo_gate
                    )
                self._penalize_repeats(logits, idx, rep_penalty, no_repeat_ngram)
                probs = self._warp_logits(logits, temp, top_k, top_p, min_p)
                return (
                    probs.argmax(-1, keepdim=True) if temp < 1e-4 else torch.multinomial(probs, 1)
                )

            confs = [] if return_conf else None
            if (
                use_cache
                and self._decode_supported(effort)
                and idx.size(1) > 0
                and idx.size(1) + n_new <= self.cfg.max_seq_len
                and seed_noise == 0.0
            ):
                cache, h = self.decode_prefill(idx, effort)  # decode trunk seeds s=0 (no noise)
                mc, hm = (
                    c_model.decode_prefill(idx, 1)
                    if (c_model is not None and c_model._decode_supported(1))
                    else (None, None)
                )
                use_lc_cache = (
                    lc_lam > 0 and lc_effort != int(effort) and self._decode_supported(lc_effort)
                )
                lc_cache, lh = self.decode_prefill(idx, lc_effort) if use_lc_cache else (None, None)
                for i in range(n_new):
                    if i > 0:
                        h = self.decode_step(idx[:, -1:], cache)
                        if c_model is not None and mc is not None:
                            hm = c_model.decode_step(idx[:, -1:], mc)
                        if use_lc_cache:
                            lh = self.decode_step(idx[:, -1:], lc_cache)
                    if self.cfg.sngp_enabled:
                        self._last_sngp_var = self.sngp_head.variance(h)
                    if return_conf:
                        confs.append(torch.sigmoid(self.conf_head(h[:, -1])).squeeze(-1))
                    hh, W = self._head_hw(h)
                    ml = None
                    if c_model is not None:
                        if hm is not None:
                            mh, mW = c_model._head_hw(hm)
                            ml = c_model._softcap_logits(F.linear(mh, mW))[:, -1]
                        else:
                            ml = c_model._gen_logits(idx[:, -self.cfg.max_seq_len :], 1)[0][:, -1]
                    ll = None
                    if use_lc_cache:
                        lhh, lW = self._head_hw(lh)
                        ll = self._softcap_logits(F.linear(lhh, lW))[:, -1]
                    idx = torch.cat([idx, pick(F.linear(hh, W)[:, -1], ml, ll)], 1)
            else:
                for _ in range(n_new):
                    ctx = idx[:, -self.cfg.max_seq_len :]
                    logits, conf = self._gen_logits(ctx, effort)
                    if return_conf:
                        confs.append(conf[:, -1])
                    ml = c_model._gen_logits(ctx, 1)[0][:, -1] if c_model is not None else None
                    ll = (
                        self._gen_logits(ctx, lc_effort)[0][:, -1]
                        if lc_lam > 0 and effort != lc_effort
                        else None
                    )
                    idx = torch.cat([idx, pick(logits[:, -1], ml, ll)], 1)
            if return_conf:
                conf_t = (
                    torch.stack(confs, 1)
                    if confs
                    else idx.new_zeros((idx.size(0), 0), dtype=torch.float32)
                )
                return idx, conf_t
            return idx
        finally:
            self._seed_noise = 0.0

    @torch.no_grad()
    def _generate_spec(
        self,
        idx,
        n_new,
        effort,
        temp,
        top_k,
        return_conf,
        draft_effort,
        draft_len,
        top_p=0.0,
        min_p=0.0,
        rep_penalty=1.0,
        no_repeat_ngram=0,
    ):
        """Self-speculative decoding through the effort dial (batch 1). The DRAFT model is this
        same model at low recurrence (draft_effort, e.g. r=1) and the TARGET is it at full effort
        (or adaptive halting when effort=None) — the recurrence depth dial yields a free,
        perfectly-vocabulary-aligned draft/target pair with zero extra weights, which a fixed-depth
        transformer cannot do. Drafts draft_len tokens cheaply, then verifies them ALL in ONE
        full-effort forward (token-parallel, so the expensive deep pass is amortized), accepting
        each with the standard speculative-sampling rule p_target/p_draft and resampling from the
        residual max(0, p_t - p_d) on the first rejection — the output distribution is exactly the
        full-effort model's. At temp=0 this reduces to "accept while the cheap and deep argmax
        agree", so greedy output is bit-identical to plain greedy decoding at full effort."""
        if idx.size(0) != 1:
            raise ValueError("self-speculative decoding supports batch 1 (serve/eval decode)")
        confs = [] if return_conf else None
        done = 0

        def warp_at(logits, prefix):
            # SAME warp (penalties + truncations) for draft q and target p at every position:
            # identical warping is what keeps the acceptance rule exact, and it makes spec-mode
            # sampling distribution-identical to plain decoding with the same knobs.
            lg = logits.clone()
            self._penalize_repeats(lg, prefix, rep_penalty, no_repeat_ngram)
            return self._warp_logits(lg, temp, top_k, top_p, min_p)

        while done < n_new:
            k = min(draft_len, n_new - done)
            base_len = idx.size(1)
            # 1) draft k tokens autoregressively at cheap effort, remembering q(draft token)
            d_idx = idx
            q_probs = []
            for _ in range(k):
                dl, _ = self._gen_logits(d_idx[:, -self.cfg.max_seq_len :], draft_effort)
                qp = warp_at(dl[:, -1], d_idx)
                nxt = qp.argmax(-1, keepdim=True) if temp < 1e-4 else torch.multinomial(qp, 1)
                q_probs.append(qp)
                d_idx = torch.cat([d_idx, nxt], 1)
            drafted = d_idx[:, idx.size(1) :]  # (1, k)
            # 2) ONE full-effort verify pass over context + drafts: logits at the position BEFORE
            # each drafted token (its target distribution) plus one bonus position at the end.
            vl, vconf = self._gen_logits(d_idx[:, -self.cfg.max_seq_len :], effort, positions=k + 1)
            n_acc = 0
            for i in range(k):
                pp = warp_at(vl[:, i], d_idx[:, : base_len + i])  # target dist for token i
                tok = drafted[:, i : i + 1]
                p_t = pp.gather(-1, tok).squeeze()
                q_t = q_probs[i].gather(-1, tok).squeeze()
                # strict <: u ~ U[0,1) can be exactly 0.0, which must NOT accept a p_t=0 token
                if torch.rand((), device=idx.device) < p_t / q_t.clamp_min(1e-20):
                    idx = torch.cat([idx, tok], 1)
                    if return_conf:
                        confs.append(vconf[:, i])
                    n_acc += 1
                    done += 1
                    if done >= n_new:
                        break
                else:
                    # resample from the residual distribution max(0, p - q), renormalized —
                    # this correction is what makes acceptance exact, not approximate.
                    resid = (pp - q_probs[i]).clamp_min(0)
                    resid = resid / resid.sum(-1, keepdim=True).clamp_min(1e-20)
                    idx = torch.cat([idx, torch.multinomial(resid, 1)], 1)
                    if return_conf:
                        confs.append(vconf[:, i])
                    done += 1
                    break
            else:
                if done < n_new:  # all k accepted: free bonus
                    bp = warp_at(vl[:, k], d_idx)
                    nxt = bp.argmax(-1, keepdim=True) if temp < 1e-4 else torch.multinomial(bp, 1)
                    idx = torch.cat([idx, nxt], 1)
                    if return_conf:
                        confs.append(vconf[:, k])
                    done += 1
        if return_conf:
            conf_t = (
                torch.stack(confs, 1)
                if confs
                else idx.new_zeros((idx.size(0), 0), dtype=torch.float32)
            )
            return idx, conf_t
        return idx


# --------------------------------------------------------------------------
# Muon optimizer (Jordan et al., modded-nanogpt; validated at 1T scale as
# MuonClip in Kimi K2). Applied to 2D hidden weights; AdamW handles the rest.
# --------------------------------------------------------------------------

__all__ = ["Charkha", "LaplaceConf", "SNGPHead"]
