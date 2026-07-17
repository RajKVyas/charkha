"""CHARKHA modules — GDN, attention, MLP, normalisation, aux heads."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext as _nullcontext
from .config import CharkhaConfig

# fla Triton kernel imports
try:
    import fla.ops.gated_delta_rule as _gdr

    _HAVE_FLA = True
except Exception:
    _gdr = None
    _HAVE_FLA = False
try:
    from fla.ops.gdn2 import chunk_gdn2 as _chunk_gdn2
    from fla.ops.gdn2 import fused_recurrent_gdn2 as _fused_recurrent_gdn2

    _HAVE_FLA_GDN2 = True
except Exception:
    _chunk_gdn2 = None
    _fused_recurrent_gdn2 = None
    _HAVE_FLA_GDN2 = False


def _dynamo_disable(fn):
    compiler_disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if compiler_disable is not None:
        try:
            return compiler_disable(fn, reason="fla Triton backward is not compile-safe here")
        except TypeError:  # older public API: no reason kwarg
            return compiler_disable(fn)
    dynamo_disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    return dynamo_disable(fn) if dynamo_disable is not None else fn


@_dynamo_disable
def _fla_chunk_gdn2(*a, **k):
    return _chunk_gdn2(*a, **k)


@_dynamo_disable
def _fla_fused_recurrent_gdn2(*a, **k):
    return _fused_recurrent_gdn2(*a, **k)


@_dynamo_disable
def _fla_chunk_gated_delta_rule(*a, **k):
    return _gdr.chunk_gated_delta_rule(*a, **k)


def have_fla() -> bool:
    """True iff the flash-linear-attention / Triton Gated-DeltaNet kernel is importable — the
    production GDN path. When False (no triton, e.g. native Windows), GDN runs `_gdn_chunk_scan`:
    correct and selftest-exact, but slower and numerically distinct from the kernel. Real training
    should run where this is True (WSL2 + CUDA); the fallback is for inference/dev/selftests."""
    return _HAVE_FLA


# PyTorch's scaled_dot_product_attention auto-picks a CUDA backend (flash / mem-efficient / math)
# per call. With gradient checkpointing, the FORWARD call and backward's RECOMPUTE call are two
# separate dispatches — if the heuristic ever picks a different backend between them (observed on
# Turing-class GPUs, e.g. Kaggle T4, where the native flash kernel's support is narrow/inconsistent),
# the recompute saves a different set of tensors than the original forward did, and
# torch.utils.checkpoint raises "Recomputed values ... different metadata" (saved/recomputed shapes
# and dtypes scrambled, since the tensor list itself is misaligned). Pinning OFF the flash backend
# for training forces the same (efficient/math) kernel on every call, which removes the ambiguity at
# a small speed cost. Inference (no checkpoint, no recompute) is unaffected and keeps flash.

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel as _sdpa_kernel

    def _stable_sdpa_ctx():
        return _sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
except ImportError:  # older torch: context-manager flag API

    def _stable_sdpa_ctx():
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=True
        )

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


_ROPE_CACHE: dict = {}


def rope_cache(T, dim, base, device):
    # cos/sin for length T are just the first T rows of the table for any length >= T, so cache
    # ONE table per (dim, base, device) sized to the largest T seen and slice it. Keying on exact
    # T (as before) leaked a fresh full table for every sequence length: during autoregressive
    # generation T grows by one each token, so a long decode at max_seq_len piled up O(max_seq_len)
    # tables (~GBs of trig at 4096) on the GPU. The recurrent core still re-runs SWA attention many
    # times per forward, so memoization across loops matters — this keeps that, bounded.
    key = (dim, base, str(device))
    hit = _ROPE_CACHE.get(key)
    if hit is not None and hit[0].size(0) >= T:
        return hit[0][:T], hit[1][:T]
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(T, device=device).float()
    f = torch.outer(t, inv)
    cos, sin = torch.cos(f), torch.sin(f)
    _ROPE_CACHE[key] = (cos, sin)
    return cos, sin


def apply_rope(x, cos, sin):  # x: (B,H,T,dh)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos.to(x.dtype)[None, None], sin.to(x.dtype)[None, None]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


# --------------------------------------------------------------------------
# Attention (GQA + QK-RMSNorm + RoPE + sliding window). 1/4 of layers.
# --------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, cfg: CharkhaConfig, window):
        super().__init__()
        self.cfg, self.window = cfg, window  # window=None -> global causal
        d, dh = cfg.d_model, cfg.head_dim
        self.wq = nn.Linear(d, cfg.n_heads * dh, bias=False)
        self.wk = nn.Linear(d, cfg.n_kv_heads * dh, bias=False)
        self.wv = nn.Linear(d, cfg.n_kv_heads * dh, bias=False)
        self.wo = nn.Linear(cfg.n_heads * dh, d, bias=False)
        self.qnorm, self.knorm = RMSNorm(dh), RMSNorm(dh)

    def forward(self, x, cache=None):
        # cache (streaming decode): {'k','v': (B,kvH,K,dh) post-RoPE keys/values, 'pos': absolute
        # position of the next token}. New tokens attend over cached + new keys with absolute
        # positions, so incremental decode is exact vs a full forward. Sliding-window layers prune
        # the cache to the window, keeping decode memory O(window); global layers keep everything.
        B, T, _ = x.shape
        cfg, dh = self.cfg, self.cfg.head_dim
        q = self.wq(x).view(B, T, cfg.n_heads, dh).transpose(1, 2)
        k = self.wk(x).view(B, T, cfg.n_kv_heads, dh).transpose(1, 2)
        v = self.wv(x).view(B, T, cfg.n_kv_heads, dh).transpose(1, 2)
        q, k = self.qnorm(q), self.knorm(k)  # QK-norm
        pos0 = cache.get("pos", 0) if cache is not None else 0
        cos, sin = rope_cache(pos0 + T, dh, cfg.rope_base, x.device)
        cos, sin = cos[pos0 : pos0 + T], sin[pos0 : pos0 + T]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], 2)
                v = torch.cat([cache["v"], v], 2)
            # attend with the FULL key set (early prefill queries need pre-window keys);
            # prune only the stored copy — future queries never look past the window.
            if self.window is not None and k.size(2) > self.window:
                cache["k"], cache["v"] = k[:, :, -self.window :], v[:, :, -self.window :]
            else:
                cache["k"], cache["v"] = k, v
            cache["pos"] = pos0 + T
        rep = cfg.n_heads // cfg.n_kv_heads  # GQA expand
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        # Pin the SDPA backend during training+grad_checkpoint so forward and the backward
        # recompute can't disagree on which kernel ran (see _stable_sdpa_ctx above).
        pin = cfg.grad_checkpoint and self.training and torch.is_grad_enabled() and x.is_cuda
        ctx = _stable_sdpa_ctx() if pin else _nullcontext()
        with ctx:
            if cache is None and (self.window is None or self.window >= T):
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                K = k.size(2)
                q_abs = pos0 + torch.arange(T, device=x.device)  # absolute query positions
                k_abs = (pos0 + T) - K + torch.arange(K, device=x.device)  # keys end at last query
                keep = q_abs[:, None] >= k_abs[None, :]
                if self.window is not None:
                    keep = keep & (q_abs[:, None] - k_abs[None, :] < self.window)
                o = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        return self.wo(o.transpose(1, 2).reshape(B, T, -1))


# --------------------------------------------------------------------------
# Gated DeltaNet (reference, sequential scan). 3/4 of layers.
#   S_t = a_t * S_{t-1} (I - b_t k_t k_t^T) + b_t v_t k_t^T ;  o_t = S_t q_t
# a_t in (0,1): Mamba-style forget gate. b_t in (0,1): delta-rule write strength.
# Production: replace with fla.ops.gated_delta_rule chunked kernel (identical math).
# --------------------------------------------------------------------------


def _gdn_sequential_ref(q, k, v, g, beta):
    # The plain O(T) recurrence, kept as the correctness reference for the chunked scan's selftest.
    # q,k,v: (B,T,H,dh); g,beta: (B,T,H). Returns o: (B,T,H,dh). State S is (B,H,v,k), fp32.
    B, T, H, dh = q.shape
    qf, kf, vf, gf, bf = q.float(), k.float(), v.float(), g.float(), beta.float()
    S = q.new_zeros(B, H, dh, dh, dtype=torch.float32)
    outs = []
    for t in range(T):
        kt, vt, qt = kf[:, t], vf[:, t], qf[:, t]
        gt = gf[:, t, :, None, None]
        bt = bf[:, t, :, None]
        S = S * gt.exp()
        delta = bt * (vt - torch.einsum("bhvk,bhk->bhv", S, kt))
        S = S + torch.einsum("bhv,bhk->bhvk", delta, kt)
        outs.append(torch.einsum("bhvk,bhk->bhv", S, qt))
    return torch.stack(outs, 1)


def _gdn_chunk_scan(q, k, v, g, beta, chunk=32, S0=None, return_state=False):
    # Chunkwise gated delta rule — math-identical to _gdn_sequential_ref but with O(T/chunk)
    # sequential steps instead of O(T). Within a chunk, the recurrence is solved in closed form:
    # the per-step "write" vectors U satisfy a unit-lower-triangular system (I+M)U=W (the delta
    # rule's interactions), after which outputs and the carried state are batched matmuls. All decay
    # ratios are exp(Δcumlogdecay) ≤ 1, so no 1/γ blow-up. Pure PyTorch (CPU/GPU), autograd-safe.
    # This is the no-triton path — used whenever fla is unavailable (always on native Windows).
    # S0/return_state: carried recurrent state for streaming decode — pass the previous call's
    # state and the scan continues the sequence exactly (the mixer's whole memory is this matrix).
    B, T, H, dh = q.shape
    dev, dt = q.device, torch.float32
    q = q.float().transpose(1, 2)
    k = k.float().transpose(1, 2)
    v = v.float().transpose(1, 2)
    g = g.float().transpose(1, 2)
    beta = beta.float().transpose(1, 2)  # (B,H,T[,dh])
    S = (
        S0.clone() if S0 is not None else torch.zeros(B, H, dh, dh, device=dev, dtype=dt)
    )  # state (v,k)
    z = torch.zeros((), device=dev, dtype=dt)
    outs = []
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        C = c1 - c0
        qc, kc, vc = q[:, :, c0:c1], k[:, :, c0:c1], v[:, :, c0:c1]  # (B,H,C,dh)
        gc, bc = g[:, :, c0:c1], beta[:, :, c0:c1]  # (B,H,C)
        lg = gc.cumsum(-1)  # log γ_t in chunk (≤0)
        gamma = lg.exp()
        D = torch.where(
            torch.tril(torch.ones(C, C, device=dev, dtype=torch.bool)),
            (lg.unsqueeze(-1) - lg.unsqueeze(-2)).exp(),
            z,
        )  # D[t,j]=γ_t/γ_j, t≥j
        strict = torch.tril(torch.ones(C, C, device=dev, dtype=torch.bool), -1)
        M = bc.unsqueeze(-1) * torch.where(strict, D * (kc @ kc.transpose(-1, -2)), z)
        KS0 = kc @ S.transpose(-1, -2)  # S₀k_t  → (B,H,C,dh)
        W = bc.unsqueeze(-1) * (vc - gamma.unsqueeze(-1) * KS0)
        U = torch.linalg.solve_triangular(
            torch.eye(C, device=dev, dtype=dt) + M, W, upper=False, unitriangular=True
        )
        O = gamma.unsqueeze(-1) * (qc @ S.transpose(-1, -2)) + (D * (qc @ kc.transpose(-1, -2))) @ U
        outs.append(O)
        cscale = (lg[:, :, -1:] - lg).exp().unsqueeze(-1)  # γ_C/γ_j  (≤1)
        S = gamma[:, :, -1:].unsqueeze(-1) * S + (U * cscale).transpose(-1, -2) @ kc
    o = torch.cat(outs, 2).transpose(1, 2)  # (B,T,H,dh)
    return (o, S) if return_state else o


# --------------------------------------------------------------------------
# GDN-2 (arXiv:2605.22791): the delta rule with a channel-wise erase gate b_t (on the KEY axis) and a
# channel-wise write gate w_t (on the value axis), decoupled, plus channel-wise (per-KEY-dim) decay g_t.
# State S[k,v]; per step:
#   S = exp(g_t)[k] ⊙ S                      (decay on key axis)
#   e_read[v] = Σ_k (b_t[k]·k_t[k]) S[k,v]   (erase readback with the b-GATED KEY ē=b⊙k — b is INSIDE
#                                             the k-contraction, which is what keeps the chunk solve
#                                             a single shared (C,C) system, not per-value-channel)
#   u_t[v] = w_t[v]·v_t[v] − e_read[v]       (write gate on value; erase already applied via e_read)
#   S[k,v] += k_t[k]·u_t[v] ;   o_t[v] = Σ_k S[k,v] q_t[k]   (output AFTER write)
# Reduces to Gated DeltaNet when b collapses to a scalar and decay is per-head scalar.
# --------------------------------------------------------------------------


def _gdn2_sequential_ref(q, k, v, b, w, g):
    # Plain O(T) recurrence — correctness reference for _gdn2_chunk_scan's selftest.
    # q,k,v,b,w,g: (B,T,H,dh). Returns o: (B,T,H,dh). State S is (B,H,k,v), fp32.
    B, T, H, dh = q.shape
    q, k, v, b, w, g = (t.float().transpose(1, 2) for t in (q, k, v, b, w, g))  # (B,H,T,dh)
    S = q.new_zeros(B, H, dh, dh)
    outs = []
    for t in range(T):
        kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
        bt, wt, gt = b[:, :, t], w[:, :, t], g[:, :, t]
        S = S * gt.exp().unsqueeze(-1)  # decay on key axis
        e_read = torch.einsum("bhk,bhkv->bhv", bt * kt, S)  # erase readback: b-GATED KEY (b⊙k)
        u = wt * vt - e_read  # write gate on value; erase via e_read
        S = S + torch.einsum("bhk,bhv->bhkv", kt, u)
        outs.append(torch.einsum("bhkv,bhk->bhv", S, qt))
    return torch.stack(outs, 2).transpose(1, 2)  # (B,T,H,dh)


def _gdn2_chunk_scan(q, k, v, b, w, g, chunk=32, S0=None, return_state=False):
    # Chunkwise-parallel GDN-2 (arXiv:2605.22791 WY form) — math-identical to _gdn2_sequential_ref,
    # O(T/chunk) sequential steps. The erase gate b is on the KEY axis (ē=b⊙k), so the erase readback
    # ēᵀS contracts over k and the in-chunk interaction T[t,j]=Σ_k b_t[k]k_t[k]·k_j[k]·exp(L_t[k]−L_j[k])
    # is a SCALAR (C,C) matrix SHARED across all value channels — one unit-lower-triangular solve
    # (I+T)U = w⊙v − (b⊙k-read of incoming state) per chunk, exactly like GDN-1's single scalar-β
    # solve (NOT a per-channel solve). L = cumlog-decay; all decay factors exp(Δcumlogdecay) ≤ 1 (no
    # 1/γ blow-up); the strict-upper triangle is masked to −∞ before exp so it is exactly 0 (avoids
    # inf·0 = NaN under strong decay). Pure PyTorch (CPU/GPU), autograd-safe, fp32.
    B, T, H, dh = q.shape
    dev = q.device
    with torch.autocast(device_type=dev.type, enabled=False):
        q, k, v, b, w, g = (t.float().transpose(1, 2) for t in (q, k, v, b, w, g))  # (B,H,T,dh)
        S = S0.clone() if S0 is not None else q.new_zeros(B, H, dh, dh)  # state S[k,v]
        outs = []
        masks = _Gdn2Masks(dev, q.dtype)
        for c0 in range(0, T, chunk):
            c1 = min(c0 + chunk, T)
            O, S = _gdn2_one_chunk(
                S,
                q[:, :, c0:c1],
                k[:, :, c0:c1],
                v[:, :, c0:c1],
                b[:, :, c0:c1],
                w[:, :, c0:c1],
                g[:, :, c0:c1],
                masks,
            )
            outs.append(O)
        o = torch.cat(outs, 2).transpose(1, 2)  # (B,T,H,dh)
        return (o, S) if return_state else o


class _Gdn2Masks:
    """Per-chunk-length cache of the triangular masks the GDN-2 chunk math reuses every chunk."""

    def __init__(self, dev, dtype):
        self.dev, self.dtype, self.cache = dev, dtype, {}

    def __call__(self, C):
        cached = self.cache.get(C)
        if cached is None:
            ones_bool = torch.ones(C, C, device=self.dev, dtype=torch.bool)
            lower = torch.tril(torch.ones(C, C, device=self.dev, dtype=self.dtype))
            cached = (
                torch.triu(ones_bool, 1),
                torch.tril(torch.ones(C, C, device=self.dev, dtype=self.dtype), -1),
                lower,
                torch.eye(C, device=self.dev, dtype=self.dtype),
            )
            self.cache[C] = cached
        return cached


def _gdn2_chunk_mats(qc, kc, vc, bc, wc, gc, masks):
    """Shared per-chunk quantities of the GDN-2 WY form (see _gdn2_chunk_scan docstring).
    Everything here depends ONLY on the chunk's inputs — not on the carried state — which is what
    makes both the forward step (_gdn2_one_chunk) and the closed-form state INVERSION
    (_gdn2_invert_chunk) able to share it."""
    upper, strict_lower, lower, eye = masks(qc.size(2))
    Lc = gc.cumsum(2)  # inclusive cumlog-decay ≤0 (B,H,C,dh)
    Ek = Lc.exp()  # exp(L_t[k]) ≤1
    diff = Lc.unsqueeze(3) - Lc.unsqueeze(2)  # L_t[k]−L_j[k] (B,H,C,C,dh)
    diff = diff.masked_fill(upper[None, None, :, :, None], float("-inf"))
    ratio = diff.exp()  # 0 in strict upper, ≤1 elsewhere
    bk = bc * kc  # b-gated key ē=b⊙k (B,H,C,dh)
    # T[t,j]=Σ_k (b_t k_t)[k] k_j[k] exp(L_t−L_j) (strict lower) — SCALAR, shared over all v;
    # B_qk uses q_t (lower-incl) for the in-chunk output contribution
    Tm = (bk.unsqueeze(3) * kc.unsqueeze(2) * ratio).sum(-1) * strict_lower
    Bqk = (qc.unsqueeze(3) * kc.unsqueeze(2) * ratio).sum(-1) * lower
    Lend = Lc[:, :, -1:, :]  # L_C[k] (B,H,1,dh)
    cscale = (Lend - Lc).exp()  # exp(L_C−L_j) ≤1
    return Ek, bk, Tm, Bqk, Lend, cscale, eye


def _gdn2_one_chunk(S, qc, kc, vc, bc, wc, gc, masks):
    """One GDN-2 chunk step: state in -> (chunk outputs, state out). Math identical to the
    original inline loop body of _gdn2_chunk_scan (see its docstring for the derivation)."""
    Ek, bk, Tm, Bqk, Lend, cscale, eye = _gdn2_chunk_mats(qc, kc, vc, bc, wc, gc, masks)
    ePS0 = torch.einsum("bhtk,bhkv->bhtv", bk * Ek, S)  # b-gated-key read of incoming state
    rhs = wc * vc - ePS0  # RHS of the in-chunk solve
    # (I + T) U = rhs  — ONE shared unit-lower-tri solve, all value channels as RHS columns
    U = torch.linalg.solve_triangular(
        eye + Tm, rhs, upper=False, unitriangular=True
    )  # (B,H,C,dh_v)
    O = torch.einsum("bhtk,bhkv->bhtv", qc * Ek, S) + torch.einsum("bhtj,bhjv->bhtv", Bqk, U)
    # carry: S_new[k,v] = exp(L_C[k])·S[k,v] + Σ_j k_j[k]·exp(L_C[k]−L_j[k])·u_j[v]
    S_new = Lend.squeeze(2).exp().unsqueeze(-1) * S + torch.einsum(
        "bhjk,bhjv->bhkv", kc * cscale, U
    )
    return O, S_new


def _gdn2_invert_chunk(S_end, qc, kc, vc, bc, wc, gc, masks):
    """Closed-form CHUNK-LEVEL inversion of the GDN-2 recurrence: recover the state at the chunk's
    START from the state at its END plus the chunk inputs. Novel piece of the reversible-BPTT path
    (no prior art found for delta-rule linear attention as of 2026-07).

    Derivation: the carry is linear in the incoming state S. With A = I+T (the WY solve matrix),
    P = ē⊙exp(L) (state-read rows), C = k⊙exp(L_C−L) (carry rows), WV = w⊙v:
        S_end = diag(exp(L_C))·S + Cᵀ A⁻¹ (WV − P·S)
      → [diag(exp(L_C)) − Cᵀ A⁻¹ P] · S = S_end − Cᵀ A⁻¹ WV
    i.e. one (dh_k × dh_k) batched solve per (B,H) — cheap next to the chunk forward itself.

    Conditioning: exp(L_C) → 0 under deep decay and the erase term can approach it — exactly why
    the reversible scan keeps sparse ANCHOR states (rev_anchor) to reset accumulated error."""
    Ek, bk, Tm, Bqk, Lend, cscale, eye = _gdn2_chunk_mats(qc, kc, vc, bc, wc, gc, masks)
    A = eye + Tm
    P = bk * Ek  # (B,H,C,dh_k)
    Ck = kc * cscale  # (B,H,C,dh_k)
    AinvP = torch.linalg.solve_triangular(A, P, upper=False, unitriangular=True)
    AinvWV = torch.linalg.solve_triangular(A, wc * vc, upper=False, unitriangular=True)
    M = torch.diag_embed(Lend.squeeze(2).exp()) - torch.einsum(
        "bhjk,bhjm->bhkm", Ck, AinvP
    )  # (B,H,dh_k,dh_k)
    RHS = S_end - torch.einsum("bhjk,bhjv->bhkv", Ck, AinvWV)
    return torch.linalg.solve(M, RHS)


class _RevGDN2Scan(torch.autograd.Function):
    """Reversible-recurrence BPTT for the GDN-2 chunk scan (cfg.rev_bptt).

    Forward runs the chunk scan under no_grad, storing NO per-chunk states — only sparse anchor
    states every `anchor` chunks (plus the inputs, which vanilla autograd would keep anyway).
    Backward walks anchor segments in REVERSE, rebuilding each segment's chunk-start states by a
    no-grad replay from its anchor, then re-running each chunk with grad — checkpoint-equivalent
    recompute, O(anchors) stored states instead of O(T/chunk), transient bounded by `anchor`.
    The closed-form state inversion (_gdn2_invert_chunk) is exact algebra but measured
    ill-conditioned under production decay (the transition contracts S per step; inverting
    across chunks amplifies error exponentially — see the inversion selftest), so replay is the
    reconstruction of record and the inversion remains a characterized tool."""

    @staticmethod
    def forward(ctx, q, k, v, b, w, g, chunk, anchor):
        B, T, H, dh = q.shape
        in_dtype = q.dtype
        with torch.no_grad():
            qf, kf, vf, bf, wf, gf = (
                t.detach().float().transpose(1, 2) for t in (q, k, v, b, w, g)
            )  # (B,H,T,dh)
            masks = _Gdn2Masks(q.device, torch.float32)
            S = qf.new_zeros(B, H, dh, dh)
            outs, anchors = [], {0: S}
            bounds = list(range(0, T, chunk))
            for ci, c0 in enumerate(bounds):
                c1 = min(c0 + chunk, T)
                if ci % max(1, anchor) == 0:
                    anchors[ci] = S
                O, S = _gdn2_one_chunk(
                    S,
                    qf[:, :, c0:c1],
                    kf[:, :, c0:c1],
                    vf[:, :, c0:c1],
                    bf[:, :, c0:c1],
                    wf[:, :, c0:c1],
                    gf[:, :, c0:c1],
                    masks,
                )
                outs.append(O)
            o = torch.cat(outs, 2)
        ctx.save_for_backward(qf, kf, vf, bf, wf, gf, S, *[anchors[i] for i in sorted(anchors)])
        ctx.meta = (chunk, anchor, sorted(anchors), in_dtype)
        return o.transpose(1, 2).to(in_dtype)  # (B,T,H,dh)

    @staticmethod
    def backward(ctx, dout):
        qf, kf, vf, bf, wf, gf, S_final, *anchor_states = ctx.saved_tensors
        chunk, anchor, anchor_idx, in_dtype = ctx.meta
        anchors = dict(zip(anchor_idx, anchor_states))
        B, H, T, dh = qf.shape
        dev = qf.device
        masks = _Gdn2Masks(dev, torch.float32)
        dout = dout.detach().float().transpose(1, 2)  # (B,H,T,dh)
        bounds = list(range(0, T, chunk))
        grads = [torch.zeros_like(t) for t in (qf, kf, vf, bf, wf, gf)]
        dS = None  # grad wrt carried state (None at end)
        # Reverse pass: walk anchor SEGMENTS last -> first. Within a segment, chunk-start states
        # are rebuilt by a no-grad forward REPLAY from the segment's anchor — exact and
        # unconditionally stable. (The closed-form inversion _gdn2_invert_chunk is exact algebra
        # but ill-conditioned under production decay rates: the delta-rule transition contracts S
        # per step, so inverting across chunks amplifies float error exponentially — measured in
        # the rev-bptt selftest. Replay bounds transient memory to `anchor` states per segment
        # instead, still O(anchors) stored vs O(T/chunk).) Each chunk is then re-run WITH grad.
        seg_starts = sorted(anchors)
        for si in range(len(seg_starts) - 1, -1, -1):
            a0 = seg_starts[si]
            a1 = len(bounds) if si == len(seg_starts) - 1 else seg_starts[si + 1]
            with torch.no_grad():
                S = anchors[a0]
                seg_states = [S]
                for ci in range(a0, a1 - 1):
                    c0 = bounds[ci]
                    c1 = min(c0 + chunk, T)
                    _, S = _gdn2_one_chunk(
                        S,
                        qf[:, :, c0:c1],
                        kf[:, :, c0:c1],
                        vf[:, :, c0:c1],
                        bf[:, :, c0:c1],
                        wf[:, :, c0:c1],
                        gf[:, :, c0:c1],
                        masks,
                    )
                    seg_states.append(S)
            for ci in range(a1 - 1, a0 - 1, -1):
                c0 = bounds[ci]
                c1 = min(c0 + chunk, T)
                S_in = seg_states[ci - a0].detach().requires_grad_(True)
                ins = [
                    t[:, :, c0:c1].detach().requires_grad_(True) for t in (qf, kf, vf, bf, wf, gf)
                ]
                with torch.enable_grad():
                    O, S_out = _gdn2_one_chunk(S_in, *ins, masks)
                    outputs, gouts = [O], [dout[:, :, c0:c1]]
                    if dS is not None:
                        outputs.append(S_out)
                        gouts.append(dS)
                    gs = torch.autograd.grad(outputs, ins + [S_in], gouts, allow_unused=True)
                for gacc, gnew in zip(grads, gs[:6]):
                    if gnew is not None:
                        gacc[:, :, c0:c1] += gnew
                dS = gs[6]
            del seg_states
        return tuple(g_.transpose(1, 2).to(in_dtype) for g_ in grads) + (None, None)


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: CharkhaConfig):
        super().__init__()
        d, H, dh = cfg.d_model, cfg.n_heads, cfg.head_dim
        self.H, self.dh = H, dh
        self.gdn_chunk = cfg.gdn_chunk
        self.use_osdn = cfg.use_osdn
        self.use_bipolar_gate = cfg.use_bipolar_gate
        if cfg.use_osdn:  # OSDN per-head per-dim key preconditioner
            self.k_precond = nn.Parameter(torch.ones(H, dh))
        self.wq = nn.Linear(d, H * dh, bias=False)
        self.wk = nn.Linear(d, H * dh, bias=False)
        self.wv = nn.Linear(d, H * dh, bias=False)
        self.wb = nn.Linear(d, H, bias=True)  # beta (write strength)
        self.wa = nn.Linear(d, H, bias=True)  # alpha (decay) input
        self.A_log = nn.Parameter(torch.zeros(H))  # per-head decay rate
        self.wg = nn.Linear(d, H * dh, bias=False)  # output gate
        self.onorm = RMSNorm(dh)
        self.wo = nn.Linear(H * dh, d, bias=False)
        # causal depthwise short conv (kernel 4) on q,k,v - Mamba/GDN standard
        self.conv = nn.Conv1d(3 * H * dh, 3 * H * dh, 4, groups=3 * H * dh, padding=3)

    def _conv_stream(self, raw, cache):
        # Streaming causal depthwise conv (kernel 4): keep the last 3 raw input columns as the
        # cache; conv over [tail | new] with no padding emits exactly the new positions' outputs.
        # A zeros tail on the first call reproduces the padded/cropped full-sequence conv exactly.
        tail = cache.get("conv")
        if tail is None:
            tail = raw.new_zeros(raw.size(0), raw.size(1), 3)
        hist = torch.cat([tail, raw], -1)
        cache["conv"] = hist[..., -3:]
        return F.conv1d(hist, self.conv.weight, self.conv.bias, groups=self.conv.groups)

    def forward(self, x, cache=None):
        # cache (streaming decode): {'conv': (B,3HD,3) raw conv tail, 'S': (B,H,dh,dh) carried
        # scan state}. The recurrent state IS the mixer's entire memory, so decode is O(1)/token.
        B, T, _ = x.shape
        H, dh = self.H, self.dh
        qkv = torch.cat([self.wq(x), self.wk(x), self.wv(x)], dim=-1)
        raw = qkv.transpose(1, 2)
        co = self._conv_stream(raw, cache) if cache is not None else self.conv(raw)[..., :T]
        qkv = F.silu(co).transpose(1, 2)
        q, k, v = qkv.split(H * dh, dim=-1)
        q, k, v = q.view(B, T, H, dh), k.view(B, T, H, dh), v.view(B, T, H, dh)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)  # L2-normed q,k
        if self.use_bipolar_gate:
            # STE: forward uses discrete signs, backward flows through as identity. k MUST stay
            # unit-norm: the delta rule assumes ‖k‖≈1, but raw sign(k) has ‖k‖=√dh, making the
            # state recurrence S(I-β·k·kᵀ) expansive (spectral radius β·dh≫1) — it overflows to
            # inf over a long sequence. Renormalize the sign vector back to unit norm; v (a value,
            # not a key) has no such constraint, so ±1 is fine.
            k = ((k.sign() / math.sqrt(self.dh)) - k).detach() + k
            v = (v.sign() - v).detach() + v
        if self.use_osdn:  # OSDN: per-dim key preconditioning
            k = k * self.k_precond.clamp(0.25, 4.0)
        beta = torch.sigmoid(self.wb(x))  # (B,T,H)
        alpha = torch.exp(
            -F.softplus(self.A_log)[None, None] * torch.sigmoid(self.wa(x))
        )  # (B,T,H)
        g = alpha.clamp(min=1e-6).log()  # log-space decay gate
        if cache is not None:
            with torch.autocast(device_type=x.device.type, enabled=False):
                o, S = _gdn_chunk_scan(
                    q, k, v, g, beta, self.gdn_chunk, S0=cache.get("S"), return_state=True
                )
            cache["S"] = S
        elif _HAVE_FLA and x.device.type == "cuda":
            # fla chunk kernel for training; fused_recurrent for pure inference
            o = _fla_chunk_gated_delta_rule(
                q.float(),
                k.float(),
                v.float(),
                g.float(),
                beta.float(),
                scale=1.0,
                use_qk_l2norm_in_kernel=False,
            )[0]
        else:
            # No triton/fla (e.g. native Windows): chunkwise parallel scan instead of a per-timestep
            # Python loop — identical math, O(T/chunk) sequential steps. The old loop launched a
            # handful of tiny kernels per timestep × T × every GDN layer, which throttled both CPU
            # and CUDA-without-fla to a crawl. Selftest-guarded for exactness vs _gdn_sequential_ref.
            with torch.autocast(device_type=x.device.type, enabled=False):
                o = _gdn_chunk_scan(q, k, v, g, beta, self.gdn_chunk)
        o = self.onorm(o.to(x.dtype)) * F.silu(self.wg(x).view(B, T, H, dh))
        return self.wo(o.reshape(B, T, H * dh))


# --------------------------------------------------------------------------
# Gated DeltaNet-2 (arXiv:2605.22791, NVIDIA). 3/4 of layers when enabled.
# Decouples the scalar erase/write gate into channel-wise b_t (erase, key axis)
# and w_t (write, value axis), plus channel-wise decay. Recovers GDN/KDA when
# gates collapse to scalars. ~0.5% more params than GDN (b_proj + w_proj),
# ~2% more compute. The architecture follows the GatedDeltaNet-2 paper and
# NVIDIA reference repository. No NVIDIA source or kernels are included here:
# this PyTorch recurrence is written from the published equations, while the
# optional production kernels are imported from flash-linear-attention (MIT).
# sequential CPU fallback matches the math exactly at lower throughput.
# --------------------------------------------------------------------------


class GatedDeltaNet2(nn.Module):
    def __init__(self, cfg: CharkhaConfig):
        super().__init__()
        d, H, dh = cfg.d_model, cfg.n_heads, cfg.head_dim
        self.H, self.dh = H, dh
        self.use_osdn = cfg.use_osdn
        self.use_bipolar_gate = cfg.use_bipolar_gate
        self.gdn_chunk = cfg.gdn_chunk  # timesteps per scan segment (bounds backward graph)
        self.rev_bptt = bool(getattr(cfg, "rev_bptt", False))
        self.rev_anchor = int(getattr(cfg, "rev_anchor", 8))
        if cfg.use_osdn:
            self.k_precond = nn.Parameter(torch.ones(H, dh))
        self.wq = nn.Linear(d, H * dh, bias=False)
        self.wk = nn.Linear(d, H * dh, bias=False)
        self.wv = nn.Linear(d, H * dh, bias=False)
        # GDN-2 specific: channel-wise erase gate b_t (key axis) and write gate w_t (value axis)
        self.wb = nn.Linear(d, H * dh, bias=False)  # erase: (B, H*dh) -> sigmoid -> (B, H, dh)
        self.ww = nn.Linear(d, H * dh, bias=False)  # write: (B, H*dh) -> sigmoid -> (B, H, dh)
        # Channel-wise decay: f_proj + A_log + dt_bias, as in the NVlabs implementation
        self.f_proj = nn.Sequential(
            nn.Linear(d, dh, bias=False),
            nn.Linear(dh, H * dh, bias=False),
        )
        self.A_log = nn.Parameter(torch.zeros(H))  # per-head log-decay rate
        self.dt_bias = nn.Parameter(torch.zeros(H * dh))  # per-channel step-size bias
        self.wg = nn.Linear(d, H * dh, bias=False)  # output gate
        self.onorm = RMSNorm(dh)
        self.wo = nn.Linear(H * dh, d, bias=False)
        self.conv = nn.Conv1d(3 * H * dh, 3 * H * dh, 4, groups=3 * H * dh, padding=3)

    _conv_stream = GatedDeltaNet._conv_stream  # same streaming causal-conv tail logic

    def forward(self, x, cache=None):
        # cache (streaming decode): see GatedDeltaNet.forward — conv tail + carried state S.
        B, T, _ = x.shape
        H, dh = self.H, self.dh
        qkv = torch.cat([self.wq(x), self.wk(x), self.wv(x)], dim=-1)
        raw = qkv.transpose(1, 2)
        co = self._conv_stream(raw, cache) if cache is not None else self.conv(raw)[..., :T]
        qkv = F.silu(co).transpose(1, 2)
        q, k, v = qkv.split(H * dh, dim=-1)
        q, k, v = q.view(B, T, H, dh), k.view(B, T, H, dh), v.view(B, T, H, dh)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        if self.use_bipolar_gate:
            # see GatedDeltaNet: keep the discretized key unit-norm or the recurrence blows up.
            k = ((k.sign() / math.sqrt(self.dh)) - k).detach() + k
            v = (v.sign() - v).detach() + v
        if self.use_osdn:
            k = k * self.k_precond.clamp(0.25, 4.0)
        # Channel-wise gates: erase (key axis) and write (value axis)
        b = torch.sigmoid(self.wb(x)).view(B, T, H, dh)  # (B,T,H,dh) channel-wise erase
        w = torch.sigmoid(self.ww(x)).view(B, T, H, dh)  # (B,T,H,dh) channel-wise write
        # Channel-wise decay: f_proj(input) -> log-space gate + dt_bias + A_log
        g_in = self.f_proj(x).view(B, T, H, dh)  # (B,T,H,dh)
        dt = -F.softplus(g_in + self.dt_bias.view(1, 1, H, dh))
        a = -F.softplus(self.A_log)[None, None, :, None]  # (1,1,H,1) per-head rate
        g = (a + dt).clamp(max=0.0)  # log-decay, capped at 0
        if cache is not None:
            if _HAVE_FLA_GDN2 and x.device.type == "cuda":
                # Fused recurrent GDN-2 step kernel — same recurrence as chunk_gdn2, state layout
                # (B,H,K,V) fp32 interchangeable with _gdn2_chunk_scan's carry (verified to
                # rel L2 ~1e-7 against the chunk-scan reference with a carried prefill state).
                o, S = _fla_fused_recurrent_gdn2(
                    q.float(),
                    k.float(),
                    v.float(),
                    g.float(),
                    b.float(),
                    w.float(),
                    scale=1.0,
                    initial_state=cache.get("S"),
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=False,
                )
            else:
                o, S = _gdn2_chunk_scan(
                    q, k, v, b, w, g, max(1, self.gdn_chunk), S0=cache.get("S"), return_state=True
                )
            cache["S"] = S
        elif self.rev_bptt and self.training and torch.is_grad_enabled():
            # Reversible-recurrence BPTT (cfg.rev_bptt): no per-chunk states are stored for
            # backward — they are reconstructed by closed-form inversion (see _RevGDN2Scan).
            # Same recompute cost as checkpointing, O(anchors) state memory. Takes priority over
            # the fla kernel because the memory saving IS the point of the flag.
            with torch.autocast(device_type=x.device.type, enabled=False):
                o = _RevGDN2Scan.apply(
                    q.float(),
                    k.float(),
                    v.float(),
                    b.float(),
                    w.float(),
                    g.float(),
                    max(1, self.gdn_chunk),
                    self.rev_anchor,
                )
        elif _HAVE_FLA_GDN2 and x.device.type == "cuda":
            # Fused Triton GDN-2 kernel (fla.ops.gdn2.chunk_gdn2) — the production path, analogous to
            # GDN-1's chunk_gated_delta_rule above. Its matrix-state recurrence
            #   S_t = (I − k_t (b_t⊙k_t)ᵀ)·Diag(exp(g_t))·S_{t-1} + k_t (w_t⊙v_t)ᵀ
            # expands term-for-term to _gdn2_sequential_ref (b = key-axis erase, w = value-axis write,
            # g = per-key-channel log-decay ≤0, output q·S_t computed after the write). scale=1.0
            # reproduces the reference's UNSCALED q·S readout; q,k are already L2-normed above so the
            # in-kernel normalisation stays off. Validated GPU-side by the gdn2-fla selftest.
            o = _fla_chunk_gdn2(
                q.float(),
                k.float(),
                v.float(),
                g.float(),
                b.float(),
                w.float(),
                scale=1.0,
                use_qk_l2norm_in_kernel=False,
            )[0]  # (B,T,H,dh)
        else:
            # No fla GDN-2 kernel (native Windows / CPU / older fla): chunkwise-parallel WY/UT scan,
            # identical math, O(T/chunk) sequential steps of batched matmuls + one batched unit-
            # triangular solve per chunk. Selftest-guarded vs
            # _gdn2_sequential_ref to fp32 roundoff.
            o = _gdn2_chunk_scan(q, k, v, b, w, g, max(1, self.gdn_chunk))  # (B,T,H,dh) fp32
        o = self.onorm(o.to(x.dtype)) * F.silu(self.wg(x).view(B, T, H, dh))
        return self.wo(o.reshape(B, T, H * dh))


class Subconscious(nn.Module):
    """A small, SEPARATE-parameter, low-dim recurrent scratchpad gated into the main residual stream
    (cfg.use_subconscious, OFF by default). Across the recurrent core's reasoning loops it carries a
    low-dim latent z (a GRU cell, ~dim^2 params, independent of the d_model trunk), reads a per-token
    summary of the evolving core state s, and writes a GATED low-rank contribution back into s. The
    write projection is zero-initialized and the gate starts closed (bias -4), so at init the module
    is an EXACT no-op and must earn its influence during training. This is deliberately NOT the fragile
    weight-slice-sharing design from an earlier brainstorm: the scratchpad owns its parameters and its
    own low-dim recurrence.

    Soundness: it is applied BETWEEN core loops, OUTSIDE the gradient-checkpointed core pass, with z
    threaded as an explicit per-call tensor (never module state) — so there is no checkpoint-recompute
    double-update hazard, and z follows the same truncated-BPTT / no-grad regime as the core state s.

    force_gate_zero is the equal-FLOP ablation control: the full read -> recurrence -> write runs (same
    compute and parameter count) but the gate is clamped shut, so any gain over this control is
    attributable to the scratchpad's INFLUENCE, not merely to the extra params/FLOPs."""

    def __init__(self, d_model, dim=32):
        super().__init__()
        self.dim = dim
        self.read = nn.Linear(d_model, dim, bias=False)  # stream summary -> scratchpad input
        self.cell = nn.GRUCell(dim, dim)  # low-dim recurrence across reasoning loops
        self.write = nn.Linear(dim, d_model, bias=False)  # scratchpad -> stream contribution
        self.gate = nn.Linear(d_model, 1)  # per-token write gate
        self.force_gate_zero = False  # equal-FLOP ablation control
        self.reset_noop()

    def reset_noop(self):
        # Exact no-op init: zero the write projection (contribution == 0) and close the gate. MUST be
        # re-called AFTER any global weight-init pass — Charkha.apply(self._init) re-randomizes every
        # Linear, which would otherwise clobber this and make the module non-inert at step 0.
        nn.init.zeros_(self.write.weight)  # contribution starts exactly 0...
        nn.init.constant_(self.gate.bias, -4.0)  # ...gate also starts ~closed (sigmoid -4)

    def init_state(self, ref):
        B, T, _ = ref.shape
        return ref.new_zeros(B * T, self.dim)

    def forward(self, s, z):
        # s: (B,T,d) current core state; z: (B*T,dim) carried scratchpad latent. Returns (s', z').
        B, T, d = s.shape
        z = self.cell(self.read(s).reshape(B * T, self.dim), z)
        contrib = self.write(z).reshape(B, T, d)
        g = torch.sigmoid(self.gate(s))  # (B,T,1) per-token write gate
        if self.force_gate_zero:
            g = g * 0.0  # ablation: identical FLOPs, no influence
        return s + g * contrib, z


class LoopAdapters(nn.Module):
    """Per-loop low-rank state deltas (cfg.use_loop_adapters, OFF by default).
    The recurrent core is weight-tied across iterations; the paper's virtual-depth
    section names this as a specialization risk. Each loop index n gets its own
    rank-r adapter s <- s + up_n(silu(down_n(s))) so iterations can differentiate
    like distinct layers, at ~2*d*rank params per allocated loop. up_n is
    zero-initialized => EXACT no-op at init (same discipline as Subconscious);
    loops beyond `max_loops` reuse the last adapter, so any inference effort is
    valid. Applied inside _core_step, which covers every core path (fixed,
    halting, converge, per-sequence, streaming decode) identically."""

    def __init__(self, d_model, rank=8, max_loops=16):
        super().__init__()
        self.max_loops = max_loops
        self.down = nn.ModuleList(nn.Linear(d_model, rank, bias=False) for _ in range(max_loops))
        self.up = nn.ModuleList(nn.Linear(rank, d_model, bias=False) for _ in range(max_loops))
        self.reset_noop()

    def reset_noop(self):
        # MUST be re-called after any global init pass (see Subconscious.reset_noop).
        for u in self.up:
            nn.init.zeros_(u.weight)

    def forward(self, s, n):
        i = min(int(n), self.max_loops - 1)
        return s + self.up[i](F.silu(self.down[i](s)))


class Block(nn.Module):
    def __init__(self, cfg, mixer):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.mixer, self.mlp = mixer, SwiGLU(cfg)

    def f_part(self, x, cache=None):
        """Mixer sublayer F(x) = mixer(n1(x)) — the first residual branch. Split out so the
        reversible two-stream coupling (cfg.reversible) can drive F and G independently."""
        if cache is not None:
            return self.mixer(self.n1(x), cache=cache.setdefault("mixer", {}))
        return self.mixer(self.n1(x))  # no kwarg: stays drop-in for any mixer module

    def g_part(self, x, cache=None):
        """MLP-side sublayer G(x) = mlp(n2(x)) — the second residual branch."""
        h2 = self.n2(x)
        out = self.mlp(h2)
        return out

    def forward(self, x, cache=None):
        # cache (streaming decode): per-block dict with sub-dicts for the mixer and the
        # sequence-dependent reasoning modules (calculus/comparator). None = full forward.
        x = x + self.f_part(x, cache=cache)
        return x + self.g_part(x, cache=cache)


def _autocast_meta(x):
    """Capture the ambient autocast state so a reversible backward can recompute F/G under the
    SAME numerics as the forward (bitwise reconstruction is what makes the inversion exact)."""
    dev = x.device.type
    try:
        en = torch.is_autocast_enabled(dev)
        dt = torch.get_autocast_dtype(dev) if en else None
    except TypeError:  # older torch: per-device APIs
        en = torch.is_autocast_cpu_enabled() if dev == "cpu" else torch.is_autocast_enabled()
        dt = (
            (torch.get_autocast_cpu_dtype() if dev == "cpu" else torch.get_autocast_gpu_dtype())
            if en
            else None
        )
    return dev, en, dt


class _RevStackFn(torch.autograd.Function):
    """RevNet-style two-stream reversible coupling over a whole block stack (cfg.reversible):
        y1 = x1 + F(x2);   y2 = x2 + G(y1)          per block, streams initialized x1 = x2 = x,
    readout (y1+y2)/2 at the stack exit. Forward stores ONLY the final (y1, y2): backward
    reconstructs every block's inputs by inversion (x2 = y2 − G(y1); x1 = y1 − F(x2)),
    recomputing F and G once each — checkpoint-equivalent compute, O(1) activation memory in
    depth. Recompute re-enters the forward's autocast state with cache_enabled=False so the
    reconstruction is exact (same reasoning as the train loop's autocast note)."""

    @staticmethod
    def forward(ctx, x, blocks, ac_meta, *params):
        with torch.no_grad():
            x1 = x2 = x
            for blk in blocks:
                x1 = x1 + blk.f_part(x2)
                x2 = x2 + blk.g_part(x1)
        ctx.blocks, ctx.ac = blocks, ac_meta
        ctx.n_params = len(params)
        ctx.save_for_backward(x1, x2)
        return 0.5 * (x1 + x2)

    @staticmethod
    def backward(ctx, dy):
        y1, y2 = ctx.saved_tensors
        dev, ac_en, ac_dt = ctx.ac
        dy1 = dy2 = 0.5 * dy
        pgrad = {}  # id(param) -> accumulated grad

        def ac():
            return torch.autocast(device_type=dev, dtype=ac_dt, enabled=ac_en, cache_enabled=False)

        for blk in reversed(ctx.blocks):
            f_params = [
                p
                for p in list(blk.n1.parameters()) + list(blk.mixer.parameters())
                if p.requires_grad
            ]
            g_mods = [blk.n2, blk.mlp]
            g_params = [p for m in g_mods for p in m.parameters() if p.requires_grad]
            with torch.enable_grad(), ac():
                y1r = y1.detach().requires_grad_(True)
                g_out = blk.g_part(y1r)
            x2 = (y2 - g_out).detach()  # invert stream 2
            with torch.enable_grad(), ac():
                x2r = x2.detach().requires_grad_(True)
                f_out = blk.f_part(x2r)
            x1 = (y1 - f_out).detach()  # invert stream 1
            gg = torch.autograd.grad(g_out, [y1r] + g_params, dy2, allow_unused=True)
            dy1_t = dy1 + gg[0]
            for p, g_ in zip(g_params, gg[1:]):
                if g_ is not None:
                    pgrad[id(p)] = g_ if id(p) not in pgrad else pgrad[id(p)] + g_
            fg = torch.autograd.grad(f_out, [x2r] + f_params, dy1_t, allow_unused=True)
            for p, g_ in zip(f_params, fg[1:]):
                if g_ is not None:
                    pgrad[id(p)] = g_ if id(p) not in pgrad else pgrad[id(p)] + g_
            y1, y2 = x1, x2
            dy1, dy2 = dy1_t, dy2 + fg[0]
        # stack entry duplicated x into both streams: dL/dx = dx1 + dx2
        all_params = [p for blk in ctx.blocks for p in blk.parameters() if p.requires_grad]
        return (dy1 + dy2, None, None, *[pgrad.get(id(p)) for p in all_params])


def _rev_run_blocks(blocks, x):
    params = [p for blk in blocks for p in blk.parameters() if p.requires_grad]
    return _RevStackFn.apply(x, list(blocks), _autocast_meta(x), *params)


def _two_stream_blocks(blocks, x, caches=None):
    """Plain two-stream forward — the FUNCTION cfg.reversible defines, used on every path that
    does not need the memory-free backward (eval, no-grad, streaming decode with caches).
    _RevStackFn computes exactly this function when training; keeping one definition here is what
    guarantees train/serve consistency for the reversible architecture."""
    x1 = x2 = x
    for i, blk in enumerate(blocks):
        c = caches[i] if caches is not None else None
        x1 = x1 + blk.f_part(x2, cache=c)
        x2 = x2 + blk.g_part(x1, cache=c)
    return 0.5 * (x1 + x2)


def make_stack(cfg, n, final_global_attn):
    blocks = []
    for i in range(n):
        is_attn = ((i + 1) % 4 == 0) or (final_global_attn and i == n - 1)
        if is_attn:
            window = None if (final_global_attn and i == n - 1) else cfg.window
            blocks.append(Block(cfg, Attention(cfg, window)))
        else:
            mixer = GatedDeltaNet2(cfg) if cfg.use_gdn2 else GatedDeltaNet(cfg)
            blocks.append(Block(cfg, mixer))
    return nn.ModuleList(blocks)


# --------------------------------------------------------------------------
# Fused cross-entropy helpers. Computing logits = h @ Wᵀ over the whole
# (B*T, vocab) at once is the activation-memory killer at T=4096 (the (B,T,50304)
# logits + its fp32 softmax ≈ several GB, held through backward). These run one row-
# chunk at a time; the model wraps each call in gradient checkpointing so the chunk's
# logits are recomputed in backward instead of stored. Math is identical to a single
# F.cross_entropy(reduction='sum'); summed over chunks then divided by token count.
# --------------------------------------------------------------------------


class FactorizedEmbedding(nn.Module):
    """ALBERT-style factorized tied embedding (cfg.embed_factor): token -> codes (V, f) -> up (f, d).
    The (V, d) table never exists; the tied LM head reads logits = (h @ up) @ codesᵀ via _head_hw.
    Both parameter names contain 'embed' so the optimizer builders route them exactly like the
    dense table (row-NormM under --symmetry-opt, AdamW otherwise)."""

    def __init__(self, vocab_size, d_model, factor):
        super().__init__()
        self.codes = nn.Embedding(vocab_size, factor)
        self.up = nn.Linear(factor, d_model, bias=False)

    def forward(self, idx):
        return self.up(self.codes(idx))

    @property
    def weight(self):
        # device/dtype probes only (e.g. lm.embed.weight.device). NOT a (V, d) matrix — every
        # real head read goes through Charkha._head_hw, which knows the factorized shape.
        return self.codes.weight


# --------------------------------------------------------------------------
# Epistemic uncertainty heads (honesty pillar). Both are OFF by default and add
# 0 cost when disabled. SNGP is a train-time parallel head; LaplaceConf is a
# post-hoc wrapper fit on a calibration set after training.
# --------------------------------------------------------------------------


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

__all__ = [
    "Attention",
    "Block",
    "FactorizedEmbedding",
    "GatedDeltaNet",
    "GatedDeltaNet2",
    "LaplaceConf",
    "LoopAdapters",
    "RMSNorm",
    "SNGPHead",
    "Subconscious",
    "SwiGLU",
    "apply_rope",
    "have_fla",
    "make_stack",
    "rope_cache",
]
