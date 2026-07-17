"""CHARKHA loss helpers — chunked fused cross-entropy."""

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt


def _ce_chunk(h, W, t, hp=None, Wp=None, pw=0.0, cap=0.0, zl=0.0):
    logits = F.linear(h, W)
    if hp is not None:  # frozen residual logit prior (train.py --logit-prior)
        logits = logits + pw * F.linear(hp, Wp)
    if cap > 0:  # final-logit softcap (Gemma-2): cap*tanh(l/cap) bounds
        logits = cap * torch.tanh(logits / cap)  # logits smoothly, preventing blow-ups/overconf.
    ce = F.cross_entropy(logits, t, reduction="sum")
    if zl > 0:  # z-loss (PaLM): pull log Z toward 0 so logit scale
        ce = ce + zl * torch.logsumexp(logits.float(), -1).pow(2).sum()  # stays bounded in bf16
    return ce


def _ce_argmax_chunk(h, W, t, hp=None, Wp=None, pw=0.0, cap=0.0, zl=0.0):
    logits = F.linear(h, W)
    if hp is not None:
        logits = logits + pw * F.linear(hp, Wp)
    if cap > 0:
        logits = cap * torch.tanh(logits / cap)
    ce = F.cross_entropy(logits, t, reduction="sum")
    if zl > 0:
        ce = ce + zl * torch.logsumexp(logits.float(), -1).pow(2).sum()
    return ce, (logits.detach().argmax(-1) == t).float()


def _ce_vchunk_lse(h, Wj, hp=None, Wpj=None, pw=0.0, cap=0.0):
    """Partial logsumexp over ONE vocab chunk (CCE-style streaming; cfg.ce_vchunk). Checkpointed
    by the caller so the (rows, Vc) logit block is recomputed in backward, never stored."""
    lg = F.linear(h, Wj)
    if hp is not None:
        lg = lg + pw * F.linear(hp, Wpj)
    if cap > 0:
        lg = cap * torch.tanh(lg / cap)
    return torch.logsumexp(lg.float(), -1)


def _ce_stream(
    h, W, t, vchunk, ckpt, hp=None, Wp=None, pw=0.0, cap=0.0, zl=0.0, want_correct=False
):
    """Vocab-streamed fused CE (cfg.ce_vchunk > 0): CE_sum = Σ_rows (lse(logits) − logit[target]),
    with the lse assembled from per-vocab-chunk partials — the full (rows, V) logit block never
    exists, only (rows, vchunk) transiently per checkpointed piece (arXiv:2411.09009 in spirit;
    pure PyTorch here). Softcap / prior / z-loss applied exactly as in _ce_chunk; math identical."""
    V = W.size(0)
    # target logit: a single gathered column per row — cheap and exact
    lt = (h * W[t]).sum(-1)
    if hp is not None:
        lt = lt + pw * (hp * Wp[t]).sum(-1)
    if cap > 0:
        lt = cap * torch.tanh(lt / cap)
    parts = []
    correct = None
    run_max = run_idx = None  # streamed argmax (no_grad, exact)
    for j in range(0, V, vchunk):
        Wj = W[j : j + vchunk]
        Wpj = Wp[j : j + vchunk] if Wp is not None else None
        p = (
            _ckpt.checkpoint(_ce_vchunk_lse, h, Wj, hp, Wpj, pw, cap, use_reentrant=False)
            if ckpt
            else _ce_vchunk_lse(h, Wj, hp, Wpj, pw, cap)
        )
        parts.append(p)
        if want_correct:
            with torch.no_grad():
                lg = F.linear(h, Wj)
                if hp is not None:
                    lg = lg + pw * F.linear(hp, Wpj)
                mx, ix = lg.max(-1)
                if run_max is None:
                    run_max, run_idx = mx, ix + j
                else:
                    upd = mx > run_max
                    run_max = torch.where(upd, mx, run_max)
                    run_idx = torch.where(upd, ix + j, run_idx)
    lse = torch.logsumexp(torch.stack(parts, -1), -1)  # combine partial lses (exact)
    ce = (lse - lt.float()).sum()
    if zl > 0:
        ce = ce + zl * lse.pow(2).sum()
    if want_correct:
        correct = (run_idx == t).float()  # softcap is monotone: argmax unchanged
    return ce, correct


# --------------------------------------------------------------------------
# Epistemic uncertainty heads (honesty pillar). Both are OFF by default and add
# 0 cost when disabled. SNGP is a train-time parallel head; LaplaceConf is a
# post-hoc wrapper fit on a calibration set after training.
# --------------------------------------------------------------------------
