"""
CHARKHA merge - same-architecture weight-to-weight checkpoint merging.
=====================================================================
The ONLY honest form of "weight-to-weight" model combination: merging checkpoints that share an
architecture (the basis problem makes cross-architecture weight transfer incoherent — use
distillation for that). This composes CHARKHA finetunes WITHOUT retraining — e.g. a math-tuned and a
code-tuned CHARKHA -> one model with both skills. Three methods, all per-tensor:

  * soup      uniform/weighted average of full weights (Model Soups, arXiv:2203.05482). Best when the
              models share a pretraining lineage (finetunes of one base).
  * slerp     spherical interpolation of two models (preserves weight norm; common for 2-model blends).
  * ties      TIES-merge (arXiv:2306.01708) of finetune DELTAS over a --base: trim each delta to its
              top-`density` magnitudes, elect a sign per parameter, average only the agreeing deltas,
              add back to the base. Reduces interference between finetunes.

A merged checkpoint carries {model, cfg} (+ step=0, meta) — load it in serve.py directly, or start a
fresh finetune with train.py --branch-from (optimizer state begins clean).

Refuses to merge mismatched architectures (different cfg shape dims) — that's the basis wall, loudly.

Usage:
  python merge.py --soup a.pt b.pt c.pt --out merged.pt
  python merge.py --slerp a.pt b.pt --t 0.4 --out merged.pt
  python merge.py --ties a.pt b.pt --base pretrain.pt --density 0.2 --out merged.pt
  python merge.py --selftest

"""

from __future__ import annotations
import argparse
import sys
import tempfile
import os

import torch

# cfg fields that change tensor SHAPES — these must match across merged models, or the state dicts
# are incompatible (the basis wall). Other cfg fields (LR-ish, flags) don't affect shapes.
_SHAPE_FIELDS = (
    "vocab_size",
    "d_model",
    "n_heads",
    "n_kv_heads",
    "d_ff",
    "n_prelude",
    "n_core",
    "n_coda",
    "use_osdn",
    "use_gdn2",
    "use_mtp_routing",
    "sngp_enabled",
    "use_nitp",
)


def _load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = (
        ck["cfg"] if isinstance(ck.get("cfg"), dict) else (vars(ck["cfg"]) if "cfg" in ck else None)
    )
    state = ck.get("model", ck)
    return state, cfg


def _check_compat(cfgs):
    """All models must agree on every shape-determining cfg field, else the tensors don't align."""
    base = cfgs[0]
    for i, c in enumerate(cfgs[1:], 1):
        if base is None or c is None:
            continue
        for f in _SHAPE_FIELDS:
            if c.get(f) != base.get(f):
                raise ValueError(
                    f"architecture mismatch on {f!r}: model0={base.get(f)} vs model{i}={c.get(f)}. "
                    f"Weight-to-weight merge needs identical architecture (basis wall) — "
                    f"use distillation across different architectures."
                )


def _check_keys(states):
    k0 = set(states[0])
    for i, s in enumerate(states[1:], 1):
        if set(s) != k0:
            raise ValueError(
                f"state-dict keys differ (model0 vs model{i}); incompatible checkpoints."
            )


def soup(states, weights=None):
    """Weighted average of full weight tensors (uniform if weights=None)."""
    _check_keys(states)
    n = len(states)
    w = [1.0 / n] * n if weights is None else [x / sum(weights) for x in weights]
    out = {}
    for k in states[0]:
        acc = None
        for wi, s in zip(w, states):
            t = s[k].float() * wi
            acc = t if acc is None else acc + t
        out[k] = acc.to(states[0][k].dtype)
    return out


def slerp(sa, sb, t=0.5, eps=1e-8):
    """Per-tensor spherical interpolation between two models (t=0 -> A, t=1 -> B)."""
    _check_keys([sa, sb])
    out = {}
    for k in sa:
        a, b = sa[k].float().flatten(), sb[k].float().flatten()
        na, nb = a.norm(), b.norm()
        if na < eps or nb < eps:  # a zero tensor -> plain lerp
            out[k] = ((1 - t) * sa[k].float() + t * sb[k].float()).to(sa[k].dtype)
            continue
        dot = torch.dot(a / na, b / nb).clamp(-1.0, 1.0)
        omega = torch.arccos(dot)
        so = torch.sin(omega)
        if so < eps:  # nearly colinear -> lerp
            merged = (1 - t) * a + t * b
        else:
            merged = (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b
        out[k] = merged.reshape(sa[k].shape).to(sa[k].dtype)
    return out


def ties(states, base, density=0.2):
    """TIES-merge of finetune deltas over `base`: trim to top-`density` magnitudes per delta, elect a
    per-parameter sign by summed magnitude, average only deltas agreeing with the elected sign, add to
    base. Cuts destructive interference between finetunes that a plain soup would average away."""
    _check_keys(states + [base])
    out = {}
    for k in base:
        b = base[k].float()
        deltas = [s[k].float() - b for s in states]
        trimmed = []
        for d in deltas:
            flat = d.flatten()
            kth = max(1, int(density * flat.numel()))
            thresh = (
                flat.abs().kthvalue(flat.numel() - kth + 1).values
                if kth < flat.numel()
                else flat.abs().min()
            )
            trimmed.append(torch.where(d.abs() >= thresh, d, torch.zeros_like(d)))
        stacked = torch.stack(trimmed)  # (M, *shape)
        sign = torch.sign(stacked.sum(0))  # elected sign per parameter
        agree = (torch.sign(stacked) == sign.unsqueeze(0)) & (stacked != 0)
        summed = (stacked * agree).sum(0)
        cnt = agree.sum(0).clamp(min=1)
        out[k] = (b + summed / cnt).to(base[k].dtype)
    return out


def _save(out_path, state, cfg):
    payload = {
        "model": state,
        "cfg": cfg,
        "step": 0,
        "meta": {"loss_ema": None, "best_val": None, "merged": True},
    }
    tmp = out_path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)


def main():
    ap = argparse.ArgumentParser(description="CHARKHA same-architecture checkpoint merge")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--soup", nargs="+", metavar="CKPT", help="average these checkpoints")
    g.add_argument(
        "--slerp", nargs=2, metavar="CKPT", help="spherically interpolate two checkpoints"
    )
    g.add_argument("--ties", nargs="+", metavar="CKPT", help="TIES-merge these (requires --base)")
    ap.add_argument("--base", help="base checkpoint for --ties (the pre-finetune model)")
    ap.add_argument("--weights", nargs="+", type=float, help="per-model weights for --soup")
    ap.add_argument("--t", type=float, default=0.5, help="--slerp interpolation (0->A, 1->B)")
    ap.add_argument("--density", type=float, default=0.2, help="--ties keep-fraction per delta")
    ap.add_argument("--out", help="output checkpoint path")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if not a.out:
        ap.error("--out is required")

    if a.soup:
        loaded = [_load(p) for p in a.soup]
        _check_compat([c for _, c in loaded])
        merged = soup([s for s, _ in loaded], a.weights)
        cfg = loaded[0][1]
        print(
            f"[merge] soup of {len(a.soup)} models"
            + (f" weights={a.weights}" if a.weights else " (uniform)")
        )
    elif a.slerp:
        (sa, ca), (sb, cb) = _load(a.slerp[0]), _load(a.slerp[1])
        _check_compat([ca, cb])
        merged = slerp(sa, sb, a.t)
        cfg = ca
        print(f"[merge] slerp t={a.t} of {a.slerp[0]} -> {a.slerp[1]}")
    elif a.ties:
        if not a.base:
            ap.error("--ties requires --base")
        loaded = [_load(p) for p in a.ties]
        sb, cb = _load(a.base)
        _check_compat([c for _, c in loaded] + [cb])
        merged = ties([s for s, _ in loaded], sb, a.density)
        cfg = cb
        print(f"[merge] TIES of {len(a.ties)} deltas over {a.base} (density={a.density})")
    else:
        ap.error("choose one of --soup / --slerp / --ties")

    _save(a.out, merged, cfg)
    print(f"[merge] wrote {a.out} (load in serve.py, or finetune fresh via train.py --branch-from)")
    return 0


# --------------------------------------------------------------------------
def selftest():
    print("CHARKHA merge self-test")
    from charkha import Charkha, CharkhaConfig

    torch.manual_seed(0)
    checks = {}
    cfg = CharkhaConfig.toy()
    a_model, b_model = Charkha(cfg), Charkha(cfg)
    sa, sb = a_model.state_dict(), b_model.state_dict()
    cfgd = dict(cfg.__dict__)

    # --- soup is the exact per-tensor average ---
    m = soup([sa, sb])
    k = "adapter.weight"
    checks["soup == mean of weights"] = torch.allclose(
        m[k].float(), (sa[k] + sb[k]).float() / 2, atol=1e-5
    )
    # weighted soup with w=[1,0] returns model A exactly
    mw = soup([sa, sb], [1.0, 0.0])
    checks["weighted soup w=[1,0] == A"] = torch.allclose(mw[k].float(), sa[k].float(), atol=1e-5)

    # --- slerp endpoints + identity ---
    s0 = slerp(sa, sb, 0.0)
    s1 = slerp(sa, sb, 1.0)
    checks["slerp t=0 == A"] = torch.allclose(s0[k].float(), sa[k].float(), atol=1e-4)
    checks["slerp t=1 == B"] = torch.allclose(s1[k].float(), sb[k].float(), atol=1e-4)
    same = slerp(sa, sa, 0.5)  # slerp of a model with itself is itself
    checks["slerp(A,A) == A"] = torch.allclose(same[k].float(), sa[k].float(), atol=1e-4)

    # --- ties over a base: zero deltas -> base unchanged ---
    t_same = ties([sa, sa], sa, density=0.5)
    checks["ties(no delta) == base"] = torch.allclose(t_same[k].float(), sa[k].float(), atol=1e-5)
    t_merge = ties([sa, sb], sa, density=0.3)  # real merge stays finite + right shape
    checks["ties merge finite + shape"] = (
        torch.isfinite(t_merge[k]).all() and t_merge[k].shape == sa[k].shape
    )

    # --- every merged state loads into a fresh model and runs a forward ---
    ok_load = True
    for name, st in (("soup", m), ("slerp", s0), ("ties", t_merge)):
        try:
            mdl = Charkha(CharkhaConfig.toy())
            mdl.load_state_dict(st)
            mdl.eval()
            with torch.no_grad():
                lg, _ = mdl(torch.randint(0, cfg.vocab_size, (1, 8)))
            ok_load &= tuple(lg.shape) == (1, 8, cfg.vocab_size)
        except Exception as e:
            print(f"  [load {name}] {type(e).__name__}: {e}")
            ok_load = False
    checks["merged states load + forward"] = ok_load

    # --- architecture mismatch is refused (the basis wall) ---
    big = dict(cfgd)
    big["d_model"] = cfgd["d_model"] * 2
    try:
        _check_compat([cfgd, big])
        refused = False
    except ValueError:
        refused = True
    checks["refuses architecture mismatch"] = refused

    # --- round-trip through disk: save + reload a merged ckpt; serve-compatible {model,cfg} ---
    tmp = tempfile.mkdtemp(prefix="charkha_merge_")
    p = os.path.join(tmp, "merged.pt")
    _save(p, m, cfgd)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    checks["saved ckpt has model+cfg"] = "model" in ck and "cfg" in ck and ck["step"] == 0
    rmdl = Charkha(CharkhaConfig.from_dict(ck["cfg"]))
    rmdl.load_state_dict(ck["model"])
    checks["reloaded ckpt loads clean"] = True

    print()
    ok = True
    for n, p_ in checks.items():
        print(f"  [{'PASS' if p_ else 'FAIL'}] {n}")
        ok &= bool(p_)
    print(
        "\nSELFTEST",
        "PASS - same-arch soup/slerp/ties merge, guards basis mismatch, serve-loadable"
        if ok
        else "FAIL",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    raise SystemExit(main())
