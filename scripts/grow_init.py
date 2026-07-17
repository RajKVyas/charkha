#!/usr/bin/env python3
"""Grow a trained small CHARKHA checkpoint into a wider model's initialization
(HyperCloning-style width expansion, arXiv:2409.12903 family).

Train a mini fluency model first (fast — a small model does thousands of tok/s on the same
card), then expand it into the big config's init so the big run starts already speaking
English instead of spending its slow tokens on orthography. Width doubles by DUPLICATING
HEADS and d_model channels; every matmul input duplication is compensated by dividing the
weight tile, so the grown model computes (approximately) the same function:

  * Linear(d-like -> d-like): tile both dims, divide by the input duplication factor.
  * Embedding / factorized codes+up: duplicate output channels only (no division).
  * Per-head params (A_log, conv channels, norms over d): tile, no division.
  * head_dim stays constant (heads duplicate) — RoPE, QK-norm, and the delta-rule state all
    replicate exactly per head.

Known approximations (reported by --verify, re-smoothed within the first warmup steps):
  * the TIED LM head sees duplicated inputs -> logits scale by ~m (a uniform temperature
    sharpening; ranking is preserved exactly);
  * fixed sinusoidal buffers (loop_embed) are rebuilt for the big width, not duplicated.
A small --noise breaks the duplicate-symmetry so the copies can differentiate.

  python scripts/grow_init.py --small runs/mini/ckpt.pt --mult 2 --out runs/grown_init.pt --verify

"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from charkha import Charkha, CharkhaConfig  # noqa: E402


def grow_cfg(small: CharkhaConfig, m: int) -> CharkhaConfig:
    big = CharkhaConfig(**dict(small.__dict__))
    big.d_model = small.d_model * m
    big.n_heads = small.n_heads * m
    big.n_kv_heads = small.n_kv_heads * m
    big.d_ff = small.d_ff * m
    if small.playground_dim:
        big.playground_dim = small.playground_dim * m
    return big


def _tile(t: torch.Tensor, shape) -> torch.Tensor:
    """Tile t up to `shape`, slicing the final repeat when the target is not an integer
    multiple. Integer-multiple growth remains exactly the old HyperCloning path; fractional
    growth is an approximate warm start used when the measured VRAM optimum sits between clean
    multiples (e.g. 1280 -> 2048)."""
    reps = []
    for a, b in zip(t.shape, shape):
        reps.append((b + a - 1) // a)
    out = t.repeat(*reps)
    return out[tuple(slice(0, b) for b in shape)]


# Parameter-name suffixes whose dim-0 is the matmul INPUT (x @ W conventions), so the
# duplication factor of dim-0 must divide the tile. Everything else follows nn.Linear's
# (out, in) convention where dim-1 is the input.
_DIM0_INPUT = ("W_hat", "M_hat", "W_hat_log", "M_hat_log", "W_sign")
# No division at all: pure duplication (lookup tables, per-channel params, biases, norms).
_NO_DIV = (
    "embed.codes.weight",
    "embed.weight",
    "A_log",
    "dt_bias",
    ".bias",
    "norm.weight",
    "n1.weight",
    "n2.weight",
    "norm_f.weight",
    "qnorm",
    "knorm",
    "onorm",
    "loop_embed",
    "abacus_embed",
)


def grow_tensor(name: str, small_t: torch.Tensor, big_shape) -> torch.Tensor:
    if ".conv." in name and small_t.shape[0] % 3 == 0:
        # GDN's fused depthwise causal conv runs over CONCATENATED [q|k|v] channels
        # (nn.Conv1d(3*H*dh, ..., groups=3*H*dh)). Naive whole-tensor repeat would tile
        # [q,k,v]->[q,k,v,q,k,v] against the grown layout [q,q,k,k,v,v] — a silent scramble
        # (measured rel-err ~1.1 at the first GDN block). Tile each qkv segment separately.
        seg = small_t.shape[0] // 3
        big_seg = big_shape[0] // 3
        parts = [
            _tile(small_t[i * seg : (i + 1) * seg], (big_seg, *big_shape[1:])) for i in range(3)
        ]
        return torch.cat(parts, 0)  # depthwise: channels independent, no division
    t = _tile(small_t, big_shape)
    if any(k in name for k in _NO_DIV):
        return t
    if "conv.weight" in name:  # depthwise: channels independent, no division
        return t
    if small_t.ndim == 2:
        in_dim = 0 if any(name.endswith(k) for k in _DIM0_INPUT) else 1
        factor = big_shape[in_dim] / small_t.shape[in_dim]
        return t / factor
    return t


def grow(small_ckpt: str, mult: int, noise: float = 0.01, cfg_sets=()):
    ck = torch.load(small_ckpt, map_location="cpu", weights_only=False)
    scfg = CharkhaConfig.from_dict(ck["cfg"])
    bcfg = grow_cfg(scfg, mult)
    # post-growth config overrides (e.g. use_recurrence=true use_halting=true): params the
    # small model never had (halt_head, ...) simply stay at the big model's fresh init via the
    # skipped-params path below. Used to re-enable the recurrence family after an r=1 fluency
    # run — pair with train.py --recurrence-curriculum so r ramps from 1 (function-preserving)
    # upward instead of jumping straight to the target depth.
    import ast

    for kv in cfg_sets:
        key, _, val = kv.partition("=")
        key = key.strip()
        if not hasattr(bcfg, key):
            raise ValueError(f"--cfg-set {key}: no such CharkhaConfig field")
        v = val.strip()
        try:
            v = ast.literal_eval({"true": "True", "false": "False"}.get(v.lower(), v))
        except (ValueError, SyntaxError):
            pass
        setattr(bcfg, key, v)
        print(f"[grow] --cfg-set {key} = {v!r}")
    big = Charkha(bcfg)
    sd = ck["model"]
    grown, skipped = 0, []
    big_sd = big.state_dict()
    for name, bt in big_sd.items():
        st = sd.get(name)
        if st is None or any(
            k in name for k in ("loop_embed", "abacus_embed", "precision", "rff_")
        ):
            skipped.append(name)  # rebuilt buffers / absent params keep big init
            continue
        try:
            g = grow_tensor(name, st, bt.shape)
        except AssertionError:
            skipped.append(name)
            continue
        if noise > 0 and g.is_floating_point() and g.ndim >= 2:
            g = g + torch.randn_like(g) * (g.std().clamp_min(1e-8) * noise)
        big_sd[name] = g
        grown += 1
    big.load_state_dict(big_sd)
    return big, bcfg, scfg, grown, skipped, ck


def verify(small_ckpt: str, big: Charkha, mult: int, seq=48, trials=3):
    """Fidelity report: next-token argmax agreement and KL(small || big/m) on random tokens."""
    ck = torch.load(small_ckpt, map_location="cpu", weights_only=False)
    scfg = CharkhaConfig.from_dict(ck["cfg"])
    small = Charkha(scfg)
    small.load_state_dict(ck["model"])
    small.eval()
    big.eval()
    agree, kl = 0.0, 0.0
    with torch.no_grad():
        for i in range(trials):
            torch.manual_seed(100 + i)
            x = torch.randint(0, scfg.vocab_size, (1, seq))
            ls, _ = small(x, r=2)
            lb, _ = big(x, r=2)
            lb = lb / mult  # undo the tied-head temperature factor
            agree += (ls.argmax(-1) == lb.argmax(-1)).float().mean().item() / trials
            ps = torch.log_softmax(ls.float(), -1)
            pb = torch.log_softmax(lb.float(), -1)
            kl += (ps.exp() * (ps - pb)).sum(-1).mean().item() / trials
    return agree, kl


def to_reversible(ckpt_path: str, out_path: str, seq=128, trials=3, r=2):
    """Checkpoint SURGERY: flip cfg.reversible on an existing single-stream checkpoint. The
    parameter set is IDENTICAL (the two-stream coupling reuses each Block's F/G sublayers) so the
    optimizer state carries over — but the computed function changes (streams evolve differently
    than the single stream did). Measures that perturbation before writing: CE + argmax agreement
    between the single-stream and reversible model on the same random batches. Expect a loss bump
    on resume that warm-start training re-heals."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = CharkhaConfig.from_dict(ck["cfg"])
    if getattr(cfg, "reversible", False):
        print("[surgery] checkpoint is already reversible; nothing to do")
        return
    import copy

    rcfg = copy.deepcopy(cfg)
    rcfg.reversible = True
    single, rev = Charkha(cfg), Charkha(rcfg)
    single.load_state_dict(ck["model"])
    rev.load_state_dict(ck["model"])  # identical params — surgery is the cfg bit
    single.eval()
    rev.eval()
    ce_s = ce_r = agree = 0.0
    with torch.no_grad():
        for i in range(trials):
            torch.manual_seed(100 + i)
            x = torch.randint(0, cfg.vocab_size, (1, seq + 1))
            xi, yi = x[:, :-1], x[:, 1:]
            _, l_s = single(xi, yi, r=r)  # targets path returns (None, loss) — fused CE
            _, l_r = rev(xi, yi, r=r)
            ce_s += float(l_s) / trials
            ce_r += float(l_r) / trials
            ls, _ = single(xi, r=r)  # logits need the no-target forward
            lr_, _ = rev(xi, r=r)
            agree += (ls.argmax(-1) == lr_.argmax(-1)).float().mean().item() / trials
    print(
        f"[surgery] CE single {ce_s:.4f} -> reversible {ce_r:.4f} (delta {ce_r - ce_s:+.4f}); "
        f"next-token argmax agreement {agree:.1%} over {trials} random batches"
    )
    out = dict(ck)
    out["cfg"] = dict(rcfg.__dict__)
    out["meta"] = {
        **(ck.get("meta") or {}),
        "reversible_surgery_from": os.path.abspath(ckpt_path),
        "surgery_ce_delta": ce_r - ce_s,
        "surgery_argmax_agree": agree,
    }
    torch.save(out, out_path)
    print(
        f"[surgery] wrote {out_path} @ step {ck.get('step')} — resume with train.py "
        "(optimizer state preserved; params unchanged)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", help="trained mini checkpoint (train.py format)")
    ap.add_argument("--mult", type=int, default=2, help="width multiplier (heads/d_model/d_ff)")
    ap.add_argument("--out", required=True, help="output init checkpoint for the big run")
    ap.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="relative symmetry-breaking noise on grown weights",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="report argmax agreement + KL between small and grown model",
    )
    ap.add_argument(
        "--cfg-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a CharkhaConfig field on the GROWN config (repeatable); "
        "params the small model lacked stay at fresh init",
    )
    ap.add_argument(
        "--to-reversible",
        metavar="CKPT",
        help="checkpoint surgery instead of growth: flip cfg.reversible on CKPT "
        "(same params, measured function delta) and write to --out",
    )
    a = ap.parse_args()
    if a.to_reversible:
        to_reversible(a.to_reversible, a.out)
        return
    if not a.small:
        ap.error("--small is required (unless using --to-reversible)")
    big, bcfg, scfg, grown, skipped, ck = grow(a.small, a.mult, a.noise, a.cfg_set)
    nb = sum(p.numel() for p in big.parameters())
    ns = sum(v.numel() for v in ck["model"].values())
    print(
        f"grew {scfg.d_model}->{bcfg.d_model} d_model ({ns / 1e6:.1f}M -> {nb / 1e6:.1f}M params); "
        f"{grown} tensors grown, {len(skipped)} kept at fresh init"
    )
    if a.verify:
        scale = bcfg.d_model / max(1, scfg.d_model)
        agree, kl = verify(a.small, big, scale)
        print(
            f"fidelity: argmax agreement {agree:.1%}, KL(small||grown/scale) {kl:.4f} "
            f"(1.0 / 0.0 = perfect; scale={scale:.3f}; noise={a.noise} lowers both by design)"
        )
    torch.save(
        {
            "model": big.state_dict(),
            "step": 0,
            "cfg": dict(bcfg.__dict__),
            "opt_mode": ck.get("opt_mode", "default"),
            "opts": None,
            "meta": {
                "grown_from": os.path.abspath(a.small),
                "mult": a.mult,
                "grow_scale": bcfg.d_model / max(1, scfg.d_model),
            },
        },
        a.out,
    )
    print(f"wrote {a.out} — launch with train.py --resume pointing at it (fresh optimizers)")


if __name__ == "__main__":
    main()
