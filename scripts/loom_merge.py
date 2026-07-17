#!/usr/bin/env python3
"""CHARKHA Loom merge: DiLoCo-style same-lineage checkpoint relay.

This is the safe version of home <-> cloud / Kaggle island training:

  1. Start every worker from the same base checkpoint.
  2. Let workers train independently for a short inner window.
  3. Merge worker deltas back into one checkpoint with an outer optimizer.
  4. Redistribute that merged checkpoint and repeat.

It intentionally does NOT merge different architectures. Cross-architecture "model eating" belongs to
KD/Carding, not raw weight averaging; the basis problem is real.

Usage:
  python scripts/loom_merge.py --base runs/main/ckpt.pt --islands kaggle.pt home.pt --out merged.pt
  python scripts/loom_merge.py --base merged_prev.pt --islands island_*.pt --momentum loom_mom.pt --out merged.pt
  python scripts/loom_merge.py --selftest
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import tempfile
from typing import Iterable

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

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


def _cfg_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return dict(obj)
    return dict(vars(obj))


def load_ckpt(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck.get("model", ck)
    cfg = _cfg_dict(ck.get("cfg"))
    return ck, state, cfg


def check_compat(cfgs: Iterable[dict | None], states: Iterable[dict[str, torch.Tensor]]):
    cfgs = list(cfgs)
    states = list(states)
    base_cfg = cfgs[0]
    for i, cfg in enumerate(cfgs[1:], 1):
        if base_cfg is None or cfg is None:
            continue
        for f in _SHAPE_FIELDS:
            if cfg.get(f) != base_cfg.get(f):
                raise ValueError(
                    f"architecture mismatch on {f}: base={base_cfg.get(f)} island{i}={cfg.get(f)}. "
                    "Loom merge requires identical CHARKHA architecture; use KD/Carding across architectures."
                )
    keys = set(states[0])
    for i, st in enumerate(states[1:], 1):
        if set(st) != keys:
            missing = sorted(keys - set(st))[:5]
            extra = sorted(set(st) - keys)[:5]
            raise ValueError(f"state keys differ for island{i}; missing={missing} extra={extra}")
        for k in keys:
            if tuple(st[k].shape) != tuple(states[0][k].shape):
                raise ValueError(
                    f"tensor shape mismatch for {k}: base={tuple(states[0][k].shape)} "
                    f"island{i}={tuple(st[k].shape)}"
                )


def _normalise_weights(n: int, weights: list[float] | None):
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"{len(weights)} weights for {n} islands")
    if any(w < 0 or not math.isfinite(w) for w in weights):
        raise ValueError(f"weights must be finite and non-negative, got {weights}")
    s = sum(weights)
    if s <= 0:
        raise ValueError("at least one island weight must be > 0")
    return [w / s for w in weights]


def _assert_finite_state(name: str, state: dict[str, torch.Tensor]):
    for k, t in state.items():
        if torch.is_floating_point(t) and not torch.isfinite(t).all():
            raise ValueError(
                f"{name} has non-finite tensor {k}; refusing to merge a poisoned checkpoint"
            )


def average_delta(
    base: dict[str, torch.Tensor],
    islands: list[dict[str, torch.Tensor]],
    weights: list[float] | None = None,
):
    """Weighted average of worker weight deltas: mean_i(worker_i - base)."""
    _assert_finite_state("base", base)
    for i, st in enumerate(islands):
        _assert_finite_state(f"island{i}", st)
    w = _normalise_weights(len(islands), weights)
    out = {}
    for k, b in base.items():
        if not torch.is_floating_point(b):
            continue
        acc = torch.zeros_like(b, dtype=torch.float32)
        bf = b.float()
        for wi, st in zip(w, islands):
            acc.add_(st[k].float() - bf, alpha=wi)
        if not torch.isfinite(acc).all():
            raise ValueError(
                f"non-finite averaged delta for {k}; refusing to write merged checkpoint"
            )
        out[k] = acc
    return out


def apply_outer(
    base: dict[str, torch.Tensor],
    avg_delta: dict[str, torch.Tensor],
    velocity: dict[str, torch.Tensor] | None = None,
    outer_lr: float = 1.0,
    outer_momentum: float = 0.0,
):
    """Apply a DiLoCo-like outer step in delta space.

    velocity_t = momentum * velocity_{t-1} + avg_delta
    new_weights = base + outer_lr * velocity_t
    """
    new_state, new_velocity = {}, {}
    for k, b in base.items():
        if k not in avg_delta:
            new_state[k] = b.clone()
            continue
        prev = None if velocity is None else velocity.get(k)
        v = avg_delta[k].clone()
        if prev is not None and outer_momentum:
            v.add_(prev.float(), alpha=outer_momentum)
        new_velocity[k] = v.cpu()
        new_state[k] = (b.float() + outer_lr * v).to(dtype=b.dtype)
    return new_state, new_velocity


def save_ckpt(
    path: str,
    base_ckpt: dict,
    state: dict[str, torch.Tensor],
    cfg: dict | None,
    island_paths: list[str],
    momentum_path: str | None,
):
    meta = dict(base_ckpt.get("meta", {}) or {})
    meta.update(
        {
            "loom_merged": True,
            "loom_islands": [os.path.abspath(p) for p in island_paths],
            "loom_momentum": os.path.abspath(momentum_path) if momentum_path else None,
        }
    )
    payload = {
        "model": state,
        "cfg": cfg if cfg is not None else base_ckpt.get("cfg"),
        "step": int(base_ckpt.get("step", 0)),
        "meta": meta,
    }
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def merge(
    base_path: str,
    island_paths: list[str],
    out_path: str,
    weights=None,
    outer_lr: float = 1.0,
    outer_momentum: float = 0.0,
    momentum_path: str | None = None,
):
    base_ck, base_state, base_cfg = load_ckpt(base_path)
    loaded = [load_ckpt(p) for p in island_paths]
    island_states = [s for _, s, _ in loaded]
    cfgs = [base_cfg] + [c for _, _, c in loaded]
    check_compat(cfgs, [base_state] + island_states)

    avg = average_delta(base_state, island_states, weights)
    mom = None
    if momentum_path and os.path.exists(momentum_path):
        mom_ck = torch.load(momentum_path, map_location="cpu", weights_only=False)
        mom = mom_ck.get("velocity", mom_ck)
    merged, new_mom = apply_outer(base_state, avg, mom, outer_lr, outer_momentum)
    save_ckpt(out_path, base_ck, merged, base_cfg, island_paths, momentum_path)
    if momentum_path:
        torch.save(
            {"velocity": new_mom, "outer_lr": outer_lr, "outer_momentum": outer_momentum},
            momentum_path + ".tmp",
        )
        os.replace(momentum_path + ".tmp", momentum_path)
    return out_path


def _write_ckpt(path, state, cfg):
    torch.save({"model": state, "cfg": cfg, "step": 0, "meta": {}}, path)


def selftest():
    from charkha import Charkha, CharkhaConfig

    print("CHARKHA Loom merge self-test")
    torch.manual_seed(0)
    tmp = tempfile.mkdtemp(prefix="charkha_loom_")
    cfg = CharkhaConfig.toy()
    model = Charkha(cfg)
    base = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def shifted(delta):
        out = {}
        for k, v in base.items():
            out[k] = (v.float() + delta).to(v.dtype) if torch.is_floating_point(v) else v.clone()
        return out

    base_p = os.path.join(tmp, "base.pt")
    a_p = os.path.join(tmp, "a.pt")
    b_p = os.path.join(tmp, "b.pt")
    out_p = os.path.join(tmp, "merged.pt")
    mom_p = os.path.join(tmp, "mom.pt")
    cfgd = dict(cfg.__dict__)
    _write_ckpt(base_p, base, cfgd)
    _write_ckpt(a_p, shifted(1.0), cfgd)
    _write_ckpt(b_p, shifted(3.0), cfgd)

    checks = {}
    merge(base_p, [a_p, b_p], out_p, outer_lr=1.0, momentum_path=mom_p)
    ck = torch.load(out_p, map_location="cpu", weights_only=False)
    k = next(k for k, v in base.items() if torch.is_floating_point(v) and v.ndim > 1)
    checks["average delta applied"] = torch.allclose(ck["model"][k].float(), base[k].float() + 2.0)
    checks["momentum state written"] = os.path.exists(mom_p)
    mdl = Charkha(CharkhaConfig.from_dict(ck["cfg"]))
    mdl.load_state_dict(ck["model"])
    with torch.no_grad():
        logits, _ = mdl(torch.randint(0, cfg.vocab_size, (1, 8)))
    checks["merged checkpoint loads and runs"] = tuple(logits.shape) == (1, 8, cfg.vocab_size)

    # Second outer step: avg delta is still +2, old velocity is +2, momentum .5 => +3 update.
    out2 = os.path.join(tmp, "merged2.pt")
    merge(base_p, [a_p, b_p], out2, outer_lr=1.0, outer_momentum=0.5, momentum_path=mom_p)
    ck2 = torch.load(out2, map_location="cpu", weights_only=False)
    checks["outer momentum accumulates"] = torch.allclose(
        ck2["model"][k].float(), base[k].float() + 3.0
    )

    bad_cfg = dict(cfgd)
    bad_cfg["d_model"] = cfgd["d_model"] * 2
    bad_p = os.path.join(tmp, "bad.pt")
    _write_ckpt(bad_p, shifted(1.0), bad_cfg)
    try:
        merge(base_p, [bad_p], os.path.join(tmp, "bad_out.pt"))
        checks["architecture mismatch refused"] = False
    except ValueError:
        checks["architecture mismatch refused"] = True

    poison = shifted(1.0)
    poison[k] = poison[k].clone()
    poison[k].view(-1)[0] = float("nan")
    poison_p = os.path.join(tmp, "poison.pt")
    _write_ckpt(poison_p, poison, cfgd)
    try:
        merge(base_p, [poison_p], os.path.join(tmp, "poison_out.pt"))
        checks["non-finite island refused"] = False
    except ValueError:
        checks["non-finite island refused"] = True

    ok = True
    for name, passed in checks.items():
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(
        "\nSELFTEST", "PASS - DiLoCo-style same-lineage merge is checkpoint-safe" if ok else "FAIL"
    )
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="CHARKHA Loom / DiLoCo-style checkpoint delta merge")
    ap.add_argument("--base", help="base checkpoint all islands started from")
    ap.add_argument("--islands", nargs="*", help="worker checkpoints to merge; globs allowed")
    ap.add_argument("--weights", nargs="*", type=float, help="optional per-island weights")
    ap.add_argument(
        "--outer-lr", type=float, default=1.0, help="outer step multiplier on averaged delta"
    )
    ap.add_argument(
        "--outer-momentum", type=float, default=0.0, help="momentum on repeated outer deltas"
    )
    ap.add_argument("--momentum", help="path to persistent outer momentum state")
    ap.add_argument("--out", help="merged checkpoint path")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.base or not args.islands or not args.out:
        ap.error("--base, --islands, and --out are required")
    islands = []
    for pat in args.islands:
        hit = sorted(glob.glob(pat))
        islands.extend(hit if hit else [pat])
    if not islands:
        ap.error("no island checkpoints found")
    merge(
        args.base,
        islands,
        args.out,
        weights=args.weights,
        outer_lr=args.outer_lr,
        outer_momentum=args.outer_momentum,
        momentum_path=args.momentum,
    )
    print(f"[loom] merged {len(islands)} islands over {args.base} -> {args.out}")
    if args.momentum:
        print(f"[loom] outer momentum state -> {args.momentum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
