#!/usr/bin/env python3
"""Micro-E10: first end-to-end rehearsal of the accordion contraction (CPU-only).

The accordion invariant requires that a scaffold
teacher discharge its obligation as cached top-k logits BEFORE it dies, so the
mainline can distill at home with ZERO teacher inference. scripts/cache_logits.py
proves the mechanics (round-trip exactness); this file measures whether the cached
logits actually TRANSFER: at a matched step/data budget, does a fresh student
trained on (tokens + cached teacher top-k) beat an identically-initialized student
trained on the same tokens alone?

Protocol (real artifacts, micro budgets):
  teacher   the real 128.5M mini (runs/mini-ts/ckpt.pt) — the project's actual
            fluency scaffold, not a toy stand-in.
  window    cache_teacher() writes top-k over the first N windows of a real
            TinyStories v8 shard (the "rented window" phase). The teacher object
            is DELETED before any student training (zero-teacher property).
  home      two fresh students, identical init (same seed):
              raw  CE on the cached token windows only (compute-matched control).
              kd   CE + the cached-KD tuple via train.py's exact forward contract.
  eval      held-out NLL on later windows of the same shard, unseen by both arms.

Success criterion: kd NLL < raw NLL at every matched budget. A negative result at
micro scale does not kill the mechanism at 0.42B (KD is capacity-sensitive), but a
positive one is the first live evidence that the come-down transfers knowledge.

Usage:
    python src/accordion_micro.py --selftest              # fast structural checks
    python src/accordion_micro.py --run [--windows 240 --steps 150 --seeds 2]

"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from charkha import Charkha, CharkhaConfig  # noqa: E402
from cache_logits import KDShardReader, cache_teacher  # noqa: E402


def student_cfg(vocab_size):
    """Tiny CPU-trainable student sharing the real tokenizer's vocab.
    embed_factor keeps the 131072-row table cheap; ce_chunk keeps CE memory flat."""
    return CharkhaConfig(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        d_ff=352,
        n_prelude=1,
        n_core=2,
        n_coda=1,
        window=64,
        max_seq_len=512,
        embed_factor=32,
        ce_chunk=256,
        use_recurrence=False,
    )


def fresh_student(cfg, seed):
    torch.manual_seed(seed)
    return Charkha(cfg)


def train_arm(model, reader, steps, bs, lr, seed, use_kd):
    """Both arms draw IDENTICAL batches (same rng seed / same reader rows); the
    only difference is whether the cached kd tuple reaches the forward pass."""
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last = None
    for _ in range(steps):
        x, y, kd = reader.batch(bs, "cpu", rng, temp=2.0, weight=1.0)
        _, loss = model(x, y, kd=kd if use_kd else None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss)
    model.eval()
    return last


@torch.no_grad()
def heldout_nll(model, stream, seq_len, n_batches, bs, offset):
    """True next-token NLL on windows past `offset` (never cached, never trained)."""
    import torch.nn.functional as F

    tot, n = 0.0, 0
    for b in range(n_batches):
        rows = []
        for i in range(bs):
            j = offset + (b * bs + i) * (seq_len + 1)
            rows.append(np.asarray(stream[j : j + seq_len + 1], dtype=np.int64))
        x = torch.from_numpy(np.stack(rows))
        h = model.hidden(x)
        h, W = model._head_hw(h)
        logits = F.linear(h[:, :-1].reshape(-1, h.size(-1)), W)
        tot += float(F.cross_entropy(logits, x[:, 1:].reshape(-1), reduction="sum"))
        n += logits.size(0)
    return tot / n


def run(args):
    from cache_logits import _load_charkha

    t0 = time.time()
    teacher, tcfg = _load_charkha(args.teacher, "cpu")
    dtype = np.uint16 if tcfg.vocab_size < 65536 else np.uint32
    stream = np.memmap(args.data, dtype=dtype, mode="r")
    out = args.cache_dir
    if not os.path.exists(os.path.join(out, "index.json")):
        print(f"[cache] teacher writing top-{args.topk} over {args.windows} windows...")
        n = cache_teacher(
            teacher,
            stream,
            out,
            args.seq_len,
            args.topk,
            tcfg.vocab_size,
            limit=args.windows,
            teacher_desc=args.teacher,
        )
        print(f"[cache] {n} windows in {time.time() - t0:.0f}s")
    else:
        print(f"[cache] reusing existing cache at {out}")
    del teacher  # zero-teacher property from here on
    reader = KDShardReader(out)
    heldout_at = (args.windows + 8) * (args.seq_len + 1)

    scfg = student_cfg(tcfg.vocab_size)
    results = {"raw": [], "kd": []}
    for s in range(args.seeds):
        for arm in ("raw", "kd"):
            m = fresh_student(scfg, seed=1000 + s)
            last = train_arm(
                m, reader, args.steps, args.bs, args.lr, seed=2000 + s, use_kd=(arm == "kd")
            )
            nll = heldout_nll(m, stream, args.seq_len, n_batches=4, bs=args.bs, offset=heldout_at)
            results[arm].append(nll)
            print(
                f"[seed {s}] {arm:3s}  final train loss {last:.3f}  "
                f"held-out NLL {nll:.4f}   ({time.time() - t0:.0f}s)"
            )
    mr, mk = (sum(results[a]) / len(results[a]) for a in ("raw", "kd"))
    print(
        json.dumps(
            {
                "raw_mean_nll": round(mr, 4),
                "kd_mean_nll": round(mk, 4),
                "kd_wins": mk < mr,
                "seeds": args.seeds,
                "steps": args.steps,
                "windows": args.windows,
                "seq_len": args.seq_len,
                "topk": args.topk,
            }
        )
    )
    return 0


def _selftest():
    import tempfile

    torch.manual_seed(0)
    checks = 0
    tcfg = CharkhaConfig.toy()
    teacher = Charkha(tcfg).eval()
    T, K, W = 16, 8, 8
    stream = (
        np.random.default_rng(0)
        .integers(0, tcfg.vocab_size, size=(W + 12) * (T + 1))
        .astype(np.uint32)
    )
    out = tempfile.mkdtemp(prefix="acc_micro_")
    n = cache_teacher(teacher, stream, out, T, K, tcfg.vocab_size, limit=W)
    assert n == W
    checks += 1

    del teacher  # students must never touch it
    reader = KDShardReader(out)
    scfg = CharkhaConfig(
        vocab_size=tcfg.vocab_size,
        d_model=64,
        n_heads=2,
        n_kv_heads=1,
        d_ff=176,
        n_prelude=1,
        n_core=1,
        n_coda=1,
        window=32,
        max_seq_len=64,
        use_recurrence=False,
    )
    # identical init across arms
    a = fresh_student(scfg, seed=7)
    b = fresh_student(scfg, seed=7)
    for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)
    checks += 1

    la = train_arm(a, reader, steps=3, bs=2, lr=1e-3, seed=3, use_kd=False)
    lb = train_arm(b, reader, steps=3, bs=2, lr=1e-3, seed=3, use_kd=True)
    assert np.isfinite(la) and np.isfinite(lb)
    assert "kd" in b._last_loss_parts and float(b._last_loss_parts["kd"]) != 0.0
    assert "kd" not in a._last_loss_parts or float(a._last_loss_parts["kd"]) == 0.0
    checks += 2

    nll = heldout_nll(a, stream, T, n_batches=2, bs=2, offset=(W + 2) * (T + 1))
    assert np.isfinite(nll) and nll > 0
    checks += 1

    print(f"[selftest] accordion_micro.py: all checks passed ({checks} groups)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--teacher", type=str, help="teacher checkpoint")
    ap.add_argument("--data", type=str, help="prepared token shard")
    ap.add_argument("--cache-dir", type=str, default="runs/kd_cache_micro")
    ap.add_argument("--windows", type=int, default=240)
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=2)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.run:
        if not a.teacher or not a.data:
            ap.error("--run requires --teacher and --data")
        sys.exit(run(a))
    ap.print_help()
