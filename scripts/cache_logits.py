#!/usr/bin/env python3
"""Pre-compute a teacher's top-k logits over a token corpus for offline distillation.

Before rented compute expires, the teacher writes its per-position top-k
distribution over the curated corpus. At home, train.py --kd-shards distills
from the cache with zero teacher inference — the cloud GPU is then freed.

Shard format (self-contained; windows are the dataset, so student batches are always
position-aligned with the cached teacher):
  <out>/kd_tokens.u32   flat uint32, n_windows * (seq_len+1) tokens (x = [:T], y = [1:])
  <out>/kd_idx.i32      flat int32,  n_windows * seq_len * topk teacher top-k token ids
  <out>/kd_prob.f16     flat float16, same shape — teacher top-k probabilities
  <out>/index.json      {n_windows, seq_len, topk, vocab_size, teacher}
The tuple handed to Charkha.forward(kd=...) is (t_idx, t_prob, temp, weight) — the
same contract distill.teacher_kd_tuple produces from a live teacher.

Usage:
    python scripts/cache_logits.py --ckpt runs/<scaffold>/ckpt.pt \
        --data data/<source>-dd/shard_0000.bin --out runs/kd_cache \
        --seq-len 512 --topk 32 [--limit N] [--device cuda]
    python scripts/cache_logits.py --selftest

"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


class KDShardWriter:
    """Append-only writer for the cached-KD shard format."""

    def __init__(self, out_dir, seq_len, topk, vocab_size, teacher_desc=""):
        os.makedirs(out_dir, exist_ok=True)
        self.dir, self.seq_len, self.topk = out_dir, seq_len, topk
        self.vocab_size, self.teacher_desc = vocab_size, teacher_desc
        self.n = 0
        self._tok = open(os.path.join(out_dir, "kd_tokens.u32"), "wb")
        self._idx = open(os.path.join(out_dir, "kd_idx.i32"), "wb")
        self._prob = open(os.path.join(out_dir, "kd_prob.f16"), "wb")

    def add_window(self, tokens, t_idx, t_prob):
        """tokens: (seq_len+1,) ids; t_idx/t_prob: (seq_len, topk) teacher top-k."""
        tokens = np.asarray(tokens, dtype=np.uint32)
        t_idx = np.asarray(t_idx, dtype=np.int32)
        t_prob = np.asarray(t_prob, dtype=np.float16)
        assert tokens.shape == (self.seq_len + 1,)
        assert t_idx.shape == (self.seq_len, self.topk) and t_prob.shape == t_idx.shape
        self._tok.write(tokens.tobytes())
        self._idx.write(t_idx.tobytes())
        self._prob.write(t_prob.tobytes())
        self.n += 1

    def finalize(self):
        for f in (self._tok, self._idx, self._prob):
            f.close()
        with open(os.path.join(self.dir, "index.json"), "w") as f:
            json.dump(
                {
                    "n_windows": self.n,
                    "seq_len": self.seq_len,
                    "topk": self.topk,
                    "vocab_size": self.vocab_size,
                    "teacher": self.teacher_desc,
                },
                f,
                indent=1,
            )
        return self.n


class KDShardReader:
    """Memory-mapped reader; yields (x, y, kd_tuple) batches for train.py."""

    def __init__(self, kd_dir):
        with open(os.path.join(kd_dir, "index.json")) as f:
            self.meta = json.load(f)
        n, T, k = self.meta["n_windows"], self.meta["seq_len"], self.meta["topk"]
        if n < 1:
            raise ValueError(f"empty KD cache at {kd_dir}")
        self.tok = np.memmap(
            os.path.join(kd_dir, "kd_tokens.u32"), dtype=np.uint32, mode="r", shape=(n, T + 1)
        )
        self.idx = np.memmap(
            os.path.join(kd_dir, "kd_idx.i32"), dtype=np.int32, mode="r", shape=(n, T, k)
        )
        self.prob = np.memmap(
            os.path.join(kd_dir, "kd_prob.f16"), dtype=np.float16, mode="r", shape=(n, T, k)
        )

    @property
    def n_windows(self):
        return self.meta["n_windows"]

    def batch(self, bs, device, rng, temp=1.0, weight=1.0):
        """Random KD windows -> (x (B,T), y (B,T), kd tuple) on `device`.
        The kd tensors are sliced to T-1 positions because Charkha.forward applies
        KD against h[:, :-1] (same convention as a live teacher_kd_tuple)."""
        rows = [rng.randrange(self.n_windows) for _ in range(bs)]
        tok = torch.from_numpy(np.array(self.tok[rows], dtype=np.int64))
        x, y = tok[:, :-1].to(device), tok[:, 1:].to(device)
        t_idx = torch.from_numpy(np.array(self.idx[rows], dtype=np.int64))
        t_prob = torch.from_numpy(np.array(self.prob[rows])).float()
        return x, y, (t_idx[:, :-1].to(device), t_prob[:, :-1].to(device), temp, weight)


@torch.no_grad()
def cache_teacher(
    model, token_stream, out_dir, seq_len, topk, vocab_size, device="cpu", limit=0, teacher_desc=""
):
    """Slide non-overlapping (seq_len+1) windows over `token_stream` (1-D array) and
    cache the teacher's top-k next-token distribution at every position.
    KD targets align with y: position t stores the distribution for predicting
    token t+1, exactly what Charkha._kd_topk consumes against h[:, :-1]."""
    model.eval()
    w = KDShardWriter(out_dir, seq_len, topk, vocab_size, teacher_desc)
    n_max = (len(token_stream) - 1) // (seq_len + 1)
    if limit:
        n_max = min(n_max, limit)
    for i in range(n_max):
        window = np.asarray(
            token_stream[i * (seq_len + 1) : (i + 1) * (seq_len + 1)], dtype=np.int64
        )
        x = torch.from_numpy(window[:-1]).unsqueeze(0).to(device)
        logits, _ = model(x)
        probs = F.softmax(logits[0].float(), dim=-1)  # (T, V)
        p, ix = probs.topk(topk, dim=-1)  # (T, k)
        w.add_window(window, ix.cpu().numpy(), p.cpu().numpy())
    return w.finalize()


def _load_charkha(ckpt_path, device):
    from charkha import Charkha, CharkhaConfig

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = (
        blob["cfg"]
        if isinstance(blob.get("cfg"), CharkhaConfig)
        else CharkhaConfig.from_dict(blob["cfg"])
    )  # tolerant: old ckpts carry removed keys
    model = Charkha(cfg)
    model.load_state_dict(blob.get("ema", blob.get("model", blob)), strict=True)
    return model.to(device).eval(), cfg


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------


def _selftest():
    import random
    import tempfile
    from charkha import Charkha, CharkhaConfig

    torch.manual_seed(0)
    checks = 0
    cfg = CharkhaConfig.toy()
    teacher = Charkha(cfg).eval()
    T, K = 12, 8
    stream = (
        np.random.default_rng(0).integers(0, cfg.vocab_size, size=6 * (T + 1) + 3).astype(np.uint32)
    )
    out = tempfile.mkdtemp(prefix="kd_cache_")

    n = cache_teacher(
        teacher, stream, out, seq_len=T, topk=K, vocab_size=cfg.vocab_size, teacher_desc="toy"
    )
    assert n == 6 and os.path.exists(os.path.join(out, "index.json"))
    checks += 1

    # round-trip exactness: cached top-k == a fresh teacher forward's top-k
    r = KDShardReader(out)
    assert r.n_windows == 6 and r.meta["topk"] == K
    win = torch.from_numpy(stream[: T + 1].astype(np.int64))
    with torch.no_grad():
        logits, _ = teacher(win[:-1].unsqueeze(0))
    p_ref, i_ref = F.softmax(logits[0].float(), -1).topk(K, dim=-1)
    assert np.array_equal(np.array(r.idx[0]), i_ref.numpy().astype(np.int32))
    assert np.allclose(
        np.array(r.prob[0], dtype=np.float32), p_ref.numpy(), atol=1e-3
    )  # fp16 storage tolerance
    checks += 2

    # student consumption: model(x, y, kd=tuple) produces a finite kd loss + grads
    rng = random.Random(0)
    x, y, kd = r.batch(2, "cpu", rng, temp=2.0, weight=0.5)
    assert x.shape == (2, T) and kd[0].shape == (2, T - 1, K) and kd[2] == 2.0
    assert torch.equal(x[:, 1:], y[:, :-1])  # windows are contiguous
    student = Charkha(cfg)
    student.train()
    _, loss = student(x, y, kd=kd, r=2)
    loss.backward()
    assert torch.isfinite(loss) and "kd" in student._last_loss_parts
    assert float(student._last_loss_parts["kd"]) != 0.0
    checks += 3

    # zero-teacher property: the batch path never touches the teacher object
    del teacher
    x2, y2, kd2 = r.batch(1, "cpu", rng)
    assert x2.shape == (1, T) and torch.isfinite(kd2[1]).all()
    checks += 1

    print(f"[selftest] cache_logits.py: all checks passed ({checks} groups)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", type=str, help="teacher checkpoint (Charkha .pt)")
    ap.add_argument("--data", type=str, help="raw token .bin (uint16/uint32 by vocab)")
    ap.add_argument("--out", type=str, help="output KD cache dir")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="max windows (0 = all)")
    ap.add_argument("--device", type=str, default="cpu")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if not (a.ckpt and a.data and a.out):
        ap.print_help()
        sys.exit(1)
    model, cfg = _load_charkha(a.ckpt, a.device)
    dtype = np.uint16 if cfg.vocab_size < 65536 else np.uint32
    stream = np.memmap(a.data, dtype=dtype, mode="r")
    n = cache_teacher(
        model,
        stream,
        a.out,
        a.seq_len,
        a.topk,
        cfg.vocab_size,
        device=a.device,
        limit=a.limit,
        teacher_desc=a.ckpt,
    )
    print(f"[cache_logits] wrote {n} windows (seq_len={a.seq_len}, topk={a.topk}) -> {a.out}")
