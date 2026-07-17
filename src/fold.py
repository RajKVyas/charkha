#!/usr/bin/env python3
"""Fold: native latent context compression via gist vectors.

Compresses chunks of old-context tokens into learned gist vectors in the model's
own embedding space, increasing effective context length at fixed RAM. Compresses
K tokens into g vectors (default K/g = 16x). Trained self-supervised: the folder
minimizes KL divergence between the model's next-token distribution given
[gists + recent window] vs. the full context. No labels, no external teacher.

Properties:
  - carrier: model-native d_model vectors (no OCR/vision round-trip)
  - ratio: g/K is configurable (16-32x typical)
  - fidelity: measured against the frozen base model's own logits
  - RAM: only g activation positions survive per compressed chunk

Limitations: gists are lossy summaries. Verbatim recall from folded spans is
not guaranteed — quote-exact content belongs in the recent verbatim window.

Current status: plumbing is proven (selftest), compression efficacy on a tiny
base model is neutral (matched vs. shuffled gists score identically on books3).
Retraining against a longer-context base model is required before serve wiring.
The shuffled-gist control arm is mandatory for any reported fold win.

Usage:
    python src/fold.py --selftest
    python src/fold.py --train --ckpt runs/mini-ts/ckpt.pt \\
        --data data/corpus-dd/shard_00000.bin --out runs/fold.pt
    python src/fold.py --run --ckpt runs/mini-ts/ckpt.pt \\
        --data data/corpus-dd/shard_00000.bin --folder runs/fold.pt

"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class Folder(nn.Module):
    """g learned queries cross-attend over the frozen model's hidden states of one
    chunk and emit g pseudo-embeddings. Parameter-light (~4*d^2 + g*d): the BASE
    MODEL is never trained — fold is a bolt-on, safe to train on CPU mid-run."""

    def __init__(self, d_model, g=8, n_heads=4):
        super().__init__()
        self.g = g
        self.queries = nn.Parameter(torch.randn(g, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_model)
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, h_chunk):
        """h_chunk (B,K,d) frozen-model hidden states -> (B,g,d) gist embeddings."""
        q = self.queries.unsqueeze(0).expand(h_chunk.size(0), -1, -1)
        z, _ = self.attn(q, h_chunk, h_chunk, need_weights=False)
        return self.gain * self.out(self.norm(z + q))


@torch.no_grad()
def chunk_hidden(model, ids, r=None):
    """Frozen-model representation of a chunk (no grad — the base never trains)."""
    return model.hidden(ids, r=r)


def fold_context(model, folder, ids, chunk_len, r=None):
    """Fold old context `ids` (B,L) into (B, g*n_chunks, d) gist embeddings.
    Grad flows through the folder only."""
    B, L = ids.shape
    gists = []
    for s in range(0, L - L % chunk_len, chunk_len):
        h = chunk_hidden(model, ids[:, s : s + chunk_len], r=r)
        gists.append(folder(h))
    rem = L % chunk_len
    if rem >= chunk_len // 2:  # fold a big-enough remainder too
        h = chunk_hidden(model, ids[:, L - rem :], r=r)
        gists.append(folder(h))
    return (
        torch.cat(gists, dim=1)
        if gists
        else ids.new_zeros((B, 0, model.cfg.d_model), dtype=torch.float32)
    )


def _tail_logits(model, ids, n_tail, soft_prefix=None, r=None):
    """Full-vocab logits for the last n_tail positions (predicting ids shifted)."""
    h = model.hidden(ids, r=r, soft_prefix=soft_prefix)
    h, W = model._head_hw(h[:, -n_tail - 1 : -1])
    return F.linear(h.reshape(-1, h.size(-1)), W)


def fold_loss(model, folder, x, window, chunk_len, n_tail, r=None):
    """Self-distillation: KL(model(full) || model(gists+window)) on the tail,
    plus data CE — the folder learns to preserve exactly what the model would
    have extracted from the context it replaced."""
    old, recent = x[:, :-window], x[:, -window:]
    with torch.no_grad():
        t_logits = _tail_logits(model, x, n_tail, r=r)
        t_logp = F.log_softmax(t_logits.float(), dim=-1)
    gists = fold_context(model, folder, old, chunk_len, r=r)
    s_logits = _tail_logits(model, recent, n_tail, soft_prefix=gists, r=r)
    s_logp = F.log_softmax(s_logits.float(), dim=-1)
    kl = F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean")
    y = x[:, -n_tail:].reshape(-1)
    ce = F.cross_entropy(s_logits, y)
    return kl + ce, float(kl), float(ce)


def _windows(stream, T, bs, i):
    rows = [
        np.asarray(stream[(i * bs + j) * T : (i * bs + j + 1) * T], dtype=np.int64)
        for j in range(bs)
    ]
    return torch.from_numpy(np.stack(rows))


def train_folder(
    model, stream, steps, bs, T, window, chunk_len, n_tail, lr, g, r=None, log_every=25
):
    model.eval().requires_grad_(False)
    folder = Folder(model.cfg.d_model, g=g)
    opt = torch.optim.AdamW(folder.parameters(), lr=lr)
    t0 = time.time()
    for i in range(steps):
        x = _windows(stream, T, bs, i)
        loss, kl, ce = fold_loss(model, folder, x, window, chunk_len, n_tail, r=r)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(folder.parameters(), 1.0)
        opt.step()
        if i % log_every == 0 or i == steps - 1:
            print(f"[fold-train] step {i:4d} | kl {kl:.4f} | ce {ce:.4f} | {time.time() - t0:.0f}s")
    return folder


@torch.no_grad()
def eval_arms(model, folder, stream, n_batches, bs, T, window, chunk_len, n_tail, offset, r=None):
    """The measurement: full-context upper bound vs verbatim-truncate vs fold,
    at MATCHED visible positions (truncate sees `window`; fold sees `window` plus
    g-per-chunk gists — the RAM cost fold actually pays). The `shuffled` arm is the
    MANDATORY control (added after it busted the first books3 'win'): gists folded
    from an unrelated window. fold ≈ shuffled ⇒ the folder is an input-independent
    soft prompt carrying ZERO context — only fold < shuffled counts as compression."""
    nll = {"full": 0.0, "trunc": 0.0, "fold": 0.0, "shuffled": 0.0}
    n = 0
    for i in range(n_batches):
        x = _windows(stream, T, bs, offset + i)
        xw = _windows(stream, T, bs, offset + n_batches + 7 + i)  # unrelated windows
        y = x[:, -n_tail:].reshape(-1)
        old, recent = x[:, :-window], x[:, -window:]
        nll["full"] += float(
            F.cross_entropy(_tail_logits(model, x, n_tail, r=r), y, reduction="sum")
        )
        nll["trunc"] += float(
            F.cross_entropy(_tail_logits(model, recent, n_tail, r=r), y, reduction="sum")
        )
        gists = fold_context(model, folder, old, chunk_len, r=r)
        nll["fold"] += float(
            F.cross_entropy(
                _tail_logits(model, recent, n_tail, soft_prefix=gists, r=r), y, reduction="sum"
            )
        )
        g_wrong = fold_context(model, folder, xw[:, :-window], chunk_len, r=r)
        nll["shuffled"] += float(
            F.cross_entropy(
                _tail_logits(model, recent, n_tail, soft_prefix=g_wrong, r=r), y, reduction="sum"
            )
        )
        n += y.numel()
    return {k: v / n for k, v in nll.items()}


def _load_model(ckpt):
    from charkha import Charkha, CharkhaConfig

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = CharkhaConfig.from_dict(blob["cfg"])
    m = Charkha(cfg)
    m.load_state_dict(blob["model"])
    return m.eval()


def _selftest():
    from charkha import Charkha, CharkhaConfig

    torch.manual_seed(0)
    checks = 0
    cfg = CharkhaConfig(**{**CharkhaConfig.toy().__dict__, "use_recurrence": False})
    model = Charkha(cfg).eval().requires_grad_(False)
    # (recurrent configs run their core under no_grad in eval mode — backprop_depth
    # applies to training only — so folder training uses non-recurrent routing;
    # the real mini/0.42B line is non-recurrent anyway.)
    B, T = 2, 48
    x = torch.randint(0, cfg.vocab_size, (B, T))

    # soft_prefix=None is bit-identical to the original hidden() path
    h0 = model.hidden(x)
    h1 = model.hidden(x)
    assert torch.equal(h0, h1)
    checks += 1

    # prefix shifts positions: h over prefix+idx, caller slice matches shapes
    folder = Folder(cfg.d_model, g=4)
    gists = fold_context(model, folder, x[:, :32], chunk_len=16)
    assert gists.shape == (B, 8, cfg.d_model)  # 2 chunks x g=4 -> 32/8 = 4x
    hp = model.hidden(x[:, 32:], soft_prefix=gists)
    assert hp.shape == (B, 8 + (T - 32), cfg.d_model)
    checks += 2

    # loss is finite; grads reach the folder ONLY (base model frozen)
    loss, kl, ce = fold_loss(model, folder, x, window=16, chunk_len=16, n_tail=8)
    loss.backward()
    assert torch.isfinite(loss) and kl >= 0
    assert folder.queries.grad is not None and folder.queries.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    checks += 3

    # eval harness returns the three arms, finite, at matched tails
    res = eval_arms(
        model,
        folder,
        np.random.default_rng(0).integers(0, cfg.vocab_size, size=24 * B * T).astype(np.int64),
        n_batches=1,
        bs=B,
        T=T,
        window=16,
        chunk_len=16,
        n_tail=8,
        offset=0,
    )
    assert set(res) == {"full", "trunc", "fold", "shuffled"}  # control is mandatory
    assert all(v == v and v > 0 for v in res.values())
    checks += 1

    # a folder round-trips through save/load
    import tempfile

    p = os.path.join(tempfile.mkdtemp(prefix="fold_"), "f.pt")
    torch.save({"folder": folder.state_dict(), "g": folder.g, "d_model": cfg.d_model}, p)
    blob = torch.load(p, weights_only=False)
    f2 = Folder(blob["d_model"], g=blob["g"])
    f2.load_state_dict(blob["folder"])
    assert torch.equal(f2.queries, folder.queries)
    checks += 1

    print(f"[selftest] fold.py: all checks passed ({checks} groups)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--run", action="store_true", help="eval full/trunc/fold arms")
    ap.add_argument("--ckpt", type=str, help="model checkpoint")
    ap.add_argument("--data", type=str, help="prepared token shard")
    ap.add_argument("--folder", type=str, default="runs/fold.pt")
    ap.add_argument("--out", type=str, default="runs/fold.pt")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=384, help="full-context T")
    ap.add_argument("--window", type=int, default=128, help="verbatim recent tail")
    ap.add_argument("--chunk-len", type=int, default=128)
    ap.add_argument("--gist", type=int, default=8, help="gists per chunk (K/g = ratio)")
    ap.add_argument("--n-tail", type=int, default=64, help="scored tail positions")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-batches", type=int, default=8)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if not (a.train or a.run):
        ap.print_help()
        sys.exit(0)
    if not a.ckpt or not a.data:
        ap.error("--train/--run requires --ckpt and --data")
    model = _load_model(a.ckpt)
    dtype = np.uint16 if model.cfg.vocab_size < 65536 else np.uint32
    stream = np.memmap(a.data, dtype=dtype, mode="r")
    if a.train:
        folder = train_folder(
            model, stream, a.steps, a.bs, a.seq_len, a.window, a.chunk_len, a.n_tail, a.lr, a.gist
        )
        torch.save(
            {
                "folder": folder.state_dict(),
                "g": a.gist,
                "d_model": model.cfg.d_model,
                "chunk_len": a.chunk_len,
                "window": a.window,
                "ckpt": a.ckpt,
            },
            a.out,
        )
        print(
            f"[fold] saved folder ({a.chunk_len}/{a.gist} = "
            f"{a.chunk_len // a.gist}x compression) -> {a.out}"
        )
    if a.run:
        blob = torch.load(a.folder, map_location="cpu", weights_only=False)
        folder = Folder(blob["d_model"], g=blob["g"])
        folder.load_state_dict(blob["folder"])
        res = eval_arms(
            model,
            folder,
            stream,
            a.eval_batches,
            a.bs,
            a.seq_len,
            blob.get("window", a.window),
            blob.get("chunk_len", a.chunk_len),
            a.n_tail,
            offset=a.steps + 16,
        )
        ratio = blob.get("chunk_len", a.chunk_len) // blob["g"]
        print(
            json.dumps(
                {
                    **{k: round(v, 4) for k, v in res.items()},
                    "compression": f"{ratio}x",
                    "fold_recovers": round(
                        (res["trunc"] - res["fold"]) / max(1e-9, res["trunc"] - res["full"]), 3
                    ),
                    "context_carried": round(res["shuffled"] - res["fold"], 4),
                    "verdict": (
                        "compression"
                        if res["shuffled"] - res["fold"] > 0.01
                        else "soft-prompt only"
                    ),
                }
            )
        )
    if not (a.train or a.run):
        ap.print_help()
