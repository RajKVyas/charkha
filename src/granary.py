#!/usr/bin/env python3
"""GRANARY: product-key memory (PKM) layer — knowledge parameters that live off the
GPU trunk.

Design: a ~1B model's parameters do double duty as *compute* and
*storage*. Split the roles across a memory hierarchy — keep compute/binding params on
the GPU trunk, move factual *storage* into a large sparse memory whose values can live
in host RAM (millions of slots, read top-k per token, ~tens of MB/s over PCIe at local
tok/s). The mechanism is not new (Lample et al. 2019 "Large Memory Layers with Product
Keys"; Meta "Memory Layers at Scale" 2024/25 — memory layers beat dense-FFN scaling on
factual tasks at iso-FLOP); the novelty here is the *placement* (host-tier params on an
8GB box) and the personal continual-write story (nightly, trunk-frozen).

A ProductKeyMemory replaces (or augments) a SwiGLU FFN:
  * query net maps x -> H heads x d_key
  * each query is split in half; each half is scored against a sqrt(N)-entry sub-key
    codebook, top-`knn` taken per half, the knn x knn candidate full keys re-scored, and
    the final top-k selected. Cost is O(sqrt(N)) in the key count, not O(N).
  * a softmax over the top-k selected slots reads a weighted sum of their value vectors.
Only the top-k value rows per token receive gradient — the backward is sparse, which is
what lets the value table be enormous and host-resident.

Staged-module contract: with `gate_init=0.0` the output is exactly zero at
init, so wiring a granary into an existing checkpoint is a function-preserving
no-op until it is deliberately switched on at a grow point.

Run:
    python src/granary.py --selftest

"""

import argparse
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductKeyMemory(nn.Module):
    """Product-key memory layer. d_model in, d_model out (drop-in for a SwiGLU FFN).

    Args:
        d_model:   trunk width (query input / value output dim).
        n_slots:   total memory slots. Rounded up to a perfect square (product of two
                   sqrt(N) sub-key codebooks); the effective count is n_sub**2.
        d_key:     per-head key dim (split into two d_key//2 halves). Must be even.
        n_heads:   independent lookups, summed. More heads = more read bandwidth, same N.
        topk:      slots read per head per token (the sparse-backward width).
        knn:       candidates kept per sub-key half before the final re-score (knn**2
                   candidate full keys). knn >= topk.
        gate_init: initial value of the scalar output gate. 0.0 => exact no-op at init.
        query_bn:  batch-norm the query before scoring (Lample: spreads key usage, cuts
                   dead slots). Off by default so single-example/eval forwards are exact.
    """

    def __init__(
        self,
        d_model,
        n_slots=2**14,
        d_key=32,
        n_heads=4,
        topk=32,
        knn=32,
        gate_init=0.0,
        query_bn=True,
    ):
        super().__init__()
        assert d_key % 2 == 0, "d_key must be even (split into two halves)"
        self.n_sub = int(math.ceil(math.sqrt(n_slots)))
        self.n_slots = self.n_sub * self.n_sub
        self.d_model, self.d_key, self.n_heads = d_model, d_key, n_heads
        self.topk = min(topk, self.n_slots)
        self.knn = min(max(knn, self.topk), self.n_sub)
        self.half = d_key // 2

        self.query = nn.Linear(d_model, n_heads * d_key, bias=False)
        self.query_bn = nn.BatchNorm1d(n_heads * d_key) if query_bn else None
        # two sub-key codebooks per head: (n_heads, n_sub, half)
        self.keys1 = nn.Parameter(torch.randn(n_heads, self.n_sub, self.half) * (self.half**-0.5))
        self.keys2 = nn.Parameter(torch.randn(n_heads, self.n_sub, self.half) * (self.half**-0.5))
        # value table — this is the "granary". EmbeddingBag would fuse the weighted sum,
        # but a plain Embedding keeps the math explicit and CPU-portable.
        self.values = nn.Embedding(self.n_slots, d_model)
        nn.init.normal_(self.values.weight, std=d_model**-0.5)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        # usage tracking for the load-balance aux loss / dead-slot diagnostics
        self.register_buffer("usage", torch.zeros(self.n_slots), persistent=False)

    def _score_half(self, qh, keys):
        # qh: (M, n_heads, half); keys: (n_heads, n_sub, half) -> (M, n_heads, n_sub)
        return torch.einsum("mhk,hnk->mhn", qh, keys)

    def forward(self, x, return_aux=False):
        B, T, D = x.shape
        M = B * T
        q = self.query(x).reshape(M, self.n_heads * self.d_key)
        if self.query_bn is not None and (self.training and M > 1):
            q = self.query_bn(q)
        q = q.reshape(M, self.n_heads, self.d_key)
        q1, q2 = q[..., : self.half], q[..., self.half :]

        s1 = self._score_half(q1, self.keys1)  # (M, H, n_sub)
        s2 = self._score_half(q2, self.keys2)
        sc1, i1 = s1.topk(self.knn, dim=-1)  # (M, H, knn)
        sc2, i2 = s2.topk(self.knn, dim=-1)
        # combine halves: knn x knn candidate scores and their full-table indices
        cand = (sc1.unsqueeze(-1) + sc2.unsqueeze(-2)).reshape(M, self.n_heads, -1)
        cand_idx = (i1.unsqueeze(-1) * self.n_sub + i2.unsqueeze(-2)).reshape(M, self.n_heads, -1)
        sc, ci = cand.topk(self.topk, dim=-1)  # (M, H, topk)
        idx = torch.gather(cand_idx, -1, ci)  # (M, H, topk) slot ids
        w = F.softmax(sc, dim=-1)  # (M, H, topk)

        v = self.values(idx)  # (M, H, topk, D)
        out = (w.unsqueeze(-1) * v).sum(dim=2).sum(dim=1)  # sum topk then heads -> (M, D)
        out = (self.gate * out).reshape(B, T, D)

        if self.training:
            with torch.no_grad():
                flat = idx.reshape(-1)
                self.usage.index_add_(0, flat, torch.ones_like(flat, dtype=self.usage.dtype))
        if not return_aux:
            return out
        # load-balance aux: soft assignment mass per slot, KL to uniform (encourage spread)
        with torch.no_grad():
            frac = self.usage / self.usage.sum().clamp_min(1.0)
            dead = float((self.usage == 0).float().mean())
        aux = {"dead_frac": dead, "usage_cv": float(frac.std() / frac.mean().clamp_min(1e-9))}
        return out, aux

    def reset_usage(self):
        self.usage.zero_()

    def flops_per_token(self):
        """Rough MAC estimate per token (query proj + key scoring + value read)."""
        q = self.d_model * self.n_heads * self.d_key
        keyscore = self.n_heads * self.n_sub * self.d_key  # both halves
        valread = self.n_heads * self.topk * self.d_model
        return q + keyscore + valread

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def _selftest():
    torch.manual_seed(0)
    d = 64
    mem = ProductKeyMemory(d, n_slots=4096, d_key=32, n_heads=4, topk=16, knn=16, gate_init=0.0)
    x = torch.randn(2, 5, d)

    # 1) exact no-op at init (gate=0)
    mem.eval()
    y0 = mem(x)
    assert torch.allclose(y0, torch.zeros_like(y0)), "gate_init=0 must be an exact no-op"
    print(f"[ok] exact no-op at init  (max|out|={y0.abs().max():.2e})")

    # 2) shape + nonzero once the gate opens
    with torch.no_grad():
        mem.gate.fill_(1.0)
    y1 = mem(x)
    assert y1.shape == x.shape, y1.shape
    assert y1.abs().max() > 0
    print(f"[ok] shape {tuple(y1.shape)} and nonzero once gate opens")

    # 3) SPARSE backward — only touched value rows get gradient
    mem.train()
    mem.reset_usage()
    y, aux = mem(x, return_aux=True)
    y.pow(2).mean().backward()
    g = mem.values.weight.grad
    touched = (g.abs().sum(-1) > 0).sum().item()
    total = mem.n_slots
    tk_cap = 2 * 5 * mem.n_heads * mem.topk
    assert 0 < touched <= tk_cap, (touched, tk_cap)
    print(f"[ok] sparse backward: {touched}/{total} value rows got grad (cap {tk_cap})")

    # 4) determinism (eval, bn off path)
    mem.eval()
    a, b = mem(x), mem(x)
    assert torch.allclose(a, b)
    print("[ok] deterministic in eval")

    # 5) capacity/FLOP accounting sanity
    big = ProductKeyMemory(512, n_slots=2**18, d_key=32, n_heads=4, topk=32, knn=32)
    dense_ff = 3 * 512 * (512 * 4)  # a SwiGLU with d_ff=4d
    print(
        f"[ok] 2^18-slot granary @ d512: {big.param_count() / 1e6:.1f}M params, "
        f"~{big.flops_per_token() / 1e3:.1f}K MAC/tok  vs  dense-FFN(4d) ~{dense_ff / 1e3:.1f}K MAC/tok"
    )
    print(
        f"     -> {big.param_count() / (dense_ff):.1f}x the params at "
        f"{big.flops_per_token() / dense_ff:.2f}x the FLOPs"
    )
    print("granary selftest: PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print(__doc__)
        sys.exit(0)
