#!/usr/bin/env python3
"""Latent Memory Spine: intrinsic short/long-term memory for CHARKHA.

This module is architecture-side memory, not serving memory. It works inside the
model's hidden states:

  short-term memory: a causal EMA over projected latent states within the sequence;
  long-term memory: a trainable prototype codebook that self-categorizes those states;
  understanding pressure: categories must predict the next latent state, not just copy
  the next token id.

The mechanism is deliberately default-off and exact-no-op at initialization when wired
into CHARKHA: the read/write path computes, but the output projection is zeroed and
the gate starts closed. Training can then decide whether the memory earns influence.

This does not prove "true reasoning." It creates measurable pressure away from pure
rote token memorization: a useful category must compress current context into a stable
state that predicts future representation and uses the codebook without collapse.

Usage:
    python src/latent_memory.py --selftest
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LatentMemoryConfig:
    d_model: int
    slots: int = 64
    mem_dim: int = 0
    decay: float = 0.85
    temp: float = 0.20
    commit_weight: float = 0.02
    pred_weight: float = 0.05
    balance_weight: float = 0.01


class LatentMemorySpine(nn.Module):
    """Self-categorizing latent memory.

    Forward path:
      h_t -> z_t -> short_t = EMA(z_<=t)
      short_t -> assignment over long-term prototypes -> read_t
      h_t <- h_t + gate_t * W_out(read_t)

    Aux losses:
      commit: short-term state and long-term read should agree;
      predict: category read at t predicts z_{t+1}, a representation-level objective;
      balance: batch uses the prototype bank without collapsing to one slot.
    """

    def __init__(self, cfg: LatentMemoryConfig):
        super().__init__()
        if cfg.slots < 2:
            raise ValueError("latent memory needs at least 2 slots")
        self.cfg = cfg
        dmem = int(cfg.mem_dim or max(16, cfg.d_model // 4))
        self.mem_dim = dmem
        self.in_proj = nn.Linear(cfg.d_model, dmem, bias=False)
        self.prototypes = nn.Parameter(torch.randn(cfg.slots, dmem) * (dmem**-0.5))
        self.pred = nn.Linear(dmem, dmem, bias=False)
        self.out_proj = nn.Linear(dmem, cfg.d_model, bias=False)
        self.gate = nn.Linear(cfg.d_model, 1)
        self.force_gate_zero = False
        self.reset_noop()

    def reset_noop(self):
        nn.init.zeros_(self.out_proj.weight)
        nn.init.constant_(self.gate.bias, -4.0)

    def _short_term(self, z, m0=None):
        decay = float(self.cfg.decay)
        if not (0.0 <= decay < 1.0):
            raise ValueError("latent memory decay must be in [0, 1)")
        m = z.new_zeros(z.size(0), z.size(2)) if m0 is None else m0
        outs = []
        alpha = 1.0 - decay
        for t in range(z.size(1)):
            m = decay * m + alpha * z[:, t]
            outs.append(m)
        return torch.stack(outs, dim=1), m

    def forward(self, h, cache=None):
        # cache (streaming decode): {'m': (B, mem_dim) raw EMA state} — the EMA is
        # the spine's ONLY sequential dependency; carrying it makes incremental
        # decode exact versus a full forward (proved in --selftest). None = full.
        z = F.normalize(self.in_proj(h), dim=-1)
        m0 = cache.get("m") if cache is not None else None
        ema, m_last = self._short_term(z, m0)
        if cache is not None:
            cache["m"] = m_last
        short = F.normalize(ema, dim=-1)
        proto = F.normalize(self.prototypes, dim=-1)
        logits = torch.matmul(short, proto.t()) / max(float(self.cfg.temp), 1e-4)
        assign = F.softmax(logits, dim=-1)
        read = torch.matmul(assign, proto)
        pred = F.normalize(self.pred(read), dim=-1)
        write = self.out_proj(read)
        gate = torch.sigmoid(self.gate(h))
        if self.force_gate_zero:
            gate = gate * 0.0
        out = h + gate * write
        info = {
            "z": z,
            "short": short,
            "read": read,
            "assign": assign,
            "pred": pred,
            "gate_mean": gate.detach().mean(),
        }
        return out, info

    def loss(self, info):
        short, read, assign, pred, z = (info[k] for k in ("short", "read", "assign", "pred", "z"))
        commit_a = 1.0 - F.cosine_similarity(short.float(), read.detach().float(), dim=-1).mean()
        commit_b = 1.0 - F.cosine_similarity(read.float(), short.detach().float(), dim=-1).mean()
        commit = 0.5 * (commit_a + commit_b)
        if z.size(1) > 1:
            pred_l = (
                1.0
                - F.cosine_similarity(
                    pred[:, :-1].float(), z[:, 1:].detach().float(), dim=-1
                ).mean()
            )
        else:
            pred_l = z.new_zeros(())
        mean_assign = assign.float().mean(dim=(0, 1)).clamp_min(1e-9)
        balance = (mean_assign * (mean_assign * assign.size(-1)).log()).sum()
        total = (
            float(self.cfg.commit_weight) * commit
            + float(self.cfg.pred_weight) * pred_l
            + float(self.cfg.balance_weight) * balance
        )
        parts = {"mem": total, "mem_commit": commit, "mem_pred": pred_l, "mem_balance": balance}
        return total, parts

    @torch.no_grad()
    def telemetry(self, info):
        assign = info["assign"].float()
        usage = assign.mean(dim=(0, 1)).clamp_min(1e-9)
        entropy = -(usage * usage.log()).sum() / math.log(assign.size(-1))
        top = assign.argmax(-1)
        used = top.unique().numel() / assign.size(-1)
        return {
            "slot_entropy": float(entropy),
            "slot_used_frac": float(used),
            "gate_mean": float(info["gate_mean"]),
        }


def _selftest():
    torch.manual_seed(0)
    checks = 0
    cfg = LatentMemoryConfig(d_model=32, slots=8, mem_dim=12, decay=0.5)
    mem = LatentMemorySpine(cfg)
    x = torch.randn(3, 7, 32)
    y, info = mem(x)
    assert torch.equal(y, x), "exact no-op init failed"
    assert info["assign"].shape == (3, 7, 8)
    assert torch.allclose(info["assign"].sum(-1), torch.ones(3, 7), atol=1e-6)
    checks += 2

    loss, parts = mem.loss(info)
    task = y.pow(2).mean() + loss
    task.backward()
    assert torch.isfinite(task)
    assert mem.in_proj.weight.grad is not None and mem.in_proj.weight.grad.abs().sum() > 0
    assert mem.prototypes.grad is not None and mem.prototypes.grad.abs().sum() > 0
    assert mem.out_proj.weight.grad is not None and mem.out_proj.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(v).all() for v in parts.values())
    checks += 4

    tel = mem.telemetry(info)
    assert 0.0 <= tel["slot_entropy"] <= 1.0 and 0.0 < tel["slot_used_frac"] <= 1.0
    checks += 1

    # Integration smoke: CHARKHA can carry the memory through train and inference.
    sys.path.insert(0, __file__.replace(chr(92), "/").rsplit("/", 1)[0])
    from charkha import Charkha, CharkhaConfig

    ccfg = CharkhaConfig.toy()
    ccfg.use_latent_memory = True
    ccfg.latent_memory_slots = 8
    ccfg.latent_memory_dim = 12
    ccfg.latent_memory_pred_weight = 0.05
    model = Charkha(ccfg)
    xb = torch.randint(0, ccfg.vocab_size, (2, 12))
    _, train_loss = model(xb, xb, r=2)
    train_loss.backward()
    assert torch.isfinite(train_loss) and "mem_pred" in model._last_loss_parts
    grads = [p.grad for p in model.latent_memory.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    model.eval()
    with torch.no_grad():
        logits, conf = model(xb, r=2)
    assert logits.shape == (2, 12, ccfg.vocab_size) and conf.shape == (2, 12)
    assert model._last_memory_info and 0.0 <= model._last_memory_info["slot_entropy"] <= 1.0
    checks += 4

    # Streaming carry state: incremental decode must be EXACT vs a full forward,
    # with the spine deliberately made non-trivial (nonzero write, open gate).
    with torch.no_grad():
        model.latent_memory.out_proj.weight.normal_(0, 0.05)
        model.latent_memory.gate.bias.fill_(0.0)
    model.eval()
    with torch.no_grad():
        full = model.hidden(xb, r=2)
        cache, h_last = model.decode_prefill(xb[:, :8], effort=2)
        assert torch.allclose(h_last, model.hidden(xb[:, :8], r=2)[:, -1:], atol=1e-5), (
            "prefill diverged from full forward with spine active"
        )
        for t in range(8, 12):
            h_last = model.decode_step(xb[:, t : t + 1], cache)
        assert torch.allclose(h_last, full[:, -1:], atol=1e-5), (
            "streaming decode diverged from full forward with spine active"
        )
        assert "m" in cache["latent_memory"], "EMA carry state not persisted"
    checks += 2

    print(f"[selftest] latent_memory.py: all checks passed ({checks} groups)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest())
    ap.print_help()
