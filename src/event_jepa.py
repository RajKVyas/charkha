"""Tiny text-event JEPA scaffold for CHARKHA.

This is not a claim of learned world modeling. It is a cheap auxiliary module and
selftestable objective: encode event text into a hashed latent, predict the next event
latent from the current event plus optional action text, and train by cosine/MSE loss.
"""

from __future__ import annotations
import argparse
import re
import zlib

import torch
import torch.nn as nn
import torch.nn.functional as F


WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def event_vec(text: str, dim: int = 128):
    v = torch.zeros(dim)
    for w in WORD_RE.findall(text.lower()):
        # crc32, not builtin hash(): hash() is salted per process (PYTHONHASHSEED), so latents
        # built in one run would be garbage relative to latents built in another.
        h = zlib.crc32(w.encode())
        v[h % dim] += 1.0 if (h >> 31) & 1 == 0 else -1.0
    return F.normalize(v, dim=0) if v.norm() > 0 else v


class EventJEPA(nn.Module):
    def __init__(self, dim=128, hidden=256):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, current, action):
        return F.normalize(self.net(torch.cat([current, action], dim=-1)), dim=-1)

    def loss(self, current, action, next_event):
        pred = self(current, action)
        tgt = F.normalize(next_event, dim=-1)
        return (1.0 - (pred * tgt).sum(-1)).mean() + 0.05 * F.mse_loss(pred, tgt)


def encode_batch(texts, dim=128):
    return torch.stack([event_vec(t, dim) for t in texts])


def selftest():
    torch.manual_seed(0)
    triples = [
        ("water is cold", "heat water", "water is hot"),
        ("door is closed", "open door", "door is open"),
        ("light is off", "flip switch", "light is on"),
        ("battery is low", "charge battery", "battery is full"),
    ] * 8
    model = EventJEPA(dim=64, hidden=96)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-2)
    cur = encode_batch([a for a, _b, _c in triples], 64)
    act = encode_batch([b for _a, b, _c in triples], 64)
    nxt = encode_batch([c for _a, _b, c in triples], 64)
    with torch.no_grad():
        before = float(model.loss(cur, act, nxt))
    for _ in range(80):
        loss = model.loss(cur, act, nxt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    after = float(model.loss(cur, act, nxt))
    print(f"event_jepa selftest loss {before:.4f} -> {after:.4f}")
    return 0 if after < before * 0.35 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tiny text-event JEPA selftest")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    raise SystemExit(selftest() if args.selftest else 0)
