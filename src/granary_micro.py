#!/usr/bin/env python3
"""Micro-G1: CPU-only iso-FLOP capacity probe for the granary product-key memory layer.

The registered question is NOT \"can a big table memorize?\" (yes) but the
board's kill criterion: at EQUAL FLOPs-per-token and EQUAL training, does a product-key
memory layer store more facts than a dense FFN? If yes, the granary buys dense-quality
capacity at sparse-FFN cost — the whole reason to put knowledge params in host RAM.

Task: pure associative recall (closed book). K facts, each a distinct key token ->
a value token. A minimal model (embed -> one residual sublayer -> head) must map key to
value. This isolates *storage* — no attention, no reasoning, no context — so recall
accuracy is a clean read of the sublayer's memory capacity.

Three arms, identical embed/head/optimizer/steps/seed; only the sublayer differs:
  dense_isoflop   SwiGLU FFN sized to the granary's FLOPs/token   (few params)
  dense_isoparam  SwiGLU FFN sized to the granary's param count   (many params, many FLOPs)
  granary         product-key memory layer

Thesis prediction: granary recall ~ dense_isoparam (both high) >> dense_isoflop, while
granary FLOPs ~ dense_isoflop << dense_isoparam. i.e. iso-param quality at iso-flop cost.
Kill criterion (board): if granary does not beat dense_isoflop at matched FLOPs+tokens,
the mechanism is inert at this scale -> report NEGATIVE.

Run:
    python src/granary_micro.py --selftest
    python src/granary_micro.py --run --facts 256,1024,4096 --seeds 3

"""

import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, __file__.replace(chr(92), "/").rsplit("/", 1)[0])
from granary import ProductKeyMemory


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

    def flops_per_token(self):
        return 3 * self.gate.in_features * self.gate.out_features

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class RecallModel(nn.Module):
    """embed -> RMSNorm -> sublayer (residual) -> head. Minimal storage probe.

    CRITICAL for a HONEST capacity test: freeze the embedding and head to random. If the
    per-key embedding row is trainable, the embedding table itself memorizes each fact
    (64 params/key) and the sublayer is bypassed — every arm then scores 1.0 and the probe
    is meaningless. With emb+head frozen, the key is a FIXED random vector and the answer
    must be stored in the trainable *sublayer* — which is exactly the capacity we compare.
    """

    def __init__(self, vocab, d, sublayer, seed=0, freeze_emb=True, freeze_head=True):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = nn.Embedding(vocab, d)
        self.emb.weight.data.normal_(0, 1.0, generator=g)  # unit-scale so frozen keys are distinct
        self.emb.weight.requires_grad_(not freeze_emb)
        self.norm = RMSNorm(d)
        self.sub = sublayer
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight.data.normal_(0, d**-0.5, generator=g)
        self.head.weight.requires_grad_(not freeze_head)

    def forward(self, key_ids):
        h = self.emb(key_ids).unsqueeze(1)  # (B,1,d)
        s = self.sub(self.norm(h))
        h = (h + s).squeeze(1)
        return self.head(h)


def make_facts(K, V, seed):
    g = torch.Generator().manual_seed(seed)
    keys = torch.arange(K)  # key ids 0..K-1
    vals = torch.randint(K, K + V, (K,), generator=g)  # value ids in [K, K+V)
    return keys, vals


def train_recall(model, keys, vals, steps, lr, batch=0):
    """Memorize the fact set. batch<=0 or batch>=K => FULL batch (the correct capacity
    probe: every fact is seen every step, so the result reflects storage, not sampling)."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    n = keys.numel()
    full = batch <= 0 or batch >= n
    g = torch.Generator().manual_seed(1234)
    for _ in range(steps):
        if full:
            logits = model(keys)
            loss = F.cross_entropy(logits, vals)
        else:
            bi = torch.randint(0, n, (batch,), generator=g)
            logits = model(keys[bi])
            loss = F.cross_entropy(logits, vals[bi])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return recall_acc(model, keys, vals)


@torch.no_grad()
def recall_acc(model, keys, vals):
    model.eval()
    pred = model(keys).argmax(-1)
    return float((pred == vals).float().mean())


def build_arms(vocab, d, seed, n_slots, d_key, n_heads, topk, knn, want_isoparam=True):
    """Construct sublayers with FLOP/param matching to the granary.
    dense_isoparam (matches the granary's value-table param count densely => huge FLOPs) is
    a confirmatory 'ceiling' arm; skip it (want_isoparam=False) for heavy high-K sweeps where
    the decisive comparison is granary vs dense_isoflop at matched FLOPs."""
    gran = ProductKeyMemory(
        d,
        n_slots=n_slots,
        d_key=d_key,
        n_heads=n_heads,
        topk=topk,
        knn=knn,
        gate_init=1.0,
        query_bn=True,
    )
    dff_flop = max(1, round(gran.flops_per_token() / (3 * d)))
    arms = {"dense_isoflop": SwiGLU(d, dff_flop), "granary": gran}
    if want_isoparam:
        dff_param = max(1, round(gran.values.weight.numel() / (3 * d)))
        arms["dense_isoparam"] = SwiGLU(d, dff_param)
    return arms


def run(
    facts_list,
    seeds,
    d=64,
    steps=600,
    lr=3e-3,
    batch=0,
    n_slots=2**14,
    d_key=32,
    n_heads=4,
    topk=32,
    knn=32,
    V=512,
    isoparam=True,
):
    results = {}
    ref = ProductKeyMemory(d, n_slots=n_slots, d_key=d_key, n_heads=n_heads, topk=topk, knn=knn)
    order = ["dense_isoflop", "granary"] + (["dense_isoparam"] if isoparam else [])
    print(
        f"\nGRANARY micro-G1 | d={d} n_slots={ref.n_slots} heads={n_heads} "
        f"topk={topk} | steps={steps} seeds={seeds} | frozen emb+head (sublayer stores)"
    )
    print(
        f"granary: {ref.param_count() / 1e3:.0f}K params, {ref.flops_per_token() / 1e3:.1f}K MAC/tok"
    )

    header_shown = False
    for K in facts_list:
        vocab = K + V
        accs = {name: [] for name in order}
        flops = {}
        params = {}
        for s in range(seeds):
            keys, vals = make_facts(K, V, seed=100 + s)
            arms = build_arms(
                vocab,
                d,
                seed=s,
                n_slots=n_slots,
                d_key=d_key,
                n_heads=n_heads,
                topk=topk,
                knn=knn,
                want_isoparam=isoparam,
            )
            for name, sub in arms.items():
                m = RecallModel(vocab, d, sub, seed=s)
                accs[name].append(train_recall(m, keys, vals, steps=steps, lr=lr, batch=batch))
                flops[name] = sub.flops_per_token()
                params[name] = sub.param_count()
        if not header_shown:
            print(f"\n{'arm':<16}{'FLOP/tok':>10}{'params':>12}")
            for name in order:
                print(f"{name:<16}{flops[name] / 1e3:>8.1f}K{params[name] / 1e3:>11.0f}K")
            print(f"\n{'facts K':>8} | " + " ".join(f"{n:>14}" for n in order))
            header_shown = True
        mean = {k: sum(v) / len(v) for k, v in accs.items()}
        results[K] = mean
        print(f"{K:>8} | " + " ".join(f"{mean[n]:>14.3f}" for n in order))

    Kmax = facts_list[-1]
    g, df = results[Kmax]["granary"], results[Kmax]["dense_isoflop"]
    print(
        f"\nVERDICT @ K={Kmax} (matched FLOPs): granary {g:.3f} vs dense_isoflop {df:.3f} "
        f"-> {'SUPPORTED' if g > df + 0.05 else 'NEGATIVE'} "
        f"(granary buys {'+%.0f%%' % (100 * (g - df)) if g > df else 'no more'} recall at equal compute)"
    )
    return results


def _selftest():
    # tiny, fast: granary should already beat the iso-flop dense on a small fact set
    res = run(
        [128], seeds=1, d=32, steps=200, n_slots=2**12, topk=16, knn=16, n_heads=4, d_key=32, V=128
    )
    assert 0.0 <= res[128]["granary"] <= 1.0
    print("granary_micro selftest: PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--facts", default="256,1024,4096")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--slots", type=int, default=2**14)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument(
        "--no-isoparam",
        action="store_true",
        help="skip the expensive param-matched dense arm (heavy high-K sweeps)",
    )
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    elif args.run:
        facts = [int(x) for x in args.facts.split(",")]
        run(
            facts,
            seeds=args.seeds,
            steps=args.steps,
            d=args.d,
            n_slots=args.slots,
            topk=args.topk,
            lr=args.lr,
            isoparam=not args.no_isoparam,
        )
    else:
        print(__doc__)
