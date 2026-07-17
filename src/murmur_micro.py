#!/usr/bin/env python3
"""Micro-E5: the first capability measurement for the murmur register (CPU-only).

E5 asks whether murmur is code or merely compute:
pause tokens, else it reinvented \\cite{goyal2023pause} and demotes itself. The full
E5 is GPU-gated on the 0.42B line; this file runs the same registered comparison at
micro scale — a toy Charkha on a synthetic checkable task (sort a digit string),
entirely on CPU, without touching any training run.

Protocol (all arms share one pretrained checkpoint and identical fine-tune budgets):
  none    continue plain prompt->answer training (compute-matched control).
  pause   k content-free tokens (one fixed band id) inserted before the answer;
          CE on the answer only — the published null hypothesis.
  murmur  bootstrap loop: sample k-token murmur blocks under the band mask, accept
          those that raise teacher-forced answer logprob, train CE on accepted
          murmur spans + answers; when nothing is accepted, train the plain trace.
Eval: held-out exact-match of the greedy-decoded answer (murmur arm decodes its
block first, then the answer under the visible mask). Band-usage entropy is logged
each round (degenerate-code alarm).

This is evidence at MICRO scale only: a positive result here does not establish the
0.42B claim, and a negative one does not kill it (STaR-family methods are
scale-sensitive in both directions). It is reported in the paper as exactly that.

Usage:
    python src/murmur_micro.py --selftest          # fast structural checks
    python src/murmur_micro.py --run --seeds 3     # the experiment (minutes, CPU)

"""

import argparse
import copy
import json
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from murmur import (
    extend_vocab,
    mask_logits,
    generate_murmur_block,
    answer_logprob,
    murmur_positions,
    band_usage_entropy,
)


# ---------------------------------------------------------------------------
# Task: sort a digit string (multi-step latent computation, checkable answer)
# ---------------------------------------------------------------------------

from micro_task import DIGITS, SEP, EOS, make_pair, make_split  # noqa: F401


# ---------------------------------------------------------------------------
# Training primitives (manual masked CE — aux losses would muddy the comparison)
# ---------------------------------------------------------------------------


def ce_on_positions(model, ids, train_from):
    """CE over predicting ids[train_from:] from their prefixes (one forward)."""
    idx = torch.tensor([ids], dtype=torch.long)
    logits, _ = model(idx)
    lp = logits[0, train_from - 1 : len(ids) - 1]
    tgt = torch.tensor(ids[train_from:], dtype=torch.long)
    return F.cross_entropy(lp.float(), tgt)


def ce_masked(model, ids, mask):
    """CE only at positions where mask[t] is True (targets are ids[t])."""
    idx = torch.tensor([ids], dtype=torch.long)
    logits, _ = model(idx)
    pos = [t for t in range(1, len(ids)) if mask[t]]
    lp = logits[0, [t - 1 for t in pos]]
    tgt = torch.tensor([ids[t] for t in pos], dtype=torch.long)
    return F.cross_entropy(lp.float(), tgt)


@torch.no_grad()
def greedy_answer(model, ctx, band, n_ans):
    """Greedy-decode n_ans tokens under the visible-phase mask."""
    idx = torch.tensor([list(ctx)], dtype=torch.long)
    out = []
    for _ in range(n_ans):
        logits, _ = model(idx)
        logits = mask_logits(logits[:, -1, :], band, "visible")
        t = int(logits.argmax(-1))
        out.append(t)
        idx = torch.cat([idx, torch.tensor([[t]])], dim=1)
    return out


@torch.no_grad()
def eval_arm(model, split, band, arm, k, gen):
    """Held-out exact-match accuracy for one arm."""
    model.eval()
    hits = 0
    for prompt, answer in split:
        if arm == "pause":
            ctx = prompt + [band.start + 2] * k
        elif arm == "murmur":
            blk = generate_murmur_block(model, prompt, band, max_len=k, gen=gen)
            ctx = prompt + blk
        else:
            ctx = prompt
        if greedy_answer(model, ctx, band, len(answer)) == answer:
            hits += 1
    return hits / len(split)


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------


def pretrain(model, band, split, steps, lr, gen):
    """Shared phase: plain prompt->answer CE (no band ids anywhere — the band
    stays untouched, preserving imitation-freedom for the murmur arm)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        prompt, answer = split[int(torch.randint(0, len(split), (1,), generator=gen))]
        loss = ce_on_positions(model, prompt + answer, len(prompt))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def finetune_arm(model, band, split, arm, rounds, per_round, k, lr, gen):
    """Matched budget: rounds*per_round optimizer steps for every arm."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log = []
    for r in range(rounds):
        model.train()
        blocks, n_acc = [], 0
        for _ in range(per_round):
            prompt, answer = split[int(torch.randint(0, len(split), (1,), generator=gen))]
            if arm == "none":
                loss = ce_on_positions(model, prompt + answer, len(prompt))
            elif arm == "pause":
                ids = prompt + [band.start + 2] * k + answer
                mask = [False] * (len(prompt) + k) + [True] * len(answer)
                loss = ce_masked(model, ids, mask)
            elif arm == "murmur":
                model.eval()
                blk = generate_murmur_block(model, prompt, band, max_len=k, gen=gen)
                base = answer_logprob(model, prompt, answer)
                up = answer_logprob(model, prompt + blk, answer) - base
                model.train()
                if up > 0:
                    n_acc += 1
                    blocks.append(blk)
                    ids = prompt + blk + answer
                    mpos = murmur_positions(ids, band).tolist()
                    mask = [m or t >= len(prompt) + len(blk) for t, m in enumerate(mpos)]
                    mask[len(prompt)] = False  # BEGIN is inserted, not sampled
                    loss = ce_masked(model, ids, mask)
                else:
                    loss = ce_on_positions(model, prompt + answer, len(prompt))
            opt.zero_grad()
            loss.backward()
            opt.step()
        log.append(
            {
                "round": r,
                "accept_rate": n_acc / per_round,
                "band_entropy": band_usage_entropy(blocks, band),
            }
        )
    return log


def run_experiment(
    seeds=3,
    n_digits=5,
    k=6,
    train_n=192,
    test_n=96,
    pretrain_steps=350,
    rounds=6,
    per_round=48,
    lr=3e-4,
    out_path="runs/murmur_micro.json",
):
    from charkha import Charkha, CharkhaConfig

    results = {
        "config": dict(
            seeds=seeds,
            n_digits=n_digits,
            k=k,
            train_n=train_n,
            test_n=test_n,
            pretrain_steps=pretrain_steps,
            rounds=rounds,
            per_round=per_round,
            lr=lr,
        ),
        "seeds": [],
    }
    for seed in range(seeds):
        t0 = time.time()
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        train = make_split(train_n, n_digits, gen)
        test = make_split(test_n, n_digits, gen)
        cfg = CharkhaConfig.toy()
        base = Charkha(cfg)
        base, cfg2, band = extend_vocab(base, cfg, band_codes=30, seed=seed)
        pretrain(base, band, train, pretrain_steps, lr, gen)
        pre_acc = eval_arm(base, test, band, "none", k, gen)
        row = {"seed": seed, "pretrain_acc": pre_acc, "arms": {}}
        for arm in ("none", "pause", "murmur"):
            m = copy.deepcopy(base)
            agen = torch.Generator().manual_seed(1000 + seed)
            log = finetune_arm(m, band, train, arm, rounds, per_round, k, lr, agen)
            acc = eval_arm(m, test, band, arm, k, agen)
            row["arms"][arm] = {"acc": acc, "log": log}
            print(
                f"[seed {seed}] {arm:6s} acc={acc:.3f}"
                + (
                    f" accept_last={log[-1]['accept_rate']:.2f}"
                    f" entropy_last={log[-1]['band_entropy']:.2f}"
                    if arm == "murmur"
                    else ""
                )
            )
        row["minutes"] = (time.time() - t0) / 60
        results["seeds"].append(row)
    import os

    os.makedirs(out_path.rsplit("/", 1)[0], exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    means = {
        a: sum(r["arms"][a]["acc"] for r in results["seeds"]) / seeds
        for a in ("none", "pause", "murmur")
    }
    print(
        f"\n[micro-E5] mean acc over {seeds} seeds: "
        f"none={means['none']:.3f} pause={means['pause']:.3f} "
        f"murmur={means['murmur']:.3f}  -> {out_path}"
    )
    return results


# ---------------------------------------------------------------------------
# Selftest (structural, fast — the experiment itself is --run)
# ---------------------------------------------------------------------------


def _selftest():
    from charkha import Charkha, CharkhaConfig

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    checks = 0

    prompt, answer = make_pair(5, gen)
    assert prompt[-1] == SEP and answer[-1] == EOS
    assert answer[:-1] == sorted(prompt[:-1])
    checks += 1

    cfg = CharkhaConfig.toy()
    model = Charkha(cfg)
    model, cfg2, band = extend_vocab(model, cfg, band_codes=10)

    # loss primitives differentiate and hit only the masked positions
    model.eval()  # deterministic recurrence so the two losses compare exactly
    ids = prompt + answer
    l1 = ce_on_positions(model, ids, len(prompt))
    l1.backward()
    assert l1.item() > 0
    mask = [False] * len(prompt) + [True] * len(answer)
    l2 = ce_masked(model, ids, mask)
    assert abs(l1.item() - l2.item()) < 1e-5  # same positions, same loss
    checks += 2

    # greedy answer respects the visible mask (no band ids ever)
    model.eval()
    out = greedy_answer(model, prompt, band, 4)
    assert len(out) == 4 and not any(band.contains(torch.tensor(t)).item() for t in out)
    checks += 1

    # one micro fine-tune round per arm runs end-to-end on a tiny budget
    split = make_split(6, 5, gen)
    for arm in ("none", "pause", "murmur"):
        m = copy.deepcopy(model)
        log = finetune_arm(
            m,
            band,
            split,
            arm,
            rounds=1,
            per_round=3,
            k=3,
            lr=1e-3,
            gen=torch.Generator().manual_seed(1),
        )
        assert len(log) == 1 and 0.0 <= log[0]["accept_rate"] <= 1.0
        acc = eval_arm(m, split[:3], band, arm, k=3, gen=torch.Generator().manual_seed(2))
        assert 0.0 <= acc <= 1.0
    checks += 3

    print(f"[selftest] murmur_micro.py: all checks passed ({checks} groups)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.run:
        run_experiment(seeds=a.seeds)
        sys.exit(0)
    ap.print_help()
