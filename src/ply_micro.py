#!/usr/bin/env python3
"""Micro-E9: CPU-only capability probe for Ply latent trajectory search.

E9 asks whether Ply is more than a serving trick. The registered question is not
"can we spend more FLOPs?" but whether branching the recurrent core finds useful
latent alternatives that a deeper single trajectory or token-space search does not.
This file supplies the first local harness for that question on a checkable toy task.

Task: sort a short digit string. All arms share one tiny pretrained Charkha.
  plain       greedy decode at fixed r.
  deep        same interface, r multiplied by n_branches (single-trajectory FLOPs).
  ply_conf    N latent branches selected by the confidence head.
  ply_margin  N latent branches selected by top1-top2 margin.
  ply_target  oracle-only ceiling: select the branch that gives the true next token
              the highest log-probability, then still emit that branch's argmax.

Interpretation:
  * If ply_target does not beat plain/deep, the branch space did not contain useful
    alternatives at this scale.
  * If ply_target wins but ply_conf/ply_margin do not, the search space exists but
    CHARKHA lacks a selector.
  * If ply_margin or ply_conf beats deep at matched core-loop budget, Ply has its
    first micro-scale capability evidence. That would still not prove the 0.42B
    claim; it would only justify running E9 at real scale.

Usage:
    python src/ply_micro.py --selftest
    python src/ply_micro.py --run --seeds 3

"""

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.replace(chr(92), "/").rsplit("/", 1)[0])
from ply import ply_forward

from micro_task import DIGITS, SEP, EOS, make_pair, make_split  # noqa: F401

MODES = ("plain", "deep", "ply_conf", "ply_margin", "ply_target")


def ce_on_positions(model, ids, train_from, r):
    """CE over predicting ids[train_from:] from their prefixes."""
    idx = torch.tensor([ids], dtype=torch.long)
    logits, _ = model(idx, r=r)
    lp = logits[0, train_from - 1 : len(ids) - 1]
    tgt = torch.tensor(ids[train_from:], dtype=torch.long)
    return F.cross_entropy(lp.float(), tgt)


def pretrain(model, split, steps, lr, r, gen):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        prompt, answer = split[int(torch.randint(0, len(split), (1,), generator=gen))]
        loss = ce_on_positions(model, prompt + answer, len(prompt), r)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def decode_answer(model, prompt, answer, mode, r, n_branches, noise, gen):
    model.eval()
    idx = torch.tensor([list(prompt)], dtype=torch.long)
    out, chosen = [], []
    for tpos in range(len(answer)):
        if mode == "plain":
            logits, _ = model(idx, r=r)
        elif mode == "deep":
            logits, _ = model(idx, r=r * n_branches)
        elif mode == "ply_conf":
            logits, _, info = ply_forward(
                model, idx, r=r, n_branches=n_branches, noise=noise, score="conf", gen=gen
            )
            chosen.append(info["chosen"][0])
        elif mode == "ply_margin":
            logits, _, info = ply_forward(
                model, idx, r=r, n_branches=n_branches, noise=noise, score="margin", gen=gen
            )
            chosen.append(info["chosen"][0])
        elif mode == "ply_target":
            logits, _, info = ply_forward(
                model,
                idx,
                r=r,
                n_branches=n_branches,
                noise=noise,
                score="target",
                target_next=answer[tpos],
                gen=gen,
            )
            chosen.append(info["chosen"][0])
        else:
            raise ValueError(f"unknown mode: {mode}")
        tok = int(logits[0, -1].argmax())
        out.append(tok)
        idx = torch.cat([idx, torch.tensor([[tok]], dtype=torch.long)], dim=1)
    return out, {"chosen": chosen}


@torch.no_grad()
def eval_mode(model, split, mode, r, n_branches, noise, gen):
    hits = 0
    branch_hist = []
    for prompt, answer in split:
        out, info = decode_answer(model, prompt, answer, mode, r, n_branches, noise, gen)
        hits += int(out == answer)
        branch_hist.extend(info["chosen"])
    acc = hits / max(len(split), 1)
    return {"acc": acc, "chosen": branch_hist}


def run_experiment(
    seeds=3,
    n_digits=5,
    train_n=192,
    test_n=96,
    pretrain_steps=350,
    lr=3e-4,
    r=2,
    n_branches=4,
    noise=0.05,
    out_path="runs/ply_micro.json",
):
    from charkha import Charkha, CharkhaConfig

    results = {
        "config": dict(
            seeds=seeds,
            n_digits=n_digits,
            train_n=train_n,
            test_n=test_n,
            pretrain_steps=pretrain_steps,
            lr=lr,
            r=r,
            n_branches=n_branches,
            noise=noise,
        ),
        "seeds": [],
    }
    for seed in range(seeds):
        t0 = time.time()
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        train = make_split(train_n, n_digits, gen)
        test = make_split(test_n, n_digits, gen)
        base = Charkha(CharkhaConfig.toy())
        pretrain(base, train, pretrain_steps, lr, r, gen)
        # clear non-leaf tensors stashed by the last training forward (deep-sup
        # trajectory, value states, convergence signal) - deepcopy rejects them
        for attr in ("_core_traj", "_value_states", "_last_convergence", "_last_sngp_var"):
            if hasattr(base, attr):
                setattr(base, attr, None)
        row = {"seed": seed, "modes": {}}
        for mode in MODES:
            m = copy.deepcopy(base).eval()
            egen = torch.Generator().manual_seed(1000 + seed)
            res = eval_mode(m, test, mode, r, n_branches, noise, egen)
            row["modes"][mode] = res
            print(f"[seed {seed}] {mode:10s} acc={res['acc']:.3f}")
        row["minutes"] = (time.time() - t0) / 60
        results["seeds"].append(row)

    parent = out_path.rsplit("/", 1)[0] if "/" in out_path else ""
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    means = {
        mode: sum(rw["modes"][mode]["acc"] for rw in results["seeds"]) / seeds for mode in MODES
    }
    print(
        "\n[micro-E9] mean exact-match over %d seeds: %s -> %s"
        % (seeds, " ".join(f"{m}={means[m]:.3f}" for m in MODES), out_path)
    )
    return results


# ---------------------------------------------------------------------------
# Selftest
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

    model = Charkha(CharkhaConfig.toy())
    model.train()
    loss = ce_on_positions(model, prompt + answer, len(prompt), r=2)
    loss.backward()
    assert loss.item() > 0
    checks += 1

    split = make_split(3, 4, gen)
    model.eval()
    for mode in MODES:
        res = eval_mode(
            model, split, mode, r=1, n_branches=2, noise=0.02, gen=torch.Generator().manual_seed(3)
        )
        assert 0.0 <= res["acc"] <= 1.0
        if mode.startswith("ply_"):
            assert len(res["chosen"]) == len(split[0][1]) * len(split)
    checks += len(MODES)

    print(f"[selftest] ply_micro.py: all checks passed ({checks} groups)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="runs/ply_micro.json")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.run:
        run_experiment(seeds=a.seeds, out_path=a.out)
        sys.exit(0)
    ap.print_help()
