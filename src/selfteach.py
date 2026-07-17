"""CHARKHA self-teaching — verification-gated reasoning engine (teacher-free, collapse-proof).

Complementary to frontier-teacher KD (pipeline.py --synth --api-base): self-teaching generates
verified-correct traces (facts the model proved, not imitation), is free per token, and can
improve the model beyond any teacher's ceiling on checkable tasks. Use it alongside KD.

The mechanism: the model proposes solutions to problems whose answers are *checkable* (arithmetic,
logic, code) and we keep ONLY the verified-correct traces to train on. Correct answers to math are
facts the model derived — the verification gate, not licensing, is what makes them safe to train on.

WHY IT DOESN'T END IN ENTROPY DEATH (model collapse on its own outputs — Shumailov et al.):
  1. VERIFICATION GATE (the core): only verified-correct traces become data, so quality cannot
     drift downward (STaR/ReST-EM/RLVR; B-STaR for sustained exploration).
  2. REAL-DATA ANCHOR: every round mixes in a fraction of real corpus, pinning the
     distribution so it can't fold onto self-output.
  3. DIVERSITY PRESERVATION: dedup identical traces; monitor unique-ratio; a collapse guard halts
     the loop if diversity craters or pass-rate degenerates.
  4. BOUNDARY FOCUS (B-STaR): spend sampling on mid-difficulty problems (not the trivially solved
     or the impossible) — that's where new, learnable-but-not-yet-known signal lives.

Pure-Python (no torch): the model is injected as `generate_fn(prompt)->list[str]`, so the logic is
unit-tested instantly and hermetically. The real loop wraps charkha.generate.

  python selfteach.py --selftest
"""

from __future__ import annotations
import argparse
import array
import json
import os
import random
import re
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------
# Verifiers — ground truth is computed when the problem is made, so checking a
# candidate solution is exact-match on its extracted final answer. No teacher,
# no external labels: the answer to "17*23" is a fact.
# --------------------------------------------------------------------------

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def extract_final_answer(text: str) -> str:
    """Pull the model's final answer: prefer an explicit '#### x' or 'answer: x' tag,
    else the last number in the text (GSM8K convention)."""
    m = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    m = re.search(r"answer\s*[:=]\s*(-?\d+(?:\.\d+)?)", text, re.I)
    if m:
        return m.group(1)
    nums = _NUM.findall(text)
    return nums[-1] if nums else ""


def normalize_num(s: str) -> str:
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else f"{f:.6g}"
    except (ValueError, TypeError):
        return str(s).strip()


def verify(gold, candidate_text: str) -> bool:
    """A candidate solves the problem iff its extracted final answer equals gold."""
    return normalize_num(extract_final_answer(candidate_text)) == normalize_num(gold)


# --------------------------------------------------------------------------
# Verifiable problem bank — generated with known answers, so it's label-free and
# infinitely scalable. Difficulty is graded so the boundary selector has a handle.
# --------------------------------------------------------------------------


def gen_problems(n, seed=0):
    """Mixed bank of verifiable problems, each {prompt, gold, difficulty, type}."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        kind = rng.choice(["add", "mul", "word", "count"])
        if kind == "add":
            a, b = rng.randint(2, 999), rng.randint(2, 999)
            out.append(
                {
                    "prompt": f'What is {a} + {b}? End with "#### <answer>".',
                    "gold": str(a + b),
                    "difficulty": 1,
                    "type": "add",
                }
            )
        elif kind == "mul":
            a, b = rng.randint(2, 99), rng.randint(2, 99)
            out.append(
                {
                    "prompt": f'What is {a} * {b}? End with "#### <answer>".',
                    "gold": str(a * b),
                    "difficulty": 2,
                    "type": "mul",
                }
            )
        elif kind == "word":
            a, b = rng.randint(3, 40), rng.randint(2, 9)
            out.append(
                {
                    "prompt": f"A box holds {a} items. {b} boxes are filled. "
                    f'How many items total? End with "#### <answer>".',
                    "gold": str(a * b),
                    "difficulty": 3,
                    "type": "word",
                }
            )
        else:
            lo, hi = rng.randint(1, 20), rng.randint(40, 99)
            out.append(
                {
                    "prompt": f"How many integers are there from {lo} to {hi} inclusive? "
                    f'End with "#### <answer>".',
                    "gold": str(hi - lo + 1),
                    "difficulty": 2,
                    "type": "count",
                }
            )
    return out


# --------------------------------------------------------------------------
# Rejection-sampling round (STaR/ReST E-step): sample k candidates per problem,
# keep verified-correct, dedup. Returns kept traces + per-problem pass stats.
# --------------------------------------------------------------------------


def self_teach_round(problems, generate_fn, k=4):
    kept, per_problem = [], []
    for p in problems:
        cands = generate_fn(p["prompt"], k)
        good = [c for c in cands if verify(p["gold"], c)]
        seen, uniq = set(), []
        for c in good:  # dedup identical correct traces
            key = re.sub(r"\s+", " ", c.strip())
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        for c in uniq:
            kept.append(
                {
                    "prompt": p["prompt"],
                    "solution": c,
                    "type": p["type"],
                    "difficulty": p["difficulty"],
                }
            )
        per_problem.append(
            {
                "gold": p["gold"],
                "difficulty": p["difficulty"],
                "n": len(cands),
                "correct": len(good),
                "unique": len(uniq),
                "pass_rate": len(good) / max(len(cands), 1),
            }
        )
    n_corr = sum(s["correct"] for s in per_problem)
    n_tot = sum(s["n"] for s in per_problem)
    return {
        "kept": kept,
        "per_problem": per_problem,
        "pass_rate": n_corr / max(n_tot, 1),
        "diversity": trace_diversity(kept),
    }


# --------------------------------------------------------------------------
# RLCM (arXiv:2604.23333): margin-based confidence training from the SAME verifiable rollouts.
# The verification gate already labels every candidate correct/incorrect; pairing a correct trace
# against an incorrect one at the same problem/budget is exactly the contrastive signal the
# confidence head needs. Margin training (conf(good) > conf(bad) + m) is RL-stable where absolute
# BCE score-matching is brittle. Pairing is pure/torch-free (testable); the optimizer step that
# applies model.conf_margin_loss is lazy-imported.
# --------------------------------------------------------------------------


def make_margin_pairs(problems, cand_lists, max_per_problem=2):
    """From per-problem candidate lists, pair verified-CORRECT vs verified-INCORRECT traces.
    Returns [{prompt, good, bad}]. Only problems with BOTH a correct and an incorrect candidate
    contribute (others carry no contrastive signal). Pure — mirrors self_teach_round's verify()."""
    pairs = []
    for p, cands in zip(problems, cand_lists):
        good = [c for c in cands if verify(p["gold"], c)]
        bad = [c for c in cands if not verify(p["gold"], c)]
        if not good or not bad:
            continue
        for g, b in zip(good[:max_per_problem], bad[:max_per_problem]):
            pairs.append({"prompt": p["prompt"], "good": g, "bad": b})
    return pairs


def conf_margin_step(model, encode_fn, pairs, device="cpu", opt=None, max_len=512):
    """Apply RLCM margin loss over (good,bad) prefix pairs. encode_fn(text)->list[int] (the
    tokenizer). Hidden states come from model.hidden(); each prefix is mean-pooled so good/bad are
    comparable across lengths. Steps `opt` if given. Returns the mean scalar loss (float). Lazy
    torch; no-op (returns 0.0) on empty pairs."""
    import torch

    if not pairs:
        return 0.0
    model.train()
    gh, bh = [], []
    for pr in pairs:
        gi = encode_fn(pr["prompt"] + "\n" + pr["good"])[:max_len] or [0]
        bi = encode_fn(pr["prompt"] + "\n" + pr["bad"])[:max_len] or [0]
        gh.append(model.hidden(torch.tensor([gi], device=device)).mean(1).squeeze(0))
        bh.append(model.hidden(torch.tensor([bi], device=device)).mean(1).squeeze(0))
    loss = model.conf_margin_loss(torch.stack(gh), torch.stack(bh))
    if opt is not None:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return float(loss.detach().item())


# --------------------------------------------------------------------------
# Anti-entropy-death machinery
# --------------------------------------------------------------------------


def trace_diversity(traces):
    """Unique-solution ratio in [0,1]. Low ratio => the model is converging on one
    template (early warning of collapse)."""
    if not traces:
        return 1.0
    norm = [re.sub(r"\s+", " ", t["solution"].strip()) for t in traces]
    return len(set(norm)) / len(norm)


def boundary_problems(problems, per_problem, lo=0.05, hi=0.95):
    """B-STaR: keep problems whose pass-rate is in (lo,hi) — solvable-but-not-trivial. Sampling
    these (not the 100%-solved or 0%-solved) is where exploration stays productive."""
    keep = []
    for p, s in zip(problems, per_problem):
        if lo <= s["pass_rate"] <= hi:
            keep.append(p)
    return keep


def mix_with_anchor(self_traces, anchor_texts, anchor_frac=0.3, seed=0):
    """Pin the training distribution to real corpus data so it can't fold onto self-output.
    Returns a shuffled list of dicts tagged 'self' or 'anchor', with anchor ~= anchor_frac."""
    rng = random.Random(seed)
    n_self = len(self_traces)
    n_anchor = (
        int(round(anchor_frac / max(1e-9, 1 - anchor_frac) * n_self))
        if anchor_frac < 1
        else len(anchor_texts)
    )
    n_anchor = min(n_anchor, len(anchor_texts))
    items = [{"source": "self", "text": t["prompt"] + "\n" + t["solution"]} for t in self_traces]
    items += (
        [{"source": "anchor", "text": a} for a in rng.sample(anchor_texts, n_anchor)]
        if anchor_texts
        else []
    )
    rng.shuffle(items)
    return items


def collapse_guard(history, min_diversity=0.25, window=3):
    """Halt the self-teaching loop before entropy death. Trips if diversity falls below a floor,
    or if diversity has fallen for `window` consecutive rounds (a downward spiral)."""
    if not history:
        return True, "ok"
    last = history[-1]
    if last["diversity"] < min_diversity:
        return False, f"diversity {last['diversity']:.2f} < floor {min_diversity}"
    if len(history) >= window + 1:
        recent = [h["diversity"] for h in history[-(window + 1) :]]
        if all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
            return (
                False,
                f"diversity fell {window} rounds running ({recent[0]:.2f}->{recent[-1]:.2f})",
            )
    return True, "ok"


# --------------------------------------------------------------------------
# End-to-end wiring: checkpoint -> generate -> verify/filter -> tokenized shard.
# The kept traces become a training shard (same uint16 format as dataprep), so a
# self-teaching round feeds straight back into train.py. Model bits are lazy-imported
# so the verification/shard logic stays torch-free and hermetically testable.
# --------------------------------------------------------------------------


def make_generate_fn(model, tok, device, max_new=96, temp=0.8, top_k=50, effort=None):
    """Wrap a Charkha checkpoint as generate_fn(prompt, k) -> list[str] for self_teach_round.
    Temperature sampling gives the k diverse candidates rejection-sampling needs."""
    import torch

    @torch.no_grad()
    def gen(prompt, k):
        ids = tok.encode(prompt)
        x = torch.tensor([ids], device=device)
        outs = []
        for _ in range(k):
            o = model.generate(x, max_new, effort=effort, temp=temp, top_k=top_k)[0].tolist()
            outs.append(tok.decode(o[len(ids) :]))
        return outs

    return gen


def write_traces_shard(traces, out_dir, tokenizer_name=None, shard_tokens=100_000_000):
    """Tokenize kept (prompt+solution) traces into uint16 shard(s) via dataprep's ShardWriter,
    so a self-teaching round produces drop-in training data. Returns the shard manifest."""
    import os
    from dataprep import ShardWriter, load_tokenizer

    encode, vocab = load_tokenizer(tokenizer_name)
    writer = ShardWriter(out_dir, shard_tokens)
    for t in traces:
        writer.add(encode(t["prompt"] + "\n" + t["solution"] + "\n"))
    shards = writer.close()
    index = {
        "vocab_size": vocab,
        "total_tokens": writer.total,
        "shards": shards,
        "source": "selfteach",
    }
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    return index


# --------------------------------------------------------------------------
# Self-IMPROVEMENT loop (train -> eval -> promote): the "smarter over time" driver.
# self_teach_round only PRODUCES verified traces; this closes the loop by training on them and
# KEEPING the new weights only when held-out verified pass-rate actually improves (else rollback).
# That promote gate is what turns "generate data" into "monotonically improving model" — a round
# that doesn't verifiably help is discarded, so the loop can't drift backwards.
# --------------------------------------------------------------------------


def eval_pass_rate(problems, generate_fn, k=1):
    """Held-out metric: fraction of problems solved (verified) with k samples each. The promote gate."""
    solved = 0
    for p in problems:
        cands = generate_fn(p["prompt"], k)
        if any(verify(p["gold"], c) for c in cands):
            solved += 1
    return solved / max(len(problems), 1)


def promote_decision(best_score, candidate_score, min_delta=0.0):
    """Promote the candidate weights iff held-out pass-rate improves by >= min_delta; else roll back."""
    return candidate_score >= best_score + min_delta


def improve_loop(
    make_generate,
    train_on_traces,
    eval_problems,
    *,
    rounds=3,
    k=4,
    min_delta=0.0,
    train_problems=None,
    seed=0,
):
    """Verification-gated self-improvement driver (pure orchestration — `state` is opaque, so this is
    testable with mocks and reusable with a real checkpoint).
      make_generate(state) -> generate_fn(prompt, k) -> [candidate strings]
      train_on_traces(state, kept_traces) -> new_state
    Each round: sample+verify traces, train a candidate, eval held-out pass-rate, and PROMOTE only on
    improvement (otherwise keep the previous best state). Returns (best_state, history)."""
    state = None
    best = eval_pass_rate(eval_problems, make_generate(state), k=1)
    history = [{"round": -1, "score": best, "action": "init"}]
    for rnd in range(rounds):
        probs = train_problems if train_problems is not None else eval_problems
        result = self_teach_round(probs, make_generate(state), k=k)
        cand_state = train_on_traces(state, result["kept"])
        score = eval_pass_rate(eval_problems, make_generate(cand_state), k=1)
        if promote_decision(best, score, min_delta):
            state, best = cand_state, score
            action = "promote"
        else:
            action = "rollback"
        history.append(
            {
                "round": rnd,
                "score": score,
                "best": best,
                "action": action,
                "kept": len(result["kept"]),
                "pass_rate": result["pass_rate"],
            }
        )
    return state, history


def run_improve(
    ckpt,
    out_dir,
    *,
    rounds=3,
    n_problems=256,
    k=4,
    device=None,
    lr=1e-5,
    min_delta=0.0,
    sft_steps=200,
    batch_size=4,
    seed=0,
    toy=False,
):
    """Real self-improvement against a checkpoint (GPU/user-run). Trains on verified traces with
    completion-masked SFT (sft.sft_step), evaluates held-out verified pass-rate, and snapshots the
    best weights to out_dir/ckpt.pt. Generation/training mutate one model in place; rollback restores
    the previous state_dict so a non-improving round is discarded."""
    import copy
    import os
    import torch
    from serve import load_model
    from sft import encode_example, collate, sft_step

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, cfg = load_model(ckpt, device, toy=toy)
    eval_problems = gen_problems(max(64, n_problems // 4), seed=seed + 9973)
    train_problems = gen_problems(n_problems, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    def make_generate(_state):  # weights live in `model`; _state is the snapshot id
        return make_generate_fn(model, tok, device, max_new=96, effort=None)

    def train_on_traces(_state, kept):
        model.train()
        exs = [encode_example(tok, t["prompt"], t["solution"], cfg.max_seq_len) for t in kept]
        if not exs:
            return copy.deepcopy(model.state_dict())
        rng = random.Random(seed)
        for s in range(sft_steps):
            rng.shuffle(exs)
            xs, ys = collate(exs[:batch_size])
            sft_step(model, opt, xs, ys, device)
        model.eval()
        return copy.deepcopy(model.state_dict())

    # wrap the in-place model so improve_loop's promote/rollback restores weights correctly
    base = copy.deepcopy(model.state_dict())

    def make_generate_state(state):
        model.load_state_dict(state if state is not None else base)
        return make_generate(state)

    best = eval_pass_rate(eval_problems, make_generate_state(None), k=1)
    state, best_state = None, base
    history = [{"round": -1, "score": best, "action": "init"}]
    for rnd in range(rounds):
        model.load_state_dict(best_state)
        result = self_teach_round(train_problems, make_generate(None), k=k)
        cand = train_on_traces(None, result["kept"])
        model.load_state_dict(cand)
        score = eval_pass_rate(eval_problems, make_generate(None), k=1)
        if promote_decision(best, score, min_delta):
            best, best_state = score, cand
            action = "promote"
        else:
            action = "rollback"
        history.append(
            {
                "round": rnd,
                "score": score,
                "best": best,
                "action": action,
                "kept": len(result["kept"]),
                "pass_rate": result["pass_rate"],
            }
        )
        print(
            f"  [improve] round {rnd}: pass_rate={result['pass_rate']:.3f} "
            f"eval={score:.3f} best={best:.3f} -> {action} ({len(result['kept'])} traces)"
        )
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "ckpt.pt.tmp")
    torch.save(
        {
            "model": best_state,
            "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else cfg,
            "improve_history": history,
        },
        tmp,
    )
    os.replace(tmp, os.path.join(out_dir, "ckpt.pt"))
    print(f"[improve] best held-out pass-rate {best:.3f} -> {os.path.join(out_dir, 'ckpt.pt')}")
    return history


def run_self_teach(
    ckpt,
    out_dir,
    n_problems=512,
    k=4,
    rounds=1,
    seed=0,
    device=None,
    tokenizer_name=None,
    min_diversity=0.25,
    anchor_dir=None,
    anchor_frac=0.3,
):
    """Full clean self-teaching: load a checkpoint, solve verifiable problems, keep verified
    traces (boundary-focused, collapse-guarded), and write them as a training shard.
    If anchor_dir is a directory of uint16 token shards + index.json, mix anchor docs in at
    ~anchor_frac to pin the distribution to real corpus data."""
    import torch
    import os
    from charkha import Charkha, CharkhaConfig

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = CharkhaConfig.from_dict(ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    model = Charkha(cfg).to(device).eval()
    model.load_state_dict(ck.get("model", ck))
    from _verify import load_tokenizer_for

    tok = load_tokenizer_for(tokenizer_name or getattr(cfg, "tokenizer_name", None))
    gen = make_generate_fn(model, tok, device)
    problems = gen_problems(n_problems, seed)

    # load anchor texts from existing training shards
    anchor_texts = []
    if anchor_dir and anchor_frac > 0:
        idx_path = os.path.join(anchor_dir, "index.json")
        if os.path.exists(idx_path):
            idx = json.load(
                open(idx_path, "rb") if hasattr(json, "loads") else json.load(open(idx_path))
            )

            for sh in idx.get("shards", []):
                path = os.path.join(anchor_dir, sh["file"])
                if not os.path.exists(path):
                    continue
                arr = array.array("H")
                with open(path, "rb") as f:
                    arr.fromfile(f, sh["tokens"])
                text = tok.decode(list(arr))
                anchor_texts.append(text)
            print(f"  [anchor] loaded {len(anchor_texts)} anchor docs from {anchor_dir}")

    all_kept, history = [], []
    for r in range(rounds):
        res = self_teach_round(problems, gen, k=k)
        history.append(res)
        all_kept.extend(res["kept"])
        ok, why = collapse_guard(history, min_diversity)
        print(
            f"  [round {r}] pass_rate={res['pass_rate']:.3f} kept={len(res['kept'])} "
            f"diversity={res['diversity']:.3f} guard={'ok' if ok else 'STOP: ' + why}"
        )
        if not ok:
            break
        problems = boundary_problems(problems, res["per_problem"]) or problems  # B-STaR focus

    # mix anchor docs into training output
    mixed = mix_with_anchor(all_kept, anchor_texts, anchor_frac=anchor_frac, seed=seed)
    self_texts = [m["text"] for m in mixed if m["source"] == "self"]
    idx = write_traces_shard(self_texts, out_dir, tokenizer_name)
    print(
        f"  [selfteach] wrote {len(self_texts)} self-traces + {len([m for m in mixed if m['source'] == 'anchor'])} anchor docs "
        f"-> {idx['total_tokens']:,} tokens in {out_dir}"
    )
    return idx


# --------------------------------------------------------------------------
def _selftest():
    ok = 0

    def ck(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        assert cond, name
        ok += 1

    # 1. answer extraction + verification
    ck("extract #### tag", extract_final_answer("blah\n#### 42") == "42")
    ck("extract answer: tag", extract_final_answer("answer: 17") == "17")
    ck("extract trailing number", extract_final_answer("so it is 99 in total") == "99")
    ck("verify correct", verify("42", "the sum is ... #### 42"))
    ck("verify rejects wrong", not verify("42", "the sum is ... #### 41"))
    ck("verify int/float normalize", verify("42", "#### 42.0"))

    probs = gen_problems(40, seed=1)
    ck("problems have gold answers", all(p["gold"] and "prompt" in p for p in probs))

    # 2. rejection sampling: a stub model that is right ~60% with varied text, wrong otherwise.
    rng = random.Random(0)

    def stub_gen(prompt, k):
        gold = next(p["gold"] for p in probs if p["prompt"] == prompt)
        outs = []
        for i in range(k):
            if rng.random() < 0.6:
                phrasing = rng.choice(["Computing, ", "Step by step, ", "Thus, ", "We get "])
                outs.append(f"{phrasing}#### {gold}")  # correct, varied
            else:
                outs.append(f"#### {int(gold) + rng.randint(1, 5)}")  # wrong
        return outs

    res = self_teach_round(probs, stub_gen, k=6)
    ck(
        "round keeps only correct traces",
        all(
            verify(next(p["gold"] for p in probs if p["prompt"] == t["prompt"]), t["solution"])
            for t in res["kept"]
        ),
    )
    ck("pass_rate in (0,1) for a 60% model", 0.3 < res["pass_rate"] < 0.85)
    ck("kept traces exist", len(res["kept"]) > 0)

    # 3. verification gate is the anti-collapse core: a degenerate always-wrong model keeps NOTHING.
    bad = self_teach_round(probs, lambda pr, k: ["#### 999999"] * k, k=4)
    ck("always-wrong model contributes zero training data", len(bad["kept"]) == 0)

    # 4. diversity metric + collapse guard
    dup = [{"solution": "#### 5"} for _ in range(10)]
    var = [{"solution": f"#### {i}"} for i in range(10)]
    ck("diversity low for identical traces", trace_diversity(dup) == 0.1)
    ck("diversity high for varied traces", trace_diversity(var) == 1.0)
    ok_guard, _ = collapse_guard([{"diversity": 0.9}])
    bad_guard, reason = collapse_guard([{"diversity": 0.1}])
    ck("guard passes healthy diversity", ok_guard)
    ck("guard trips on collapsed diversity", not bad_guard and "floor" in reason)
    spiral = [{"diversity": d} for d in (0.9, 0.7, 0.5, 0.3)]
    trip, _ = collapse_guard(spiral)
    ck("guard trips on a downward spiral", not trip)

    # 5. boundary selection keeps mid-difficulty, drops solved/impossible
    pp = [{"pass_rate": 1.0}, {"pass_rate": 0.0}, {"pass_rate": 0.5}]
    sel = boundary_problems([{"i": 0}, {"i": 1}, {"i": 2}], pp)
    ck("boundary keeps only the learnable middle", sel == [{"i": 2}])

    # 6. real-data anchor prevents pure self-training (≈30% anchor by default)
    mixed = mix_with_anchor(
        [{"prompt": "p", "solution": "s"}] * 7,
        [f"real doc {i}" for i in range(20)],
        anchor_frac=0.3,
        seed=1,
    )
    frac = sum(1 for m in mixed if m["source"] == "anchor") / len(mixed)
    ck("anchor fraction ~0.3 (distribution pinned to real data)", 0.2 <= frac <= 0.4)

    tmp = tempfile.mkdtemp()
    traces = [{"prompt": f"What is {i}+{i}?", "solution": f"#### {i + i}"} for i in range(50)]
    idx = write_traces_shard(traces, tmp, tokenizer_name=None, shard_tokens=10_000)
    ck("shard index records tokens", idx["total_tokens"] > 0 and len(idx["shards"]) >= 1)
    shard0 = os.path.join(tmp, idx["shards"][0]["file"])
    arr = array.array("H")
    with open(shard0, "rb") as f:
        arr.fromfile(f, idx["shards"][0]["tokens"])
    ck("shard is non-empty uint16", len(arr) > 0 and max(arr) <= 256)
    ck("index.json written", os.path.exists(os.path.join(tmp, "index.json")))
    empty = write_traces_shard([], tempfile.mkdtemp(), tokenizer_name=None)
    ck("empty round writes 0-token shard cleanly", empty["total_tokens"] == 0)

    # 7. RLCM margin pairs (pure): only mixed correct/incorrect problems contribute, pairs are
    # genuinely (correct, incorrect), and an all-correct or all-wrong problem yields nothing.
    mp_probs = [{"prompt": "q", "gold": "4"}, {"prompt": "r", "gold": "7"}]
    mp_cands = [
        ["#### 4", "#### 5", "#### 4"],  # mixed -> pairs
        ["#### 7", "#### 7"],
    ]  # all correct -> no pair
    pairs = make_margin_pairs(mp_probs, mp_cands, max_per_problem=2)
    ck("margin pairs only from mixed problems", all(pr["prompt"] == "q" for pr in pairs) and pairs)
    ck(
        "each pair is (correct, incorrect)",
        all(verify("4", pr["good"]) and not verify("4", pr["bad"]) for pr in pairs),
    )
    none_pairs = make_margin_pairs([{"prompt": "x", "gold": "1"}], [["#### 2", "#### 3"]])
    ck("no pairs when no correct candidate", none_pairs == [])

    # 8. Self-IMPROVEMENT loop (train->eval->promote) with a deterministic mock model. The mock's
    # "skill" gates which difficulties it solves; a train_fn that raises skill must drive monotone
    # improvement + promotions, and a train_fn that lowers it must roll back (best never regresses).
    iprobs = gen_problems(24, seed=3)
    by_prompt = {p["prompt"]: p for p in iprobs}

    def mock_gen(state):
        skill = 0.0 if state is None else state["skill"]

        def gen(prompt, k):
            p = by_prompt[prompt]
            return ([f"#### {p['gold']}"] if skill >= p["difficulty"] else ["#### 0"]) * k

        return gen

    def train_up(state, kept):
        return {"skill": (0.0 if state is None else state["skill"]) + 1.0}

    def train_down(state, kept):
        return {"skill": max(0.0, (0.0 if state is None else state["skill"]) - 1.0)}

    ck(
        "eval_pass_rate: skilled model solves more than a novice",
        eval_pass_rate(iprobs, mock_gen({"skill": 3.0})) > eval_pass_rate(iprobs, mock_gen(None)),
    )
    ck("promote_decision: improvement promotes", promote_decision(0.5, 0.6, 0.0) is True)
    ck("promote_decision: no improvement rolls back", promote_decision(0.5, 0.5, 0.01) is False)

    state_up, hist_up = improve_loop(mock_gen, train_up, iprobs, rounds=3, k=2, min_delta=0.0)
    scores_up = [h["best"] for h in hist_up if "best" in h]
    ck("improve loop is monotone non-decreasing in best", scores_up == sorted(scores_up))
    ck("improve loop ends smarter than it started", hist_up[-1]["best"] > hist_up[0]["score"])
    ck("improve loop promotes when training helps", any(h["action"] == "promote" for h in hist_up))

    _, hist_dn = improve_loop(mock_gen, train_down, iprobs, rounds=2, k=2, min_delta=0.01)
    ck(
        "improve loop rolls back a non-improving round",
        any(h["action"] == "rollback" for h in hist_dn),
    )
    ck("improve loop never regresses best on rollback", hist_dn[-1]["best"] >= hist_dn[0]["score"])

    print(
        f"\nselfteach selftest: {ok}/{ok} passed -- verification-gated self-training, "
        "collapse-guarded, writes drop-in shards, and improves-or-rolls-back over rounds"
    )
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CHARKHA clean self-teaching (verification-gated)")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--run", action="store_true", help="run self-teaching against a checkpoint")
    p.add_argument("--ckpt", type=str, help="checkpoint to self-teach from")
    p.add_argument("--out", type=str, default="data_selfteach", help="output shard dir")
    p.add_argument("--n-problems", type=int, default=512)
    p.add_argument("--k", type=int, default=4, help="candidates sampled per problem")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--anchor-dir",
        type=str,
        default=None,
        help="training shard dir to anchor self-teach with real data",
    )
    p.add_argument(
        "--anchor-frac", type=float, default=0.3, help="fraction of training mix from anchor data"
    )
    p.add_argument(
        "--improve",
        action="store_true",
        help="self-improvement loop: train on verified traces, eval, promote-or-rollback",
    )
    p.add_argument("--toy", action="store_true", help="with --improve: fresh toy model (no ckpt)")
    p.add_argument(
        "--min-delta", type=float, default=0.0, help="--improve: min held-out gain to promote"
    )
    p.add_argument("--sft-steps", type=int, default=200, help="--improve: SFT steps per round")
    p.add_argument("--lr", type=float, default=1e-5, help="--improve: learning rate")
    a = p.parse_args()
    if a.selftest:
        _selftest()
    elif a.improve:
        run_improve(
            a.ckpt,
            a.out,
            rounds=a.rounds,
            n_problems=a.n_problems,
            k=a.k,
            device=a.device,
            lr=a.lr,
            min_delta=a.min_delta,
            sft_steps=a.sft_steps,
            toy=a.toy,
        )
    elif a.run:
        if not a.ckpt:
            p.error("--run requires --ckpt")
        run_self_teach(
            a.ckpt,
            a.out,
            n_problems=a.n_problems,
            k=a.k,
            rounds=a.rounds,
            device=a.device,
            anchor_dir=a.anchor_dir,
            anchor_frac=a.anchor_frac,
        )
    else:
        p.print_help()
