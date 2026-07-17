"""
CHARKHA pipeline - eval harness + synthetic-data generator.
Runs evals without pausing training and generates synthetic reasoning data.
Teacher choice is unrestricted — a local HF teacher or a frontier teacher via an
OpenAI-compatible API; frontier-KD output is personal / non-distributable.

Usage:
  python pipeline.py --eval --ckpt out/ckpt.pt --tasks hellaswag,arc_easy
  python pipeline.py --synth --teacher <model-id-or-path> --tokens 100000
  python pipeline.py --synth --api-base http://localhost:4000/v1 --api-model deepseek-v4 --tokens 1000000  # frontier
"""

from __future__ import annotations
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

# Windows consoles default to cp1252 and crash when printing model bytes / unicode.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# --------------------------------------------------------------------------
# Selective prediction / calibrated abstention — the "honest tiny model".
#
# CHARKHA's conf_head emits per-token P(top-1 correct) at inference. These pure
# functions turn that signal into the fundable artifact: a risk-coverage curve
# (how accuracy rises as you abstain on the least-confident inputs) and a
# split-conformal threshold (a finite-sample upper bound on error among the
# inputs you DO answer). All stdlib+math so the math is unit-tested on CPU;
# the model/data collector below feeds it on real runs.
# --------------------------------------------------------------------------


def risk_coverage_curve(conf, correct):
    """Sort by confidence desc; for each prefix of the k most-confident items,
    report (coverage=k/n, risk=error-rate among those k, accuracy=1-risk).
    Returns a list of dicts ordered by increasing coverage."""
    n = len(conf)
    assert n == len(correct) and n > 0
    order = sorted(range(n), key=lambda i: conf[i], reverse=True)
    pts, errs = [], 0
    for k, i in enumerate(order, start=1):
        errs += 0 if correct[i] else 1
        pts.append(
            {"coverage": k / n, "risk": errs / k, "accuracy": 1 - errs / k, "threshold": conf[i]}
        )
    return pts


def auarc(conf, correct):
    """Area Under the Accuracy-Coverage curve (higher is better; the headline
    number). A model whose confidence is uninformative scores ~base accuracy;
    a well-ranked one scores higher because it answers its sure things first."""
    pts = risk_coverage_curve(conf, correct)
    # trapezoid over coverage in [1/n .. 1]; prepend (0, first-accuracy) anchor.
    xs = [0.0] + [p["coverage"] for p in pts]
    ys = [pts[0]["accuracy"]] + [p["accuracy"] for p in pts]
    return sum((xs[i] - xs[i - 1]) * (ys[i] + ys[i - 1]) / 2 for i in range(1, len(xs)))


def selective_threshold(cal_conf, cal_correct, target_risk, delta=0.05):
    """Split-conformal selective prediction. Find the LOWEST confidence threshold
    (=> highest coverage) such that a (1-delta) Hoeffding upper bound on the error
    rate among accepted (conf >= tau) calibration items stays <= target_risk.
    Returns (tau, achieved_coverage, empirical_risk, upper_bound) or
    (None, 0, 0, 0) if no threshold satisfies the guarantee.

    Hoeffding: with prob >= 1-delta, true_risk <= emp_risk + sqrt(ln(1/delta)/(2m))
    over the m accepted items. Conservative but assumption-free and honest."""
    n = len(cal_conf)
    assert n == len(cal_correct) and n > 0
    order = sorted(range(n), key=lambda i: cal_conf[i], reverse=True)
    slack_num = math.log(1.0 / delta)
    best = (None, 0.0, 0.0, 0.0)
    errs = 0
    for k, i in enumerate(order, start=1):
        errs += 0 if cal_correct[i] else 1
        emp = errs / k
        ub = emp + math.sqrt(slack_num / (2 * k))
        if ub <= target_risk:  # guarantee holds at this coverage
            best = (cal_conf[i], k / n, emp, ub)  # keep extending => higher coverage
    return best


def _selftest():
    """Unit-test the selective-prediction math on synthetic oracles — no GPU/data."""
    checks = []

    def ck(name, cond):
        checks.append((name, bool(cond)))
        print(f"  [{'ok' if cond else 'FAIL'}] {name}")

    # 1. Perfect confidence ranking: all errors are the least-confident items.
    #    Accuracy@low-coverage must be 1.0; AUARC must beat a shuffled baseline.
    conf = [0.99, 0.95, 0.90, 0.80, 0.20, 0.10]
    correct = [1, 1, 1, 1, 0, 0]  # the 2 wrong are least confident
    pts = risk_coverage_curve(conf, correct)
    ck(
        "perfect ranking: accuracy@coverage<=0.67 is 1.0",
        all(p["accuracy"] == 1.0 for p in pts if p["coverage"] <= 0.67 + 1e-9),
    )
    ck("full-coverage risk = base error rate (2/6)", abs(pts[-1]["risk"] - 2 / 6) < 1e-9)
    base = sum(correct) / len(correct)  # 4/6; flat confidence => AUARC == base acc
    a_good = auarc(conf, correct)
    ck("AUARC: informative ranking > base accuracy", a_good > base)
    ck("AUARC in (0,1]", 0 < a_good <= 1.0)

    # 2. Worst-case (anti-correlated) ranking scores below base accuracy.
    a_anti = auarc([0.1, 0.2, 0.3, 0.4, 0.95, 0.99], correct)
    ck("AUARC: anti-correlated < base accuracy", a_anti < base)

    # 3. Split-conformal: a cleanly separable calibration set (top by confidence
    #    are correct) must yield a threshold whose Hoeffding bound respects 10% risk.
    n_cal = 1000
    cc = [i / n_cal for i in range(n_cal)]  # confidence = rank
    cal_correct = [1 if c >= 0.3 else 0 for c in cc]  # top 70% correct, separable
    tau, cov, emp, ub = selective_threshold(cc, cal_correct, target_risk=0.10, delta=0.05)
    ck("conformal: returns a threshold for reachable target", tau is not None)
    ck("conformal: upper bound respects target", ub <= 0.10 + 1e-9)
    ck("conformal: positive coverage", cov > 0)
    # 4. Impossible target (lower than any achievable bound) => no threshold.
    tau2, *_ = selective_threshold(cc, cal_correct, target_risk=0.0, delta=0.05)
    ck("conformal: impossible target -> None", tau2 is None)
    # 5. Monotonicity: looser risk target never reduces coverage.
    _, cov_loose, *_ = selective_threshold(cc, cal_correct, 0.20, 0.05)
    ck("conformal: looser target -> >= coverage", cov_loose >= cov)

    # 6. E1 elasticity harness runs on a toy model + synthetic tokens (no GPU/ckpt): one row per
    #    effort, finite accuracy in [0,1] and bits>0, and rel_flops strictly increasing with effort.
    import numpy as np
    from charkha import Charkha, CharkhaConfig

    tm = Charkha(CharkhaConfig.toy())
    rng = np.random.default_rng(0)
    toks = rng.integers(0, CharkhaConfig.toy().vocab_size, size=4096).astype(np.uint16)
    rows = elasticity_curve(tm, toks, device="cpu", max_tokens=1500, seq_len=128, efforts=(1, 2, 4))
    ck("elasticity: one row per effort", len(rows) == 3)
    ck(
        "elasticity: accuracy in [0,1] and bits>0",
        all(0.0 <= r["accuracy"] <= 1.0 and r["bits"] > 0 for r in rows),
    )
    ck(
        "elasticity: rel_flops strictly increases with effort",
        rows[0]["rel_flops"] < rows[1]["rel_flops"] < rows[2]["rel_flops"],
    )
    ck("elasticity: tokens scored > 0", rows[0]["tokens"] > 0)

    # 7. KD turnkey: text traces -> docs (paragraph split + provenance tag), no network/tokenizer.
    import tempfile as _tf

    kp = os.path.join(_tf.mkdtemp(), "traces.txt")
    with open(kp, "w", encoding="utf-8") as _f:
        _f.write("# header comment\n\nfirst trace paragraph.\n\nsecond trace paragraph.\n")
    kdocs = list(_text_file_to_docs(kp))
    ck("kd: splits paragraphs, drops comments", len(kdocs) == 2)
    ck(
        "kd: docs carry teacher provenance",
        all(d["license"] == "teacher-derived" and d["text"] for d in kdocs),
    )

    # 8. train-shallow/infer-deep gain metric (pure): acc lift, bits drop, flop multiple.
    sg = [
        {"effort": 1, "accuracy": 0.40, "bits": 2.0, "rel_flops": 4.0},
        {"effort": 4, "accuracy": 0.60, "bits": 1.5, "rel_flops": 12.0},
    ]
    g = elasticity_gain(sg)
    ck("elasticity_gain: accuracy lift", abs(g["acc_gain"] - 0.20) < 1e-9)
    ck("elasticity_gain: bits drop", abs(g["bits_drop"] - 0.50) < 1e-9)
    ck("elasticity_gain: flop multiple", abs(g["flops_mult"] - 3.0) < 1e-9)
    ck("elasticity_gain: empty is safe", elasticity_gain([])["acc_gain"] == 0.0)

    npass = sum(c for _, c in checks)
    print(f"\nrisk-coverage selftest: {npass}/{len(checks)} passed")
    return npass == len(checks)


@torch.no_grad()
def collect_token_level(
    ckpt_path, shard_path, device="cuda", max_tokens=200_000, seq_len=1024, effort=None
):
    """Real-run collector: run the model over a tokenized uint16 val shard and
    gather (confidence, correct) for next-token prediction. correct = the
    conf_head's own target (argmax==next-token); conf = sigmoid(conf_head).
    This is the data behind the risk-coverage plot on a real checkpoint."""
    import numpy as np
    from charkha import Charkha, CharkhaConfig
    from dataprep import shard_format_for_vocab

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = CharkhaConfig.from_dict(ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    model = Charkha(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    # shard_path is a raw .bin file (no sibling index.json to read), so the dtype is decided by
    # the CHECKPOINT's own vocab_size -- the shard this ckpt was trained on must match it.
    _, bytes_per_token = shard_format_for_vocab(cfg.vocab_size)
    data = np.memmap(shard_path, dtype=(np.uint16 if bytes_per_token == 2 else np.uint32), mode="r")
    sl = min(seq_len, cfg.max_seq_len)
    conf_all, corr_all, n = [], [], 0
    for s in range(0, len(data) - sl - 1, sl):
        if n >= max_tokens:
            break
        window = torch.from_numpy(data[s : s + sl + 1].astype(np.int64)).to(device)
        idx, tgt = window[:-1].unsqueeze(0), window[1:]
        logits, conf = model(idx, r=effort)
        pred = logits[0].argmax(-1)
        corr_all.append((pred == tgt).float().cpu())
        conf_all.append(conf[0].cpu())
        n += sl
    conf = torch.cat(conf_all).tolist()
    correct = torch.cat(corr_all).tolist()
    return conf, correct


def run_risk_coverage(
    ckpt_path,
    shard_path,
    device="cuda",
    max_tokens=200_000,
    target_risk=0.10,
    delta=0.05,
    effort=None,
):
    """End-to-end on a real checkpoint: collect (conf, correct), print the
    risk-coverage table, AUARC, and a conformal selective threshold."""
    conf, correct = collect_token_level(ckpt_path, shard_path, device, max_tokens, effort=effort)
    base_acc = sum(correct) / len(correct)
    pts = risk_coverage_curve(conf, correct)
    print(f"\n=== Risk-Coverage ({len(correct):,} predictions) ===")
    print(f"  base accuracy (coverage=1.0): {base_acc:.4f}")
    print(f"  AUARC (higher=better):        {auarc(conf, correct):.4f}")
    print(f"  {'coverage':>9} {'accuracy':>9} {'risk':>8} {'conf>=':>8}")
    for target_cov in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        p = min(pts, key=lambda q: abs(q["coverage"] - target_cov))
        print(
            f"  {p['coverage']:>9.3f} {p['accuracy']:>9.4f} {p['risk']:>8.4f} "
            f"{p['threshold']:>8.4f}"
        )
    tau, cov, emp, ub = selective_threshold(conf, correct, target_risk, delta)
    print(
        f"\n  conformal selective prediction (target risk <= {target_risk}, {1 - delta:.0%} conf):"
    )
    if tau is None:
        print("    unreachable on this checkpoint — model not calibrated enough yet")
    else:
        print(
            f"    answer when conf >= {tau:.4f}  ->  coverage {cov:.1%}, "
            f"empirical risk {emp:.4f} (bound {ub:.4f})"
        )
    return {
        "auarc": auarc(conf, correct),
        "base_acc": base_acc,
        "conformal": {"tau": tau, "coverage": cov},
    }


# E1 — compute-elasticity curve: accuracy/NLL vs inference compute.
# A fixed-depth transformer is a single point; a depth-recurrent model forms a curve —
# run more core loops at inference and measure whether accuracy rises.
# Monotonic improvement shows the test-time compute benefit.


@torch.no_grad()
def elasticity_curve(
    model, data, device="cpu", max_tokens=50_000, seq_len=256, efforts=(1, 2, 4, 8, 16)
):
    """For each fixed effort (core loop count) measure next-token accuracy and NLL
    (bits/token) over a uint16 token array. rel_flops is the block-pass count
    (n_prelude + n_coda + effort*n_core) — the honest compute axis. Returns one
    {effort, accuracy, bits, rel_flops} per effort; the plot is accuracy vs rel_flops."""
    import numpy as np

    model.eval()
    cfg = model.cfg
    sl = min(seq_len, cfg.max_seq_len)
    fixed_blocks = cfg.n_prelude + cfg.n_coda
    rows = []
    for eff in efforts:
        nll_sum, ncorrect, ntok = 0.0, 0, 0
        for s in range(0, len(data) - sl - 1, sl):
            if ntok >= max_tokens:
                break
            window = torch.from_numpy(np.asarray(data[s : s + sl + 1]).astype(np.int64)).to(device)
            idx, tgt = window[:-1].unsqueeze(0), window[1:]
            logits, _ = model(idx, r=eff)
            lp = F.log_softmax(logits[0].float(), dim=-1)
            nll_sum += float(F.nll_loss(lp, tgt, reduction="sum"))
            ncorrect += int((logits[0].argmax(-1) == tgt).sum())
            ntok += tgt.numel()
        rows.append(
            {
                "effort": int(eff),
                "accuracy": ncorrect / max(ntok, 1),
                "bits": (nll_sum / max(ntok, 1)) / math.log(2),
                "rel_flops": fixed_blocks + eff * cfg.n_core,
                "tokens": ntok,
            }
        )
    return rows


def elasticity_gain(rows):
    """Quantify the train-shallow/infer-deep payoff from an elasticity curve: the accuracy lift (and
    bits drop) from the cheapest to the most expensive inference effort on a FIXED checkpoint, plus
    the FLOP multiple it cost. acc_gain > 0 means deeper recurrence at inference buys accuracy a
    dense model couldn't — the core CHARKHA thesis, and the cheap pre-check before the expensive run
    (train at low mean_r, then verify the curve still rises when you extrapolate r upward)."""
    if not rows:
        return {"acc_gain": 0.0, "bits_drop": 0.0, "flops_mult": 1.0}
    lo, hi = rows[0], rows[-1]
    return {
        "effort_lo": lo["effort"],
        "effort_hi": hi["effort"],
        "acc_gain": hi["accuracy"] - lo["accuracy"],
        "bits_drop": lo["bits"] - hi["bits"],
        "flops_mult": hi["rel_flops"] / max(lo["rel_flops"], 1e-9),
    }


def run_elasticity(
    ckpt_path, shard_path, device="cuda", max_tokens=50_000, efforts=(1, 2, 4, 8, 16)
):
    """E1 on a real checkpoint: print accuracy/bits vs compute and the headline delta."""
    import numpy as np
    from charkha import Charkha, CharkhaConfig
    from dataprep import shard_format_for_vocab

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = CharkhaConfig.from_dict(ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    model = Charkha(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    _, bytes_per_token = shard_format_for_vocab(cfg.vocab_size)
    data = np.memmap(shard_path, dtype=(np.uint16 if bytes_per_token == 2 else np.uint32), mode="r")
    rows = elasticity_curve(model, data, device, max_tokens, efforts=efforts)
    print(f"\n=== Compute-Elasticity Curve ({rows[0]['tokens']:,} tokens/effort) ===")
    print(f"  {'effort':>6} {'rel_flops':>9} {'accuracy':>9} {'bits/tok':>9}")
    for r in rows:
        print(f"  {r['effort']:>6} {r['rel_flops']:>9} {r['accuracy']:>9.4f} {r['bits']:>9.4f}")
    acc_gain = rows[-1]["accuracy"] - rows[0]["accuracy"]
    bits_drop = rows[0]["bits"] - rows[-1]["bits"]
    verdict = (
        "RISING (test-time compute helps — the curve a dense model cannot make)"
        if acc_gain > 0
        else "FLAT/of inverted (more loops did not help on this ckpt)"
    )
    print(
        f"\n  effort {rows[0]['effort']}->{rows[-1]['effort']}: "
        f"accuracy {acc_gain:+.4f}, bits {bits_drop:+.4f}  =>  {verdict}"
    )
    return rows


# --------------------------------------------------------------------------
# Eval harness — runs lm-eval tasks on a Charkha checkpoint.
# --------------------------------------------------------------------------


def run_eval(
    ckpt_path: str, tasks: list[str], device: str = "cuda", batch_size: int = 8, limit: int = None
):
    """Run standard lm-eval benchmarks on a Charkha checkpoint.
    Loads the model, wraps it for lm-eval's API, runs the requested tasks.
    """
    from charkha import Charkha, CharkhaConfig

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = CharkhaConfig.from_dict(ck["cfg"]) if isinstance(ck["cfg"], dict) else ck["cfg"]
    model = Charkha(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(
        f"Loaded {ckpt_path}: step {ck.get('step', '?')}, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )

    from _verify import load_tokenizer_for

    tok = load_tokenizer_for(getattr(cfg, "tokenizer_name", None))
    if getattr(tok, "vocab_size", cfg.vocab_size) != cfg.vocab_size:
        print(
            f"Note: tokenizer vocab {tok.vocab_size} != model vocab {cfg.vocab_size} "
            f"(model padded to /128)"
        )

    # Wrap for lm-eval's API. simple_evaluate(model=<instance>) requires a proper
    # lm_eval.api.model.LM subclass - the evaluator reads .rank/.world_size/.cache_hook,
    # which the base class supplies; a bare class AttributeErrors on the first of those.
    from lm_eval.api.model import LM

    class CharkhaLM(LM):
        def __init__(self, model, tokenizer, device, batch_size):
            super().__init__()
            self.model = model
            self.tokenizer = tokenizer
            self._device = device
            self._batch_size = batch_size
            self.eos_token_id = tokenizer.eos_token_id
            self.max_length = model.cfg.max_seq_len

        @property
        def device(self):
            return self._device

        def generate_until(self, requests):
            """Greedy-decode each request, honoring its stop strings.
            Greedy (temp=0, top_k=1) makes generative tasks (gsm8k etc.) reproducible;
            we slice generated *token ids* (not string-strip the prompt, which is fragile
            under BPE whitespace round-tripping) and truncate at the first stop string.
            """
            results = []
            for req in requests:
                gen_kwargs = req.args[1] if len(req.args) > 1 else {}
                until = gen_kwargs.get("until", []) or []
                if isinstance(until, str):
                    until = [until]
                inp = self.tokenizer.encode(req.args[0], add_special_tokens=False)
                inp_t = torch.tensor([inp], device=self._device, dtype=torch.long)
                n_new = min(gen_kwargs.get("max_gen_toks", 256), self.max_length - len(inp))
                out = model.generate(
                    inp_t, n_new=n_new, effort=None, temp=0.0, top_k=1
                )  # deterministic greedy
                new_ids = out[0].tolist()[len(inp) :]  # slice ids, don't string-strip
                txt = self.tokenizer.decode(new_ids, skip_special_tokens=True)
                for stop in until:
                    pos = txt.find(stop)
                    if pos != -1:
                        txt = txt[:pos]
                results.append(txt)
            return results

        def loglikelihood(self, requests):
            """Compute log-likelihood of continuations."""
            results = []
            for req in requests:
                ctx, cont = req.args[0], req.args[1]
                full = self.tokenizer.encode(ctx + cont, add_special_tokens=False)
                ctx_ids = self.tokenizer.encode(ctx, add_special_tokens=False)
                cont_ids = full[len(ctx_ids) :]
                if not cont_ids:
                    results.append((0.0, False))
                    continue
                inp = torch.tensor(full, device=self._device).unsqueeze(0)
                with torch.no_grad():
                    logits, _ = model(inp)
                sl = logits[0, len(ctx_ids) - 1 : len(ctx_ids) - 1 + len(cont_ids)].float()
                cont_t = torch.tensor(cont_ids, device=self._device)
                loss = F.cross_entropy(sl, cont_t, reduction="sum")
                # (total log-likelihood, was-greedy): is_greedy must be a real argmax
                # check - the old len(cont)==len(full)-len(ctx) is a tautology (always True).
                is_greedy = bool((sl.argmax(-1) == cont_t).all().item())
                results.append((-loss.item(), is_greedy))
            return results

        def loglikelihood_rolling(self, requests):
            # lm-eval expects the TOTAL log-likelihood of the string (a negative number,
            # summed over tokens) - not a per-token mean, and not positive. We also window
            # docs longer than max_length (non-overlapping; each window's first token is
            # unconditioned - a standard, slightly pessimistic approximation) so a long
            # doc can't run past the model's context and OOM/crash.
            results = []
            for req in requests:
                ids = self.tokenizer.encode(req.args[0], add_special_tokens=False)
                if len(ids) < 2:
                    results.append(0.0)
                    continue
                total = 0.0
                for s in range(0, len(ids), self.max_length):
                    window = ids[s : s + self.max_length]
                    if len(window) < 2:
                        break
                    inp_t = torch.tensor(window, device=self._device).unsqueeze(0)
                    with torch.no_grad():
                        logits, _ = model(inp_t)
                    total += F.cross_entropy(
                        logits[0, :-1].float(),
                        torch.tensor(window[1:], device=self._device),
                        reduction="sum",
                    ).item()
                results.append(-total)
            return results

    wrapper = CharkhaLM(model, tok, device, batch_size)

    import lm_eval

    results = lm_eval.simple_evaluate(
        model=wrapper,
        tasks=tasks,
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
    )
    print("\n=== Eval Results ===")
    for task, metrics in sorted(results.get("results", {}).items()):
        print(f"  {task}:")
        for k, v in sorted(metrics.items()):
            if v is not None:
                print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    return results


# --------------------------------------------------------------------------
# Synthetic-data generator — any teacher (local HF or frontier API); output non-distributable.
# --------------------------------------------------------------------------

SYNTH_PROMPTS = [
    # Reasoning traces
    ("Solve this step by step, showing your reasoning:\n\n{problem}\n\nSolution:"),
    ("Break down this problem and solve it:\n\n{problem}\n\nReasoning:"),
    # Boundary/epistemic ("knows its limits")
    (
        "Answer the following, noting any uncertainty or unknown facts. "
        "If you need to look something up, state what you would search for.\n\n"
        "Question: {question}\n\nResponse:"
    ),
    (
        "Identify what is known, what is unknown, and provide a reasoned answer:\n\n"
        "{question}\n\nAnalysis:"
    ),
    # Textbook-style
    ("Write a clear explanation of the following concept:\n\n{concept}\n\nExplanation:"),
    ("Explain this as if teaching a student:\n\n{concept}\n\n"),
    # Instruction-following
    ("{instruction}\n\nResponse:"),
]

# --------------------------------------------------------------------------
# Pre-written tool-calling examples — teach the model to emit [[calc: ...]] patterns.
# These are written directly to the output (bypass the teacher) so the ~0.4B model
# learns the tool syntax during pretraining.
# --------------------------------------------------------------------------

TOOL_CALLING_EXAMPLES = [
    "user: If I have 847 apples and give away 293, how many are left?\n"
    "assistant: <thinking>[[calc: 847 - 293]] means 554 remain</thinking>\n"
    "You have 554 apples left.\n",
    "user: What's 15% of 280?\n"
    "assistant: <thinking>[[calc: 280 * 0.15]]</thinking>\n"
    "15% of 280 is 42.\n",
    "user: Convert 68 Fahrenheit to Celsius.\n"
    "assistant: <thinking>[[calc: (68 - 32) * 5 / 9]]</thinking>\n"
    "68°F is 20°C.\n",
    "user: A rectangle is 12.5m by 8.3m. What's its area?\n"
    "assistant: <thinking>[[calc: 12.5 * 8.3]]</thinking>\n"
    "The area is 103.75 square meters.\n",
    "user: If a train goes 320 km in 4 hours, what's its speed?\n"
    "assistant: <thinking>[[calc: 320 / 4]]</thinking>\n"
    "The train's speed is 80 km/h.\n",
    "user: I bought items costing $12.99, $8.50, and $24.75. What's the total with 8% tax?\n"
    "assistant: <thinking>[[calc: (12.99 + 8.50 + 24.75) * 1.08]]</thinking>\n"
    "Your total with tax is approximately $49.94.\n",
    "user: How many minutes are in 3 days?\n"
    "assistant: <thinking>[[calc: 3 * 24 * 60]]</thinking>\n"
    "There are 4,320 minutes in 3 days.\n",
    "user: What's the square root of 144?\n"
    "assistant: <thinking>[[calc: 144 ** 0.5]]</thinking>\n"
    "The square root of 144 is 12.\n",
    "user: If I save $150 monthly for 18 months with 5% annual interest compounded monthly, "
    "how much will I have?\n"
    "assistant: <thinking>This needs compound interest. Let me compute step by step. "
    "Monthly rate = 0.05/12. [[calc: 150 * ((1 + 0.05/12) ** 18 - 1) / (0.05/12)]]</thinking>\n"
    "You'll have approximately $2,821 after 18 months.\n",
    "user: A pizza has 8 slices. If 3 people each eat 2 slices, how many are left?\n"
    "assistant: <thinking>[[calc: 8 - 3 * 2]]</thinking>\n"
    "There are 2 slices left.\n",
    "user: What's 7 cubed plus 4 squared?\n"
    "assistant: <thinking>[[calc: 7**3 + 4**2]]</thinking>\n"
    "7³ + 4² = 343 + 16 = 359.\n",
    "user: If a shirt costs $45 and is on sale for 30% off, what's the sale price?\n"
    "assistant: <thinking>[[calc: 45 * (1 - 0.30)]]</thinking>\n"
    "The sale price is $31.50.\n",
]


def _build_synth_seeds():
    """Shared seed problems/questions/concepts/instructions for synthetic generation.
    Returns a list of single-key dicts (key ∈ problem/question/concept/instruction)."""
    import random

    seeds = []
    # Grade-school math
    for a in range(1, 100, 7):
        for b in range(1, 50, 5):
            op = random.choice(["+", "-", "*"])
            seeds.append({"problem": f"What is {a} {op} {b}?"})
    # Logic puzzles
    logic_puzzles = [
        "If all A are B and all B are C, are all A necessarily C? Why?",
        "Is the statement 'this statement is false' true or false? Explain.",
        "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost?",
    ]
    for p in logic_puzzles:
        seeds.append({"problem": p})
    # Concepts
    concepts = [
        "the Pythagorean theorem",
        "Newton's laws of motion",
        "how photosynthesis works",
        "the principle of conservation of energy",
        "what makes a good story",
        "how computers represent numbers",
        "the water cycle",
        "why the sky is blue",
    ]
    for c in concepts:
        seeds.append({"concept": c})
    # General knowledge
    knowledge_qs = [
        "What is the largest planet in our solar system?",
        "What year did World War II end?",
        "What is the chemical symbol for gold?",
        "How many continents are there?",
        "What is the capital of Japan?",
    ]
    for q in knowledge_qs:
        seeds.append({"question": q})
    # Instructions
    instructions = [
        "Write a haiku about a spinning wheel.",
        "Write a Python function to compute Fibonacci numbers.",
        "Explain how to tie a shoelace.",
        "Describe the process of baking bread.",
    ]
    for i in instructions:
        seeds.append({"instruction": i})
    return seeds


def generate_synthetic(
    teacher_id: str,
    output_path: str,
    num_tokens: int,
    device: str = "cuda",
    temperature: float = 0.8,
):
    """Generate synthetic reasoning data with a LOCAL HuggingFace teacher (logit/token gen).
    Any teacher is allowed — frontier-teacher provenance just makes the
    output non-distributable. For a FRONTIER teacher behind an OpenAI-compatible API
    (DeepSeek/GPT/Claude/Gemini via LiteLLM), use generate_synthetic_api instead."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import random

    print(f"Loading teacher: {teacher_id}")
    tok = AutoTokenizer.from_pretrained(teacher_id)
    model = AutoModelForCausalLM.from_pretrained(
        teacher_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"Teacher loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")

    seeds = _build_synth_seeds()

    total_tokens = 0
    written = 0
    f = open(output_path, "w", encoding="utf-8")
    f.write("# CHARKHA synthetic data (local teacher)\n")
    f.write(f"# Teacher: {teacher_id}\n")
    f.write(f"# Date: {time.strftime('%Y-%m-%d')}\n\n")

    # index templates by the placeholder they expect, so a seed is only ever paired with a
    # compatible template (random pairing previously KeyError'd whenever the drawn template's
    # placeholder != the seed's key - i.e. most draws, crashing the synth/distill path).
    tmpl_by_key = {
        k: [t for t in SYNTH_PROMPTS if "{" + k + "}" in t]
        for k in ("problem", "question", "concept", "instruction")
    }

    # Write pre-built tool-calling training examples directly (bypass teacher).
    # These teach the model to emit [[calc: ...]] syntax during inference.
    for ex in TOOL_CALLING_EXAMPLES:
        f.write(ex + "\n")
        total_tokens += len(tok.encode(ex))
        written += 1
    while total_tokens < num_tokens:
        seed = random.choice(seeds)
        key = next(iter(seed))  # each seed has exactly one key
        tmpl = random.choice(tmpl_by_key[key])
        prompt = tmpl.format(**seed)
        inp = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=min(512, num_tokens - total_tokens),
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        full = tok.decode(out[0], skip_special_tokens=True)
        # Keep only the generated part (strip the prompt)
        response = full[len(prompt) :] if full.startswith(prompt) else full
        if len(response) < 20:  # skip empty/degenerate
            continue
        f.write(full + "\n\n")
        nt = len(out[0]) - len(inp["input_ids"][0])
        total_tokens += nt
        written += 1
        if written % 10 == 0:
            print(f"  {total_tokens:,}/{num_tokens:,} tokens, {written} examples")
            f.flush()

    f.close()
    print(f"Done: {written} examples, {total_tokens:,} tokens -> {output_path}")


def _chat_completion(
    api_base, api_key, model, prompt, system, max_tokens, temperature, timeout=120
):
    """One OpenAI-compatible /chat/completions call via stdlib urllib (no extra dependency).
    Works against LiteLLM, DeepSeek, OpenAI, vLLM, Together, etc. Returns the assistant text."""
    import urllib.request
    import json as _json

    url = api_base.rstrip("/") + "/chat/completions"
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body = _json.dumps(
        {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature}
    ).encode()
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "charkha-kd/1.0 (Linux)",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = _json.loads(r.read().decode())
    msg = out["choices"][0]["message"]
    # Reasoning models (gpt-oss, DeepSeek-R1, etc.) emit chain-of-thought in `reasoning_content`
    # and the final answer in `content`. For sequence-level KD we want BOTH — the CoT is the most
    # valuable signal for a student learning to think. Stitch them so the trace reads naturally and
    # we never return an empty string when the model spent its budget reasoning.
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    if reasoning and content:
        return f"{reasoning}\n\n{content}"
    return content or reasoning


def generate_synthetic_api(
    api_base: str,
    model: str,
    output_path: str,
    num_tokens: int,
    api_key: str = "",
    temperature: float = 0.8,
    max_new: int = 1024,
    est_chars_per_token: float = 4.0,
):
    """Sequence-level KD from a FRONTIER teacher behind an OpenAI-compatible API (DeepSeek-V4 via the
    user's LiteLLM, GPT/Claude/Gemini-class, etc.). Generates reasoning + tool-call traces as TEXT
    (tokenizer-agnostic — the teacher's tokenizer mismatch with GPT-NeoX is irrelevant for seq-level
    KD; dataprep re-tokenizes into token shards).
    Frontier-KD data is teacher-derived; check your teacher's terms of service.

    No torch/transformers needed — pure stdlib HTTP. Token count is ESTIMATED from char length
    (we don't load the GPT-NeoX tokenizer here); dataprep gives the exact count downstream."""
    import random

    system = (
        "You are a careful teacher generating training data for a small student model. "
        "Reason step by step. When a calculation is needed, show it inline as "
        "[[calc: <expression>]]. Note uncertainty honestly rather than guessing."
    )
    seeds = _build_synth_seeds()
    tmpl_by_key = {
        k: [t for t in SYNTH_PROMPTS if "{" + k + "}" in t]
        for k in ("problem", "question", "concept", "instruction")
    }

    est_tokens = 0
    written = 0
    f = open(output_path, "w", encoding="utf-8")
    f.write("# CHARKHA synthetic data (frontier teacher, sequence-level KD)\n")
    f.write(f"# Teacher: {model} via {api_base}\n")
    f.write("# PROVENANCE: frontier-KD — personal / non-distributable\n")
    f.write(f"# Date: {time.strftime('%Y-%m-%d')}\n\n")

    # Tool-syntax examples written directly (same as the local path) so the student learns [[calc:]].
    for ex in TOOL_CALLING_EXAMPLES:
        f.write(ex + "\n")
        est_tokens += int(len(ex) / est_chars_per_token)
        written += 1

    fails = 0
    while est_tokens < num_tokens:
        seed = random.choice(seeds)
        key = next(iter(seed))
        tmpl = random.choice(tmpl_by_key[key])
        prompt = tmpl.format(**seed)
        try:
            resp = _chat_completion(
                api_base,
                api_key,
                model,
                prompt,
                system,
                max_tokens=max_new,
                temperature=temperature,
            )
        except Exception as e:
            fails += 1
            print(f"  [api error {fails}] {type(e).__name__}: {e}")
            if fails >= 20:
                print("  too many API failures — stopping")
                break
            continue
        if not resp or len(resp) < 20:
            continue
        f.write(prompt + "\n" + resp + "\n\n")
        est_tokens += int((len(prompt) + len(resp)) / est_chars_per_token)
        written += 1
        if written % 10 == 0:
            print(f"  ~{est_tokens:,}/{num_tokens:,} est tokens, {written} examples")
            f.flush()

    f.close()
    print(f"Done: {written} examples, ~{est_tokens:,} est tokens -> {output_path}")
    print("Next: tokenize with dataprep to get exact uint16 shards + a true token count.")


def _text_file_to_docs(path):
    """Split a generated-traces text file into dataprep documents (paragraphs on blank lines),
    tagging teacher provenance. Pure (no network/tokenizer) so the selftest can exercise it."""
    with open(path, "r", encoding="utf-8") as f:
        blob = f.read()
    for i, para in enumerate(re.split(r"\n\s*\n", blob)):
        para = para.strip()
        if para and not para.startswith("#"):
            yield {"id": f"kd_{i}", "text": para, "license": "teacher-derived", "url": ""}


def kd_pipeline(
    out_dir,
    tokens,
    *,
    api_base=None,
    api_model=None,
    api_key="",
    teacher=None,
    device="cpu",
    temperature=0.8,
    tokenizer="charkha_tokenizer.json",
):
    """Turnkey distilled-data pipeline: generate teacher traces -> tokenize -> uint16 shards,
    ready for `train.py --data <out_dir>`. Uses the frontier API teacher when api_base is set,
    else a local HF teacher. The license gate is OFF (teacher-derived data is non-distributable)."""
    os.makedirs(out_dir, exist_ok=True)
    text_path = os.path.join(out_dir, "traces.txt")
    if api_base:
        generate_synthetic_api(
            api_base, api_model, text_path, tokens, api_key=api_key, temperature=temperature
        )
    elif teacher:
        generate_synthetic(teacher, text_path, tokens, device, temperature)
    else:
        raise ValueError("set either api_base/api_model or an explicit teacher model")
    from dataprep import run_pipeline

    cfg = {
        "allow": [],
        "deny": [],
        "optout": [],
        "bench": [],
        "min_words": 5,
        "dedup_mode": "exact",
        "tokenizer": tokenizer,
        "shard_tokens": 100_000_000,
        "license_gate": False,
    }
    idx, _ = run_pipeline(_text_file_to_docs(text_path), out_dir, cfg)
    print(f"[kd] {idx['total_tokens']:,} tokens -> {out_dir}  (train with --data {out_dir})")
    return out_dir


# --------------------------------------------------------------------------
