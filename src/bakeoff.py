"""
CHARKHA bakeoff — two (or N) checkpoints enter, one model leaves.
================================================================================
Built for the "two parallel runs, which wins / can we fuse them" question (local 4060 Ti vs a
cloud booster, same arch + same data, DIFFERENT random init). It answers three things end to end:

  1. SCOREBOARD   — val NLL / ppl / bits-per-token for each individual checkpoint, every weight
                    MERGE (soup / slerp / ties), the output-space ENSEMBLE, and the ENSEMBLE
                    DISTILLED back into one student. One table, a declared winner.
  2. DIVERGENCE   — a broad DOMAIN suite (arithmetic, facts, code, reasoning, language, chat).
                    For every probe it greedy-completes from each model, measures how much the
                    two AGREE (token overlap + first-step symmetric-KL), auto-scores who's RIGHT
                    where a checker exists, and surfaces the biggest disagreements — the
                    capability-divergence map (like the training milestones, but pairwise).
  3. FUSION INFRA — the actual machinery to turn 2 models into 1 WITHOUT the weight-merge basis
                    wall: an EnsembleTeacher (averaged top-k distribution) feeding the existing
                    memory-frugal KD path (charkha.forward(kd=...)) to train a single student.

Weight-merge (soup/ties) only works when checkpoints share a basin (shared init); ensemble->distill
works regardless of init because it combines OUTPUTS, not weights. Bakeoff runs BOTH and lets the
numbers decide — see README.md / the merge-vs-distill discussion.

Optional HITL: `--hitl N` writes the N most-divergent un-auto-scorable probes to a review JSONL you
fill in (winner: a|b|tie|neither); `--apply-hitl FILE` folds your verdicts back into the tally.
Lightweight by design — you only ever judge the handful of genuinely ambiguous cases.

  python src/bakeoff.py --selftest                      # hermetic toy bakeoff (CPU, no download)
  python src/bakeoff.py --a model-a.pt --b model-b.pt --shard data/eval-dd --all
  python src/bakeoff.py --a model-a.pt --b model-b.pt --shard data/eval-dd --merge --distill --hitl 10

"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import glob
import tempfile

import torch

try:  # we print decoded model bytes; don't die on win32 cp1252
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

MAXCTX = 512  # rolling context cap for generation / windowed NLL


def _bar(frac, width=24):
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def _progress(prefix, i, total, suffix="", done=False):
    """Single-line, carriage-return progress meter. ASCII-only (win32 cp1252-safe terminals)."""
    bar = _bar(i / max(1, total))
    line = f"\r  {prefix} {bar} {i}/{total} {suffix}"
    pad = " " * max(0, 100 - len(line))
    print(line + pad, end="\n" if done else "", flush=True)


# ============================================================================
# Loading
# ============================================================================
def load_one(path, device, toy=False, digit_split=False):
    """Return (model.eval(), tokenizer, cfg). Reuses serve.load_model so tokenizer/cfg handling
    is identical to real inference."""
    from serve import load_model

    model, tok, cfg = load_model(None if toy else path, device, toy=toy, digit_split=digit_split)
    model.eval()
    return model, tok, cfg


# ============================================================================
# Output-space combination (NO weight averaging — sidesteps the LMC basin barrier)
# ============================================================================
@torch.no_grad()
def _avg_probs_full(models, weights, idx):
    """Weighted-average next-token probability over models, full (B,T,V). Teacher-forced."""
    probs = None
    for w, m in zip(weights, models):
        lg, _ = m(idx)  # (B,T,V) — inference path, no targets
        p = lg.float().softmax(-1) * w
        probs = p if probs is None else probs + p
    return probs / sum(weights)


@torch.no_grad()
def _avg_logprobs_last(models, weights, idx):
    """Weighted-average next-token logprob at the LAST position only, (B,V) — for generation."""
    probs = None
    for w, m in zip(weights, models):
        lg, _ = m(idx)
        p = lg[:, -1].float().softmax(-1) * w
        probs = p if probs is None else probs + p
    return (probs / sum(weights)).clamp_min(1e-12).log()


@torch.no_grad()
def greedy(models, weights, prompt_ids, n, device):
    """Greedy continuation from the (weighted) combination. models=[m] => single-model greedy;
    models=[a,b] => ensemble greedy. Returns the generated id list (prompt excluded)."""
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    out = []
    for _ in range(n):
        lp = _avg_logprobs_last(models, weights, idx)
        nt = int(lp.argmax(-1))
        out.append(nt)
        idx = torch.cat([idx, torch.tensor([[nt]], device=device)], 1)
        if idx.size(1) > MAXCTX:
            idx = idx[:, -MAXCTX:]
    return out


# ============================================================================
# 1. SCOREBOARD — windowed NLL / ppl / bits over a tokenized shard
# ============================================================================
def read_shard_ids(path, max_tokens):
    """Read up to max_tokens ids from a .bin file or the first .bin in a shard dir. dtype (uint16
    vs uint32) is decided by the sibling index.json's vocab_size -- either `path` IS the shard dir
    (index.json right there) or `path` is a .bin file (index.json is its sibling in the same dir)."""
    import json
    import numpy as np
    from dataprep import shard_format_for_vocab

    if os.path.isdir(path):
        idx_dir = path
        bins = sorted(glob.glob(os.path.join(path, "*.bin")))
        if not bins:
            raise FileNotFoundError(f"no .bin shards under {path}")
        path = bins[0]
    else:
        idx_dir = os.path.dirname(path)
    vocab_size = None
    idx_path = os.path.join(idx_dir, "index.json")
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            vocab_size = json.load(f).get("vocab_size")
    _, bytes_per_token = shard_format_for_vocab(vocab_size)
    dtype = np.uint16 if bytes_per_token == 2 else np.uint32
    arr = np.fromfile(path, dtype=dtype, count=max_tokens)
    return arr.astype(np.int64)


@torch.no_grad()
def score_nll(models, weights, ids, T, device, vocab=None, label=None):
    """Mean NLL / ppl / bits-per-token of the (combined) predictor over teacher-forced windows.
    Lower is better — this is the headline 'who is on top' number."""
    import numpy as np

    n_tok = 0
    nll_sum = 0.0
    step = T
    windows = list(range(0, len(ids) - T - 1, step))
    t0 = time.time()
    for wi, s in enumerate(windows):
        win = ids[s : s + T + 1]
        if vocab is not None:
            win = np.clip(win, 0, vocab - 1)
        idx = torch.tensor(win[:-1], dtype=torch.long, device=device).unsqueeze(0)  # (1,T)
        tgt = torch.tensor(win[1:], dtype=torch.long, device=device)  # (T,)
        probs = _avg_probs_full(models, weights, idx)[0]  # (T,V)
        lp = probs.clamp_min(1e-12).log()
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        nll_sum += float(nll.sum())
        n_tok += tgt.numel()
        if label:
            running_ppl = float(torch.exp(torch.tensor(nll_sum / max(1, n_tok))))
            _progress(
                f"[score:{label}]",
                wi + 1,
                len(windows),
                f"| running ppl {running_ppl:7.2f} | {n_tok / (time.time() - t0):.0f} tok/s",
                done=(wi == len(windows) - 1),
            )
    if n_tok == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = nll_sum / n_tok
    ppl = float(torch.exp(torch.tensor(mean)))
    bits = mean / 0.6931471805599453
    return mean, ppl, bits, n_tok


# ============================================================================
# 2. DIVERGENCE — domain capability map + pairwise disagreement
# ============================================================================
def _firstint(s):
    m = re.search(r"-?\d+", s)
    return int(m.group()) if m else None


def _num_eq(target):
    return lambda txt: _firstint(txt) == target


def _contains(*subs):
    subs = [x.lower() for x in subs]
    return lambda txt: any(x in txt.lower() for x in subs)


# (domain, prompt, checker-or-None). A checker maps the decoded completion -> True/False (model is
# "right"). Where no objective checker exists we still measure agreement/divergence, just no winner.
DOMAINS = [
    ("arithmetic", "1 + 1 = ", _num_eq(2)),
    ("arithmetic", "2 + 3 = ", _num_eq(5)),
    ("arithmetic", "7 + 5 = ", _num_eq(12)),
    ("arithmetic", "6 * 7 = ", _num_eq(42)),
    ("arithmetic", "10 - 4 = ", _num_eq(6)),
    ("facts", "The capital of France is ", _contains("paris")),
    ("facts", "The sky is ", _contains("blue")),
    ("facts", "The opposite of hot is ", _contains("cold")),
    ("facts", "Water is made of hydrogen and ", _contains("oxygen")),
    ("facts", "The largest planet in the solar system is ", _contains("jupiter")),
    ("code", "def add(a, b):\n    return ", _contains("a + b", "a+b")),
    ("code", "for i in range(10):\n    print(", _contains("i")),
    ("code", "import numpy as ", _contains("np")),
    ("reasoning", "All cats are animals. Tom is a cat. Therefore Tom is an ", _contains("animal")),
    (
        "reasoning",
        "If it is raining, the ground is wet. It is raining, so the ground is ",
        _contains("wet"),
    ),
    ("reasoning", "A is bigger than B. B is bigger than C. So A is bigger than ", _contains("c")),
    ("language", "Once upon a time, there ", None),
    ("language", "The quick brown fox jumps over the lazy ", _contains("dog")),
    ("language", "Roses are red, violets are ", _contains("blue")),
    ("chat", "Hi! How are ", _contains("you")),
    ("chat", "Q: What is your name?\nA: ", None),
    ("chat", "Hello, my name is ", None),
]


@torch.no_grad()
def _sym_kl_first(a, b, prompt_ids, device, vocab=None):
    """Symmetric KL between the two models' next-token distributions at the FIRST generated step.
    A cheap scalar 'how differently do they think here' (0 = identical)."""

    pi = list(prompt_ids)
    if vocab is not None:
        pi = [min(max(t, 0), vocab - 1) for t in pi]
    idx = torch.tensor([pi], dtype=torch.long, device=device)
    pa = _avg_probs_full([a], [1.0], idx)[0, -1]
    pb = _avg_probs_full([b], [1.0], idx)[0, -1]
    pa = pa.clamp_min(1e-12)
    pb = pb.clamp_min(1e-12)
    kl = (pa * (pa.log() - pb.log())).sum() + (pb * (pb.log() - pa.log())).sum()
    return float(kl)


def divergence_probe(a, b, tok, device, n=24, extra=None, vocab=None, verbose=True):
    """For each domain probe: greedy-complete from A and B, decode, score correctness where a
    checker exists, measure token-overlap agreement + first-step symmetric KL. Returns a list of
    per-probe dicts and a per-domain aggregate."""
    probes = list(DOMAINS) + (extra or [])
    rows = []
    if verbose:
        print(
            f"\n[divergence] {len(probes)} probes x 2 models, {n} tok each — watching them diverge live:"
        )
    for i, (domain, prompt, checker) in enumerate(probes):
        pid = tok.encode(prompt) or [0]
        ga = greedy([a], [1.0], pid, n, device)
        gb = greedy([b], [1.0], pid, n, device)
        ta, tb = tok.decode(ga), tok.decode(gb)
        overlap = sum(1 for x, y in zip(ga, gb) if x == y) / max(len(ga), 1)
        kl = _sym_kl_first(a, b, pid, device, vocab=vocab)
        win = None
        if checker is not None:
            ca, cb = bool(checker(ta)), bool(checker(tb))
            win = (
                "a"
                if ca and not cb
                else "b"
                if cb and not ca
                else "both"
                if ca and cb
                else "neither"
            )
        rows.append(
            {
                "domain": domain,
                "prompt": prompt,
                "a": ta,
                "b": tb,
                "overlap": overlap,
                "sym_kl": kl,
                "auto_win": win,
                "checked": checker is not None,
            }
        )
        if verbose:
            tag = f"auto_win={win}" if win else "unchecked"
            print(f"  [{i + 1:2d}/{len(probes)}] {domain:10s} {prompt!r}")
            print(f"         a: {ta!r}")
            print(f"         b: {tb!r}")
            print(f"         overlap={overlap:.2f} sym_kl={kl:.3f} {tag}")
    # aggregate
    agg = {}
    for r in rows:
        d = agg.setdefault(
            r["domain"],
            {
                "n": 0,
                "overlap": 0.0,
                "kl": 0.0,
                "a": 0,
                "b": 0,
                "both": 0,
                "neither": 0,
                "checked": 0,
            },
        )
        d["n"] += 1
        d["overlap"] += r["overlap"]
        d["kl"] += r["sym_kl"]
        if r["auto_win"]:
            d["checked"] += 1
            d[r["auto_win"]] += 1
    for d in agg.values():
        d["overlap"] /= d["n"]
        d["kl"] /= d["n"]
    return rows, agg


# ============================================================================
# 3. FUSION — EnsembleTeacher + distill the ensemble into ONE student
# ============================================================================
class EnsembleTeacher:
    """Emits the averaged-over-models top-k next-token distribution in the (idx, prob) shape the
    KD path expects. This is the output-space fusion that has NO basin/LMC constraint."""

    def __init__(self, models, weights, k=32, temp=2.0):
        from distill import topk_from_logits

        self.models, self.weights, self.k, self.temp = models, weights, k, temp
        self._topk = topk_from_logits

    @torch.no_grad()
    def topk(self, x):
        probs = _avg_probs_full(self.models, self.weights, x)[:, :-1]  # (B,T-1,V), align w/ targets
        logp = probs.clamp_min(1e-12).log()  # feed as "logits"
        return self._topk(logp, self.k, self.temp)


def distill_ensemble(
    models,
    weights,
    cfg,
    ids,
    device,
    steps=200,
    T=256,
    lr=3e-3,
    k=32,
    temp=2.0,
    ce_weight=0.5,
    log_every=1,
    tok=None,
    sample_every=0,
    sample_prompt="The ",
    sample_tokens=40,
    offload_optim=True,
):
    """Train a FRESH student to match the ensemble's averaged top-k (KD) + the true next token
    (CE). Returns the student state_dict + cfg. This is the 'two models -> one model' artifact.

    offload_optim=True streams the student's Muon momentum to CPU RAM (same trick as train.py's
    --offload-optim) and turns on grad_checkpoint — the frozen A+B teachers plus a from-scratch
    student (params+grads+optimizer state) is 3 models' worth of VRAM at once, the tightest point
    in the whole bake-off on an 8GB card. tok+sample_every prints actual generated text from the
    student periodically, so you can watch it sharpen instead of staring at a loss number."""
    from charkha import Charkha, CharkhaConfig, build_optimizers
    import numpy as np

    scfg = CharkhaConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
    if offload_optim and device == "cuda":
        scfg.grad_checkpoint = True
    student = Charkha(scfg).to(device).train()
    teacher = EnsembleTeacher(models, weights, k=k, temp=temp)
    opts = build_optimizers(student, adam_lr=lr, offload=offload_optim)  # (muon, adam) tuple
    vocab = scfg.vocab_size
    nwin = max(1, (len(ids) - T - 1))
    amp = device == "cuda"
    t0 = time.time()
    for step in range(steps):
        s = (step * T) % nwin
        win = np.clip(ids[s : s + T + 1], 0, vocab - 1)
        x = torch.tensor(win[:-1], dtype=torch.long, device=device).unsqueeze(0)
        tgt = torch.tensor(win[1:], dtype=torch.long, device=device).unsqueeze(0)
        from distill import teacher_kd_tuple

        kd = teacher_kd_tuple(teacher, x, kd_weight=1.0, kd_temp=temp)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp, cache_enabled=False):
            _, loss = student(
                x, tgt if ce_weight > 0 else x, kd=kd
            )  # cache off: checkpoint determinism
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            o.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            _progress(
                "[distill]",
                step + 1,
                steps,
                f"| loss {loss.item():7.4f} | {((step + 1) * T) / (time.time() - t0):.0f} tok/s",
                done=(step == steps - 1),
            )
        if tok is not None and sample_every and (step % sample_every == 0 or step == steps - 1):
            student.eval()
            pid = tok.encode(sample_prompt) or [0]
            gen = greedy([student], [1.0], pid, sample_tokens, device)
            student.train()
            print(f"\n  [student @ step {step + 1}] {sample_prompt!r} -> {tok.decode(gen)!r}")
    student.eval()
    return student.state_dict(), (vars(scfg) if not isinstance(cfg, dict) else cfg)


# ============================================================================
# Weight-merge wrappers (will fail to BEAT parents on different-init models — run to PROVE it)
# ============================================================================
def build_merges(a_path, b_path, outdir, base_path=None, density=0.2, t=0.5):
    """Produce soup/slerp(/ties) checkpoints from the two parents. Returns {name: path}."""
    from merge import _load, _save, soup, slerp, ties

    os.makedirs(outdir, exist_ok=True)
    sa, ca = _load(a_path)
    sb, cb = _load(b_path)
    made = {}
    try:
        merged = soup([sa, sb])
        p = os.path.join(outdir, "soup.pt")
        _save(p, merged, ca)
        made["soup"] = p
    except Exception as e:
        print(f"  [merge] soup failed: {type(e).__name__}: {e}")
    try:
        merged = slerp(sa, sb, t=t)
        p = os.path.join(outdir, "slerp.pt")
        _save(p, merged, ca)
        made["slerp"] = p
    except Exception as e:
        print(f"  [merge] slerp failed: {type(e).__name__}: {e}")
    if base_path:
        try:
            sbase, _ = _load(base_path)
            merged = ties([sa, sb], sbase, density=density)
            p = os.path.join(outdir, "ties.pt")
            _save(p, merged, ca)
            made["ties"] = p
        except Exception as e:
            print(f"  [merge] ties failed: {type(e).__name__}: {e}")
    return made


# ============================================================================
# HITL — write the most-divergent un-auto-scorable probes for human review
# ============================================================================
def write_hitl(rows, path, top_n):
    """Pick the top-N highest-divergence probes WITHOUT an auto checker, write a review JSONL."""
    cand = [r for r in rows if not r["checked"]]
    cand.sort(key=lambda r: (-r["sym_kl"], r["overlap"]))
    picked = cand[:top_n]
    with open(path, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(
                json.dumps(
                    {
                        "domain": r["domain"],
                        "prompt": r["prompt"],
                        "completion_a": r["a"],
                        "completion_b": r["b"],
                        "sym_kl": round(r["sym_kl"], 3),
                        "winner": "",
                    }
                )
                + "\n"
            )
    return len(picked), path


def apply_hitl(path):
    """Read back human verdicts (winner in a|b|tie|neither). Returns a tally dict."""
    tally = {"a": 0, "b": 0, "tie": 0, "neither": 0, "unjudged": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            w = (json.loads(line).get("winner") or "").strip().lower()
            tally[w if w in tally else "unjudged"] += 1
    return tally


# ============================================================================
# Report / dashboard (ASCII only — win32 cp1252 safe)
# ============================================================================
def hr(c="="):
    return c * 78


def print_scoreboard(scores):
    print("\n" + hr())
    print("  SCOREBOARD  (lower ppl / bits = better)")
    print("  " + hr("-"))
    print(f"  {'model':<14}{'val nll':>10}{'ppl':>12}{'bits/tok':>12}{'tokens':>12}")
    best = min((s for s in scores if s["ppl"] == s["ppl"]), key=lambda s: s["ppl"], default=None)
    for s in scores:
        tag = "  <-- WINNER" if best and s["name"] == best["name"] else ""
        print(
            f"  {s['name']:<14}{s['nll']:>10.4f}{s['ppl']:>12.2f}{s['bits']:>12.4f}{s['tokens']:>12,}{tag}"
        )
    print("  " + hr("-"))
    if best:
        print(f"  WINNER: {best['name']}  (ppl {best['ppl']:.2f})")


def print_divergence(agg, rows):
    print("\n" + hr())
    print("  DIVERGENCE MAP  (overlap=greedy token agreement, sym_kl=how differently they think)")
    print("  " + hr("-"))
    print(f"  {'domain':<12}{'overlap':>9}{'sym_kl':>9}   {'auto: A / B / both / neither'}")
    for d, v in sorted(agg.items()):
        ab = (
            f"{v['a']} / {v['b']} / {v['both']} / {v['neither']}"
            if v["checked"]
            else "(no checker)"
        )
        print(f"  {d:<12}{v['overlap'] * 100:>8.0f}%{v['kl']:>9.2f}   {ab}")
    print("  " + hr("-"))
    a_tot = sum(v["a"] for v in agg.values())
    b_tot = sum(v["b"] for v in agg.values())
    print(f"  capability wins (objective probes only):  A={a_tot}   B={b_tot}")
    top = sorted(rows, key=lambda r: -r["sym_kl"])[:5]
    print("  biggest disagreements:")
    for r in top:
        print(f"    [{r['domain']}] {r['prompt']!r}")
        print(f"        A-> {r['a'][:60]!r}")
        print(f"        B-> {r['b'][:60]!r}")


# ============================================================================
# Orchestration
# ============================================================================
def run_bakeoff(args, device):
    report = {}
    # --- load the two parents ---
    print(hr())
    print(f"[bakeoff] loading A: {args.a or 'toyA'}")
    a, tok, cfg_a = load_one(args.a, device, toy=args.toy, digit_split=args.digit_split)
    print(f"[bakeoff] loading B: {args.b or 'toyB'}")
    b, _, cfg_b = load_one(args.b, device, toy=args.toy, digit_split=args.digit_split)
    vocab = (cfg_a if isinstance(cfg_a, dict) else vars(cfg_a))["vocab_size"]
    print(f"[bakeoff] both loaded. vocab={vocab}  device={device}")
    print(hr())

    # --- data for NLL + distillation ---
    if args.shard:
        print(f"[bakeoff] reading up to {args.max_tokens:,} tokens from {args.shard} ...")
        ids = read_shard_ids(args.shard, args.max_tokens)
        print(f"[bakeoff] got {len(ids):,} tokens.")
    else:
        ids = None

    # --- SCOREBOARD: individuals + ensemble (+ merges + distilled if requested) ---
    # NOTE: merge/distilled candidates are scored and FREED immediately, one at a time, rather
    # than kept resident alongside A/B. On an 8GB card, A+B+soup+slerp+ties+student all loaded
    # at once (fp32, plus the student's own optimizer state) is enough to spill out of VRAM and
    # crawl badly instead of running at full speed. Only A and B stay resident throughout (the
    # divergence map needs them till the end); everything else is load -> score -> drop.
    def _free(m):
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

    scores = []
    if ids is not None:
        print("\n[stage 1/4] scoring A, B, ensemble ...")
        for name, (models, weights) in (
            ("A", ([a], [1.0])),
            ("B", ([b], [1.0])),
            ("ensemble", ([a, b], [args.weight_a, 1.0 - args.weight_a])),
        ):
            nll, ppl, bits, ntok = score_nll(
                models, weights, ids, args.T, device, vocab=vocab, label=name
            )
            scores.append({"name": name, "nll": nll, "ppl": ppl, "bits": bits, "tokens": ntok})
            print(f"  -> {name}: ppl={ppl:.2f}  bits/tok={bits:.3f}  nll={nll:.4f}  ({ntok:,} tok)")

    if args.merge or args.all:
        print("\n[stage 2/4] building + scoring weight-merges (soup/slerp/ties) ...")
        made = (
            build_merges(
                args.a,
                args.b,
                args.outdir,
                base_path=args.base,
                density=args.density,
                t=args.slerp_t,
            )
            if not args.toy
            else {}
        )
        for name, p in made.items():
            print(f"  loading merge candidate: {name} <- {p}")
            m, _, _ = load_one(p, device)
            if ids is not None:
                nll, ppl, bits, ntok = score_nll(
                    [m], [1.0], ids, args.T, device, vocab=vocab, label=name
                )
                scores.append({"name": name, "nll": nll, "ppl": ppl, "bits": bits, "tokens": ntok})
                print(
                    f"  -> {name}: ppl={ppl:.2f}  bits/tok={bits:.3f}  nll={nll:.4f}  ({ntok:,} tok)"
                )
            _free(m)
            print(f"  freed {name} from VRAM.")
        report["merges"] = list(made.keys())
    else:
        print("\n[stage 2/4] weight-merges skipped (pass --merge or --all to run them).")

    student_state = None
    if (args.distill or args.all) and ids is not None:
        print(
            f"\n[stage 3/4] distilling the ensemble into ONE student ({args.distill_steps} steps) ..."
        )
        cfg_for_student = cfg_a if isinstance(cfg_a, dict) else vars(cfg_a)
        student_state, scfg = distill_ensemble(
            [a, b],
            [args.weight_a, 1.0 - args.weight_a],
            cfg_for_student,
            ids,
            device,
            steps=args.distill_steps,
            T=min(args.T, 256),
            tok=tok,
            sample_every=max(1, args.distill_steps // 8),
        )
        sp = os.path.join(args.outdir, "distilled.pt")
        os.makedirs(args.outdir, exist_ok=True)
        tmp = sp + ".tmp"
        torch.save({"model": student_state, "cfg": scfg, "step": 0}, tmp)
        os.replace(tmp, sp)
        print(f"  saved -> {sp}, scoring it now ...")
        sm, _, _ = load_one(sp, device)
        nll, ppl, bits, ntok = score_nll(
            [sm], [1.0], ids, args.T, device, vocab=vocab, label="distilled"
        )
        scores.append({"name": "distilled", "nll": nll, "ppl": ppl, "bits": bits, "tokens": ntok})
        print(f"  -> distilled: ppl={ppl:.2f}  bits/tok={bits:.3f}  nll={nll:.4f}  ({ntok:,} tok)")
        _free(sm)
        report["distilled_ckpt"] = sp
    else:
        print(
            "\n[stage 3/4] distillation skipped (pass --distill or --all, with --shard set, to run it)."
        )

    if ids is not None:
        print_scoreboard(scores)
        report["scoreboard"] = scores
    else:
        print("\n[bakeoff] no --shard given: skipping NLL scoreboard (divergence map still runs).")

    # --- DIVERGENCE MAP ---
    print(f"\n[stage 4/4] domain divergence map ({len(DOMAINS)} probes) ...")
    rows, agg = divergence_probe(a, b, tok, device, n=args.gen_tokens, vocab=vocab)
    print_divergence(agg, rows)
    report["divergence"] = {
        "aggregate": agg,
        "probes": [
            {k: r[k] for k in ("domain", "prompt", "overlap", "sym_kl", "auto_win")} for r in rows
        ],
    }

    # --- HITL ---
    if args.hitl:
        n, path = write_hitl(rows, os.path.join(args.outdir, "bakeoff_hitl.jsonl"), args.hitl)
        os.makedirs(args.outdir, exist_ok=True)
        write_hitl(rows, path, args.hitl)
        print(f"\n[hitl] wrote {n} ambiguous probes to {path}")
        print("       fill in winner: a|b|tie|neither, then: --apply-hitl " + path)

    # --- report file ---
    os.makedirs(args.outdir, exist_ok=True)
    rp = os.path.join(args.outdir, "bakeoff_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[bakeoff] full report -> {rp}")
    return report


# ============================================================================
# Selftest (hermetic: two DIFFERENT-init toy models, CPU, no download)
# ============================================================================
def selftest():
    from charkha import Charkha, CharkhaConfig
    from train import make_synthetic_shards

    ok = 0

    def ck(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        assert cond, name
        ok += 1

    dev = "cpu"

    torch.manual_seed(1)
    ca = CharkhaConfig.toy()
    ca.vocab_size = 384
    A = Charkha(ca).eval()
    torch.manual_seed(2)
    cb = CharkhaConfig.toy()
    cb.vocab_size = 384
    B = Charkha(cb).eval()
    ck(
        "two toy models built with different seeds",
        not torch.equal(next(A.parameters()), next(B.parameters())),
    )

    # ensemble logprobs: shape + valid distribution
    idx = torch.randint(0, 384, (1, 16))
    p = _avg_probs_full([A, B], [0.5, 0.5], idx)
    ck("ensemble probs shape", tuple(p.shape) == (1, 16, 384))
    ck("ensemble probs normalize", torch.allclose(p.sum(-1), torch.ones(1, 16), atol=1e-4))

    # greedy works for single and ensemble
    g1 = greedy([A], [1.0], [10, 20, 30], 8, dev)
    ge = greedy([A, B], [0.5, 0.5], [10, 20, 30], 8, dev)
    ck("single greedy length", len(g1) == 8)
    ck("ensemble greedy length", len(ge) == 8)

    # NLL scoreboard over a synthetic shard
    tmp = tempfile.mkdtemp(prefix="bakeoff_")
    make_synthetic_shards(os.path.join(tmp, "data"), n_shards=2, toks_per=6000)
    ids = read_shard_ids(os.path.join(tmp, "data"), 4000)
    nll, ppl, bits, ntok = score_nll([A], [1.0], ids, 64, dev, vocab=384)
    ck("single NLL finite", nll == nll and ppl > 0 and ntok > 0)
    nlle, pple, *_ = score_nll([A, B], [0.5, 0.5], ids, 64, dev, vocab=384)
    ck("ensemble NLL finite", nlle == nlle and pple > 0)

    # divergence probe runs, produces per-domain aggregate
    from serve import ByteTokenizer

    rows, agg = divergence_probe(A, B, ByteTokenizer(), dev, n=8, vocab=384)
    ck("divergence rows for every probe", len(rows) == len(DOMAINS))
    ck("aggregate has domains", set(agg) == {d for d, _, _ in DOMAINS})
    ck("overlap in [0,1]", all(0.0 <= r["overlap"] <= 1.0 for r in rows))
    ck("sym_kl non-negative", all(r["sym_kl"] >= -1e-4 for r in rows))

    # EnsembleTeacher emits a valid top-k
    teach = EnsembleTeacher([A, B], [0.5, 0.5], k=8, temp=2.0)
    ti, tp = teach.topk(idx)
    ck(
        "ensemble-teacher topk shapes",
        tuple(ti.shape) == (1, 15, 8) and tuple(tp.shape) == (1, 15, 8),
    )
    ck("ensemble-teacher probs normalize", torch.allclose(tp.sum(-1), torch.ones(1, 15), atol=1e-4))

    # distill the ensemble into one student for a few steps — loss finite, weights move
    before = next(A.parameters()).clone()
    state, scfg = distill_ensemble(
        [A, B], [0.5, 0.5], vars(ca), ids, dev, steps=3, T=64, lr=3e-3, k=8, temp=2.0, log_every=0
    )
    ck(
        "distilled student state has params",
        len(state) > 0 and all(torch.isfinite(v).all() for v in state.values()),
    )

    # weight-merge wrappers (soup/slerp) produce loadable ckpts
    pa = os.path.join(tmp, "a.pt")
    pb = os.path.join(tmp, "b.pt")
    torch.save({"model": A.state_dict(), "cfg": vars(ca)}, pa)
    torch.save({"model": B.state_dict(), "cfg": vars(cb)}, pb)
    made = build_merges(pa, pb, os.path.join(tmp, "merges"))
    ck("soup + slerp checkpoints produced", "soup" in made and "slerp" in made)
    sm = torch.load(made["soup"], map_location="cpu", weights_only=False)
    ck("merged ckpt is loadable {model,cfg}", "model" in sm and "cfg" in sm)

    # HITL round-trip
    hp = os.path.join(tmp, "hitl.jsonl")
    n, _ = write_hitl(rows, hp, 5)
    with open(hp, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]
    lines[0]["winner"] = "a"
    with open(hp, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")
    tally = apply_hitl(hp)
    ck("HITL write+apply round-trips", n > 0 and tally["a"] == 1)

    print(
        f"\nbakeoff selftest: {ok}/{ok} passed -- ensemble, NLL scoreboard, divergence map, "
        "ensemble-distill, merges, and HITL all green"
    )
    return True


def main():
    p = argparse.ArgumentParser(description="CHARKHA bakeoff -- 2 checkpoints in, 1 model out.")
    p.add_argument("--a", type=str, default=None, help="checkpoint A (e.g. the cloud run)")
    p.add_argument("--b", type=str, default=None, help="checkpoint B (e.g. the local run)")
    p.add_argument(
        "--shard", type=str, default=None, help="tokenized uint16 .bin or shard dir for NLL/distill"
    )
    p.add_argument(
        "--outdir", type=str, default="runs/bakeoff", help="where merges/distilled/report go"
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="run everything: ensemble + merges + distill + divergence",
    )
    p.add_argument(
        "--merge", action="store_true", help="also build + score soup/slerp(/ties) merges"
    )
    p.add_argument(
        "--distill",
        action="store_true",
        help="also distill the ensemble into one student + score it",
    )
    p.add_argument(
        "--base", type=str, default=None, help="base ckpt for TIES (delta merge); omit to skip ties"
    )
    p.add_argument("--density", type=float, default=0.2, help="TIES trim density")
    p.add_argument("--slerp-t", type=float, default=0.5, help="SLERP interpolation t (0=A, 1=B)")
    p.add_argument(
        "--weight-a", type=float, default=0.5, help="ensemble weight on A (B gets 1-this)"
    )
    p.add_argument("--T", type=int, default=256, help="window length for NLL scoring")
    p.add_argument("--max-tokens", type=int, default=200000, help="cap tokens read from the shard")
    p.add_argument(
        "--gen-tokens", type=int, default=24, help="tokens to generate per divergence probe"
    )
    p.add_argument("--distill-steps", type=int, default=200, help="ensemble-distillation steps")
    p.add_argument(
        "--hitl", type=int, default=0, help="write top-N ambiguous probes for human review"
    )
    p.add_argument(
        "--apply-hitl", type=str, default=None, help="read a filled HITL JSONL and tally verdicts"
    )
    p.add_argument(
        "--toy", action="store_true", help="two fresh byte-level toy models (smoke, no ckpts)"
    )
    p.add_argument(
        "--digit-split", action="store_true", help="digit-split tokenizer (match training)"
    )
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        return 0 if selftest() else 1
    if args.apply_hitl:
        tally = apply_hitl(args.apply_hitl)
        print(f"[hitl] verdicts: {tally}")
        if tally["a"] or tally["b"]:
            print(
                f"  human-judged edge: {'A' if tally['a'] > tally['b'] else 'B' if tally['b'] > tally['a'] else 'tie'}"
            )
        return 0
    if not args.toy and not (args.a and args.b):
        p.error("need --a and --b checkpoints (or --toy for a smoke run, or --selftest)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_bakeoff(args, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
