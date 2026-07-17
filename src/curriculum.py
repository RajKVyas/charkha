"""CHARKHA self-curating curriculum — the model's own uncertainty picks its next lessons.

Most curriculum learning hard-codes an order ("English, then math, then science"). That has two
problems: (1) it forgets earlier domains once it moves on (catastrophic forgetting), and (2) a human
guesses the order instead of measuring where the model is actually weak. This module does neither.

THE LOOP (closed-loop, self-paced — closer to mastery-learning than to a fixed syllabus):
  1. PROBE   — run the model over a small diagnostic set per domain and read its CONFIDENCE HEAD
               (model.forward returns per-token conf in [0,1]). Low mean confidence on a domain's
               probes == the model knows it's shaky there. This is a signal most LMs don't expose;
               CHARKHA trains a calibration head precisely so we can.
  2. SELECT  — rank domains by weakness (1-conf). Focus the weakest few. Domains that cross a mastery
               threshold are de-prioritised but REVISITED occasionally (spaced repetition), so nothing
               is ever "finished" and left to rot.
  3. TARGET  — ask the frontier teacher (pipeline._chat_completion -> gpt-oss / deepseek) to generate
               GRADED lessons + practice aimed exactly at the weak domains, with a per-domain trace
               budget proportional to weakness (weaker -> more data).
  4. WRITE   — append traces and re-tokenize into a uint16 shard dir, drop-in for `train.py --data`.
  Train a bit on the new shard, then re-probe: the model's confidence is the syllabus designer, every
  round, forever. Across rounds we persist a per-domain weakness history so you can watch it level up.

HONEST CAVEATS (don't trust the signal blindly):
  * "confidence low here" can mean "needs more data here" OR "hit a capacity/architecture limit" —
    a 0.42B model will plateau on some domains no matter how much you feed it. Watch the history: if a
    domain's weakness stops falling across rounds despite spending budget, that's a capacity wall, not
    a data gap. Spending more teacher tokens there is waste.
  * the probe + teacher calls cost latency and (for metered teachers) money — keep probe sets small.

DESIGN (mirrors selfteach.py): the orchestration core is PURE python (model injected as probe_fn,
teacher injected as generate_fn) so --selftest exercises the whole loop hermetically with mocks, no
torch / no network. torch + the API are lazy-imported only in the real driver.

  python src/curriculum.py --selftest                      # pure, CPU, no GPU/network
  python src/curriculum.py --run --ckpt runs/charkha/ckpt.pt --out data/curriculum \\
      --api-base http://localhost:8000/v1 --api-model "ggml-org/gpt-oss-120b-GGUF" --rounds 1
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time

try:
    sys.stdout.reconfigure(
        encoding="utf-8"
    )  # win32 stdout may be cp1252; model bytes would crash print
except Exception:
    pass

DEFAULT_TEACHER_SYSTEM = (
    "You are a careful teacher producing training data for a small (~0.4B) student language model. "
    "Write a clear, self-contained lesson: explain the concept from first principles, then give a "
    "fully worked example, then one practice problem WITH its worked solution. Reason step by step. "
    "When a calculation is needed, show it inline as [[calc: <expression>]]. Prefer correctness and "
    "clarity over length; note uncertainty honestly rather than guessing."
)

# --------------------------------------------------------------------------
# The domain map. Each domain carries (a) a few cheap diagnostic PROBES whose mean confidence scores
# the model's grip on that domain, (b) a LESSON template the teacher fills to generate targeted data,
# and (c) SUBTOPICS to diversify what the teacher produces. Add domains freely — the loop is generic.
# --------------------------------------------------------------------------
CURRICULUM = [
    {
        "domain": "arithmetic",
        "probes": ["What is 47 + 88?", "Compute 13 * 12.", "What is 144 divided by 12?"],
        "subtopics": [
            "addition and subtraction",
            "multiplication",
            "division and remainders",
            "fractions",
            "percentages",
            "order of operations",
        ],
        "lesson": "Teach a {level} lesson on {subtopic} in arithmetic, with a worked example and a "
        "practice problem (show the solution).",
    },
    {
        "domain": "algebra",
        "probes": [
            "Solve for x: 2x + 5 = 17.",
            "Expand (x + 3)(x - 2).",
            "If y = 3x and x = 4, what is y?",
        ],
        "subtopics": [
            "solving linear equations",
            "factoring",
            "systems of equations",
            "inequalities",
            "quadratics",
            "functions",
        ],
        "lesson": "Teach a {level} lesson on {subtopic} in algebra, with a worked example and a "
        "practice problem (show the solution).",
    },
    {
        "domain": "science",
        "probes": [
            "Why does ice float on water?",
            "What is photosynthesis?",
            "What causes the seasons on Earth?",
        ],
        "subtopics": [
            "basic physics",
            "chemistry fundamentals",
            "biology and cells",
            "astronomy",
            "energy and forces",
            "the scientific method",
        ],
        "lesson": "Write a {level} explanation teaching {subtopic}, with a concrete real-world example "
        "and a check-your-understanding question (with the answer).",
    },
    {
        "domain": "history",
        "probes": [
            "What were the main causes of World War I?",
            "Who was Genghis Khan and why is he significant?",
            "What was the Industrial Revolution?",
        ],
        "subtopics": [
            "ancient civilizations",
            "major wars and their causes",
            "key historical figures",
            "political revolutions",
            "economic and social change",
            "timelines of events",
        ],
        "lesson": "Write a {level} history lesson on {subtopic}, with key dates, causes and effects, "
        "and one short comprehension question (with the answer).",
    },
    {
        "domain": "language",
        "probes": [
            "What is the difference between 'their', 'there', and 'they're'?",
            "Identify the subject and verb in: 'The dog ran quickly.'",
            "What is a metaphor? Give an example.",
        ],
        "subtopics": [
            "grammar and parts of speech",
            "punctuation",
            "vocabulary and word meaning",
            "sentence structure",
            "figurative language",
            "reading comprehension",
        ],
        "lesson": "Teach a {level} English-language lesson on {subtopic}, with clear examples and a "
        "short exercise (with the answer).",
    },
    {
        "domain": "coding",
        "probes": [
            "Write a Python function that returns the factorial of n.",
            "What does a for-loop do in Python?",
            "How do you reverse a list in Python?",
        ],
        "subtopics": [
            "variables and types",
            "loops and conditionals",
            "functions",
            "lists and dictionaries",
            "recursion",
            "debugging common errors",
        ],
        "lesson": "Write a {level} Python programming lesson on {subtopic}, with a commented code "
        "example and a small exercise (with the worked solution).",
    },
    {
        "domain": "reasoning",
        "probes": [
            "If all roses are flowers and some flowers fade quickly, do all roses fade quickly?",
            "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
            "What comes next: 2, 4, 8, 16, ...?",
        ],
        "subtopics": [
            "logical deduction",
            "common reasoning traps",
            "pattern recognition",
            "word problems",
            "cause and effect",
            "evaluating arguments",
        ],
        "lesson": "Write a {level} lesson on {subtopic} in logical reasoning, with a worked example "
        "that shows the step-by-step thinking, then a practice puzzle (with the solution).",
    },
    {
        "domain": "world_knowledge",
        "probes": [
            "What is the capital of Japan?",
            "How many continents are there?",
            "What is the largest ocean on Earth?",
        ],
        "subtopics": [
            "geography",
            "countries and capitals",
            "famous landmarks",
            "general science facts",
            "culture and society",
            "everyday practical knowledge",
        ],
        "lesson": "Write a {level} informative lesson covering key facts about {subtopic}, framed as "
        "clear question-and-answer pairs a student should know.",
    },
]

# --------------------------------------------------------------------------
# Pure orchestration core — every function below is torch-free and network-free, so the whole
# probe->select->target loop is unit-tested with a mock probe_fn (returns confidences) and a mock
# generate_fn (returns teacher text). This is the selftest's contract.
# --------------------------------------------------------------------------


def domain_weakness(domains, probe_fn):
    """Score each domain's weakness from the model's confidence on its probes.
    probe_fn(list[str]) -> list[float] (mean per-prompt confidence in [0,1]).
    Returns a list of {domain, conf, weakness}, sorted weakest-first."""
    out = []
    for spec in domains:
        confs = probe_fn(spec["probes"])
        mean_conf = sum(confs) / max(len(confs), 1)
        out.append({"domain": spec["domain"], "conf": mean_conf, "weakness": 1.0 - mean_conf})
    out.sort(key=lambda r: r["weakness"], reverse=True)
    return out


def select_focus(report, k, mastered=0.8):
    """Pick the k domains to work on this round: weakest-first among the not-yet-mastered, then (if
    fewer than k remain unmastered) backfill with the MOST-confident domains for a spaced-repetition
    revisit — so mastery doesn't mean abandonment. Returns a list of domain names (len <= k)."""
    unmastered = [r["domain"] for r in report if r["conf"] < mastered]  # report is weakest-first
    focus = unmastered[:k]
    if len(focus) < k:
        for r in sorted(report, key=lambda x: x["conf"], reverse=True):  # most-confident first
            if r["domain"] not in focus:
                focus.append(r["domain"])
            if len(focus) >= k:
                break
    return focus[:k]


def allocate_budget(report, total, focus):
    """Split a round's trace budget across the focus domains, proportional to weakness (weaker gets
    more), with a floor of 1 each. Returns {domain: n_traces}."""
    wmap = {r["domain"]: max(r["weakness"], 1e-3) for r in report}
    fw = {d: wmap.get(d, 1e-3) for d in focus}
    s = sum(fw.values()) or 1.0
    alloc = {d: max(1, int(round(total * fw[d] / s))) for d in focus}
    return alloc


def build_lesson_prompts(spec, n, rng):
    """n teacher prompts for a domain, varying subtopic and difficulty level for diversity."""
    subs = spec.get("subtopics") or ["the fundamentals"]
    levels = ["beginner", "intermediate", "advanced"]
    return [
        spec["lesson"].format(subtopic=rng.choice(subs), level=rng.choice(levels)) for _ in range(n)
    ]


def curriculum_round(
    domains,
    probe_fn,
    generate_fn,
    *,
    focus_k=3,
    traces_per_round=60,
    mastered=0.8,
    system=None,
    rng=None,
):
    """One probe -> select -> target -> generate round (PURE orchestration; I/O injected).
      probe_fn(list[str]) -> list[float]      (model confidence — the syllabus signal)
      generate_fn(prompt, system) -> str      (teacher lesson text)
    Returns {report, focus, alloc, traces:[{domain,prompt,response}], skipped}."""
    rng = rng or random.Random(0)
    spec_by = {s["domain"]: s for s in domains}
    report = domain_weakness(domains, probe_fn)
    focus = select_focus(report, focus_k, mastered)
    alloc = allocate_budget(report, traces_per_round, focus)
    traces, skipped = [], 0
    for d in focus:
        for lp in build_lesson_prompts(spec_by[d], alloc[d], rng):
            resp = generate_fn(lp, system or DEFAULT_TEACHER_SYSTEM)
            if resp and resp.strip():
                traces.append({"domain": d, "prompt": lp, "response": resp.strip()})
            else:
                skipped += 1
    return {"report": report, "focus": focus, "alloc": alloc, "traces": traces, "skipped": skipped}


def format_report(report, focus):
    """One-line-per-domain confidence bar, focus domains starred — the 'feel it learning' view."""
    lines = []
    for r in report:
        bar = "█" * int(round(r["conf"] * 20))
        star = " *FOCUS*" if r["domain"] in focus else ""
        lines.append(f"  {r['domain']:<16} conf {r['conf']:.2f} |{bar:<20}|{star}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Real wiring — torch + API lazy-imported. The probe reads CHARKHA's confidence head; the teacher is
# the frontier API used everywhere else (pipeline._chat_completion); output is a uint16 shard dir.
# --------------------------------------------------------------------------


def make_probe_fn(model, tok, device, max_len=256):
    """Wrap a Charkha checkpoint as probe_fn(prompts)->[mean confidence]. model(x) returns
    (logits, conf) at inference (targets=None); we mean-pool the per-token confidence head."""
    import torch

    @torch.no_grad()
    def probe(prompts):
        model.eval()
        confs = []
        for p in prompts:
            ids = (tok.encode(p) or [0])[:max_len]
            x = torch.tensor([ids], device=device)
            _logits, conf = model(x)  # conf: (B, T) per-token P(top-1 correct)
            confs.append(float(conf.mean().item()))
        return confs

    return probe


def _retokenize(out_dir, tokenizer):
    """Re-tokenize the whole master traces.txt into out_dir as uint16 shards. Returns exact tokens."""
    from dataprep import run_pipeline

    sys.path.insert(0, os.path.dirname(__file__))
    from pipeline import _text_file_to_docs

    text_path = os.path.join(out_dir, "traces.txt")
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
    return int(idx["total_tokens"])


def run_curriculum(
    ckpt,
    out_dir,
    *,
    api_base,
    api_model,
    api_key="",
    rounds=1,
    focus_k=3,
    traces_per_round=60,
    max_new=1024,
    temperature=0.7,
    tokenizer="charkha_tokenizer.json",
    device=None,
    toy=False,
):
    """Real self-curating curriculum against a checkpoint (GPU/user-run). Each round: probe the
    model's confidence per domain, focus the weakest, have the teacher generate targeted graded
    lessons, append + re-tokenize into out_dir (drop-in `train.py --data out_dir`), and persist a
    weakness history so you can watch domains level up across rounds. Training itself is a separate
    step (run train.py on out_dir between rounds) — this produces the targeted data, by design."""
    import torch

    sys.path.insert(0, os.path.dirname(__file__))
    from serve import load_model
    from pipeline import _chat_completion

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, _cfg = load_model(ckpt, device, toy=toy)
    probe_fn = make_probe_fn(model, tok, device)

    def generate_fn(prompt, system):
        try:
            return _chat_completion(
                api_base,
                api_key,
                api_model,
                prompt,
                system,
                max_tokens=max_new,
                temperature=temperature,
            )
        except Exception as e:
            print(f"  [teacher error] {type(e).__name__}: {str(e)[:160]}", flush=True)
            return ""

    os.makedirs(out_dir, exist_ok=True)
    text_path = os.path.join(out_dir, "traces.txt")
    if not os.path.exists(text_path):
        with open(text_path, "w", encoding="utf-8") as f:
            f.write("# CHARKHA self-curating curriculum traces\n")
            f.write(f"# Teacher: {api_model} via {api_base}\n")
            f.write(
                "# PROVENANCE: frontier-KD, confidence-targeted — personal / non-distributable\n\n"
            )

    hist_path = os.path.join(out_dir, "curriculum_history.json")
    history = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    rng = random.Random(len(history))  # vary subtopic draws across resumed rounds

    for r in range(rounds):
        res = curriculum_round(
            CURRICULUM,
            probe_fn,
            generate_fn,
            focus_k=focus_k,
            traces_per_round=traces_per_round,
            rng=rng,
        )
        with open(text_path, "a", encoding="utf-8") as f:
            for t in res["traces"]:
                f.write(f"\nuser: {t['prompt']}\nassistant: {t['response']}\n")
        tok_count = _retokenize(out_dir, tokenizer)
        print(f"\n[curriculum] round {len(history)} — confidence map:")
        print(format_report(res["report"], res["focus"]))
        print(
            f"[curriculum] focus={res['focus']} alloc={res['alloc']} "
            f"-> {len(res['traces'])} traces ({res['skipped']} skipped), corpus={tok_count:,} tok"
        )
        history.append(
            {
                "round": len(history),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "report": res["report"],
                "focus": res["focus"],
                "alloc": res["alloc"],
                "new_traces": len(res["traces"]),
                "total_tokens": tok_count,
            }
        )
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

    print(
        f"\n[curriculum] done. Train on the targeted data:  "
        f"python src/train.py --data {out_dir} --out runs/charkha --resume ..."
    )
    print("[curriculum] then re-run this to re-probe and target the next weak spots.")
    return history


# --------------------------------------------------------------------------
def _selftest():
    ok = 0

    def ck(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        assert cond, name
        ok += 1

    domains = CURRICULUM

    # 1. every domain is well-formed (probes, lesson template with both placeholders, subtopics)
    ck("all domains have probes", all(d["probes"] for d in domains))
    ck(
        "lesson templates accept level+subtopic",
        all("{level}" in d["lesson"] and "{subtopic}" in d["lesson"] for d in domains),
    )
    ck("all domains have subtopics", all(d.get("subtopics") for d in domains))

    # 2. weakness scoring: a probe_fn that returns fixed per-domain confidence must rank weakest-first
    fake_conf = {
        "arithmetic": 0.9,
        "algebra": 0.2,
        "science": 0.5,
        "history": 0.1,
        "language": 0.7,
        "coding": 0.3,
        "reasoning": 0.6,
        "world_knowledge": 0.95,
    }
    cur_domain = {"d": None}

    def probe_fixed(prompts):
        # identify domain by matching the probe list (selftest harness)
        for d in domains:
            if d["probes"] == prompts:
                return [fake_conf[d["domain"]]] * len(prompts)
        return [0.5] * len(prompts)

    report = domain_weakness(domains, probe_fixed)
    ck(
        "report sorted weakest-first",
        all(report[i]["weakness"] >= report[i + 1]["weakness"] for i in range(len(report) - 1)),
    )
    ck(
        "weakness = 1 - conf",
        abs(report[0]["weakness"] - (1 - fake_conf[report[0]["domain"]])) < 1e-9,
    )
    ck("weakest domain is history (conf 0.1)", report[0]["domain"] == "history")

    # 3. focus selection: weakest unmastered first
    focus = select_focus(report, 3, mastered=0.8)
    ck("focus picks 3", len(focus) == 3)
    ck("focus is the 3 weakest unmastered", set(focus) == {"history", "algebra", "coding"})

    # 4. spaced-repetition backfill: if almost everything is mastered, revisit the most-confident
    high = [{"domain": d["domain"], "conf": 0.95, "weakness": 0.05} for d in domains]
    high[0]["conf"], high[0]["weakness"] = 0.1, 0.9  # one genuinely weak
    f2 = select_focus(high, 3, mastered=0.8)
    ck("weak domain always in focus", high[0]["domain"] in f2)
    ck("backfill fills to k even when all-but-one mastered", len(f2) == 3)

    # 5. budget allocation: weaker domains get >= budget, totals are sane, floor of 1
    alloc = allocate_budget(report, 60, focus)
    ck(
        "every focus domain funded",
        set(alloc) == set(focus) and all(v >= 1 for v in alloc.values()),
    )
    ck(
        "weaker gets more than stronger within focus", alloc["history"] >= alloc["coding"]
    )  # history (0.9) weaker than coding (0.7)
    ck("budget roughly conserved", 50 <= sum(alloc.values()) <= 70)
    ck(
        "tiny budget still gives each focus >=1",
        all(v >= 1 for v in allocate_budget(report, 1, focus).values()),
    )

    # 6. lesson prompt generation: count, filled placeholders, diversity
    rng = random.Random(0)
    prompts = build_lesson_prompts(domains[0], 12, rng)
    ck("build_lesson_prompts count", len(prompts) == 12)
    ck("no unfilled placeholders", all("{" not in p for p in prompts))
    ck("lesson prompts vary (subtopic/level draws)", len(set(prompts)) > 1)

    # 7. full round with mock teacher: traces only for focus domains, budget respected, empties skipped
    call_log = []

    def gen_mock(prompt, system):
        call_log.append(prompt)
        return (
            "" if len(call_log) % 7 == 0 else f"LESSON: {prompt[:30]} ... [[calc: 1+1]] answer 2."
        )

    res = curriculum_round(
        domains, probe_fixed, gen_mock, focus_k=3, traces_per_round=30, rng=random.Random(1)
    )
    ck("round focuses 3 domains", len(res["focus"]) == 3)
    ck("traces tagged with a focus domain", all(t["domain"] in res["focus"] for t in res["traces"]))
    ck(
        "empty teacher replies are skipped, not written",
        res["skipped"] > 0 and all(t["response"] for t in res["traces"]),
    )
    ck("trace count = calls - skipped", len(res["traces"]) == len(call_log) - res["skipped"])
    ck(
        "weakest domain got the most teacher calls",
        res["alloc"][res["focus"][0]] == max(res["alloc"].values()),
    )

    # 8. report formatting renders one line per domain and stars the focus
    txt = format_report(res["report"], res["focus"])
    ck("report has a line per domain", txt.count("\n") == len(domains) - 1)
    ck("focus domains are starred", txt.count("*FOCUS*") == len(res["focus"]))

    # 9. an ALL-mastered model: nothing unmastered, focus still backfills (spaced revisit keeps it sharp)
    allhigh = [{"domain": d["domain"], "conf": 0.99, "weakness": 0.01} for d in domains]
    fa = select_focus(allhigh, 2, mastered=0.8)
    ck("mastered model still revisits (no abandonment)", len(fa) == 2)

    # 10. determinism: same seed -> same prompts (resume-safe)
    a = build_lesson_prompts(domains[2], 8, random.Random(42))
    b = build_lesson_prompts(domains[2], 8, random.Random(42))
    ck("lesson generation is seed-deterministic", a == b)

    print(
        f"\ncurriculum selftest: {ok}/{ok} passed — confidence-probed, weakness-ranked, "
        "budget-allocated, spaced-repetition focus, teacher-targeted, drop-in shards."
    )
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CHARKHA self-curating curriculum (confidence-targeted KD)"
    )
    p.add_argument("--selftest", action="store_true", help="pure unit tests (no torch/network)")
    p.add_argument(
        "--run", action="store_true", help="run the real loop against a checkpoint + teacher"
    )
    p.add_argument("--ckpt", type=str, help="checkpoint to probe (omit with --toy)")
    p.add_argument("--out", type=str, default="data/curriculum", help="output shard dir")
    p.add_argument("--api-base", type=str, help="OpenAI-compatible teacher base URL")
    p.add_argument("--api-model", type=str, help="teacher model name to request")
    p.add_argument("--api-key", type=str, default="", help="teacher API key (if metered)")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--focus-k", type=int, default=3, help="weakest domains to target per round")
    p.add_argument("--traces-per-round", type=int, default=60)
    p.add_argument("--max-new", type=int, default=1024, help="teacher max_tokens per lesson")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--tokenizer", type=str, default="charkha_tokenizer.json")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--toy", action="store_true", help="probe a fresh toy model (no ckpt) — smoke test"
    )
    a = p.parse_args()
    if a.selftest:
        _selftest()
    elif a.run:
        if not (a.api_base and a.api_model):
            p.error("--run requires --api-base and --api-model")
        if not a.ckpt and not a.toy:
            p.error("--run requires --ckpt (or --toy for a smoke test)")
        run_curriculum(
            a.ckpt,
            a.out,
            api_base=a.api_base,
            api_model=a.api_model,
            api_key=a.api_key,
            rounds=a.rounds,
            focus_k=a.focus_k,
            traces_per_round=a.traces_per_round,
            max_new=a.max_new,
            temperature=a.temperature,
            tokenizer=a.tokenizer,
            device=a.device,
            toy=a.toy,
        )
    else:
        p.print_help()
