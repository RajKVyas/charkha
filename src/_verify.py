"""CHARKHA verification — proof contracts, arithmetic, tool resolution."""

import re
import ast
import operator
import math
import os
import torch

_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def safe_calc(expr: str):
    """Evaluate an arithmetic expression with no names, calls, or attribute access."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)}")

    return ev(ast.parse(expr.strip(), mode="eval"))


def reason_engine(expr):
    """Structured reasoning: 'from:to' format. Returns intermediate step."""
    parts = expr.split(":", 1)
    if len(parts) == 2:
        return f"({parts[0].strip()} → {parts[1].strip()})"
    return f"[reasoning: {expr}]"


def verify_engine(expr):
    """Self-consistency check on a sub-claim. Passthrough for now."""
    return f"[verified: {expr}]"


def causal_engine(expr):
    """Tiny causal scratchpad.

    Format: "a=2; b=a+3; if a=10". Assignments are evaluated left to right; an optional
    `if x=y` intervention overwrites a variable and recomputes downstream assignments.
    This is not a theorem prover, but it gives the model a deterministic intervention
    surface for simple variable dependencies.
    """
    parts = [p.strip() for p in re.split(r"[;\n]+", expr) if p.strip()]
    assigns, intervention = [], None
    for p in parts:
        if p.lower().startswith("if "):
            intervention = p[3:].strip()
        elif "=" in p:
            k, v = p.split("=", 1)
            assigns.append((k.strip(), v.strip()))
    env = {}

    def eval_expr(s):
        for k, v in sorted(env.items(), key=lambda kv: -len(kv[0])):
            s = re.sub(rf"\b{re.escape(k)}\b", str(v), s)
        return safe_calc(s)

    name_ok = re.compile(r"^[A-Za-z_]\w*$")
    deps = {}
    for k, v in assigns:
        if name_ok.match(k):
            deps[k] = {
                kk for kk, _vv in assigns if kk != k and re.search(rf"\b{re.escape(kk)}\b", v)
            }
            env[k] = eval_expr(v)
    base = dict(env)
    if intervention and "=" in intervention:
        k, v = intervention.split("=", 1)
        k = k.strip()
        if name_ok.match(k):
            env[k] = eval_expr(v)
            changed = {k}
            for kk, vv in assigns:
                if kk == k:
                    continue
                if deps.get(kk, set()) & changed:
                    env[kk] = eval_expr(vv)
                    changed.add(kk)
    return f"base={base}; after={env}"


TOOLS = {
    "calc": safe_calc,
    "reason": reason_engine,
    "verify": verify_engine,
    "causal": causal_engine,
}
TOOL_RE = re.compile(r"\[\[(\w+):\s*(.*?)\]\]", re.S)


def resolve_tools(text: str):
    """Replace every [[tool: arg]] with `arg = result`. Returns (new_text, calls_made)."""
    calls = []

    def repl(m):
        name, arg = m.group(1).lower(), m.group(2).strip()
        try:
            out = TOOLS[name](arg) if name in TOOLS else f"unknown tool {name!r}"
        except Exception as e:  # a bad tool call must not crash a turn
            out = f"error: {e}"
        calls.append((name, arg, out))
        return f"{arg} = {out}"

    return TOOL_RE.sub(repl, text), calls


# --------------------------------------------------------------------------
# <thinking> scratchpad - latent reasoning is generated, then stripped from the reply.
# --------------------------------------------------------------------------

THINK_RE = re.compile(r"<thinking>(.*?)</thinking>", re.S)


def split_thinking(text: str):
    """Return (thinking, answer): scratchpad text and the user-visible remainder."""
    thoughts = "\n".join(m.strip() for m in THINK_RE.findall(text))
    answer = THINK_RE.sub("", text).strip()
    return thoughts, answer


# --------------------------------------------------------------------------
# Grounding - a net has no clock or place; we hand it functional awareness cheaply.
# --------------------------------------------------------------------------


def system_preamble(date: str, location: str, cutoff: str, concise: bool = False) -> str:
    base = (
        "[SYSTEM]\n"
        f"date: {date}\n"
        f"location: {location}\n"
        f"knowledge-cutoff: {cutoff}\n"
        "You are CHARKHA, a small open language model. Reason privately inside "
        "<thinking>...</thinking>, then give a short, direct answer. For exact arithmetic "
        "call a tool, e.g. [[calc: 12*9]]. If a fact is past your cutoff or you are unsure, "
        "say so and decompose the problem instead of guessing."
    )
    if concise:
        base += (
            "\nCONCISE MODE: The best answer is the shortest one that works. "
            "One word > one sentence > one paragraph. "
            "Skip explanation unless asked. Skip hedging unless uncertain. "
            "Never restate the question. Never pad for length."
        )
    base += "\n[/SYSTEM]"
    return base


# --------------------------------------------------------------------------
# External continuity memory (SQLite, stdlib) - survives across sessions.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Tokenizers - byte-level for the hermetic toy path, checkpoint-selected for real checkpoints.
# --------------------------------------------------------------------------


class ByteTokenizer:
    vocab_size = 256

    def encode(self, s: str):
        return list(s.encode("utf-8"))

    def decode(self, ids):
        return bytes(b & 0xFF for b in ids).decode("utf-8", errors="replace")


def load_neox():
    """Compatibility alias for loading the repository's included tokenizer."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo_root, "charkha_tokenizer.json")
    if not os.path.exists(path):
        return ByteTokenizer()
    from tokenizers import Tokenizer

    return _HFTokenizerCompat(Tokenizer.from_file(path))


class _HFTokenizerCompat:
    """Wraps a raw tokenizers.Tokenizer (local .json, e.g. charkha_tokenizer_v5.json) in the
    AutoTokenizer-shaped interface used by the serving path."""

    def __init__(self, tok):
        self._tok = tok
        self.vocab_size = tok.get_vocab_size()

    def encode(self, s):
        return self._tok.encode(s).ids

    def decode(self, ids):
        return self._tok.decode(ids)

    def token_to_id(self, t):
        return self._tok.token_to_id(t)


def load_tokenizer_for(name):
    """Load whatever tokenizer the checkpoint's cfg.tokenizer_name says produced its training
    shards (a local tokenizer.json path or an HF hub name) — NOT a hardcoded one, so inference
    always matches training. None/missing falls back to the tokenizer included in this repository."""
    if not name:
        return load_neox()
    # resolve a local tokenizer.json: as given, or relative to the repo root (so a shard's
    # tokenizer path works no matter what cwd serve.py is launched from — e.g. from src/).
    cand = name
    if not os.path.exists(cand):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base = os.path.basename(name)
        # repo root, then the tokenizer artifact store — checkpoints record absolute
        # paths that break when a tokenizer file is later archived into artifacts/.
        for alt in (
            os.path.join(repo_root, base),
            os.path.join(repo_root, "artifacts", "tokenizers", base),
        ):
            if os.path.exists(alt):
                cand = alt
                break
    if os.path.exists(cand):
        from tokenizers import Tokenizer

        return _HFTokenizerCompat(Tokenizer.from_file(cand))
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


class DigitSplitTokenizer:
    """Wraps any tokenizer so encode() digit-splits the input and decode() re-glues digit runs —
    so a model trained with dataprep --digit-split sees the SAME one-token-per-digit stream at
    inference. Without this, a digit-split-trained model would be fed fused multi-digit tokens it
    never saw in training (silent train/serve mismatch, especially on math). Off unless requested."""

    def __init__(self, base):
        from dataprep import split_digits, join_digits

        self._base, self._split, self._join = base, split_digits, join_digits
        self.vocab_size = getattr(base, "vocab_size", None)

    def encode(self, s):
        return self._base.encode(self._split(s))

    def decode(self, ids):
        return self._join(self._base.decode(ids))


# --------------------------------------------------------------------------
# Effort policy - the confidence head is a signal, so spend more loops when unsure.
# Pure function (testable without a model): escalate until confident or capped.
# --------------------------------------------------------------------------


def escalate(conf_fn, base: int, max_effort: int, threshold: float):
    """conf_fn(effort)->confidence in [0,1]. Returns (effort, confidence, trace)."""
    effort = base
    trace = []
    while True:
        c = conf_fn(effort)
        trace.append((effort, c))
        if c >= threshold or effort >= max_effort:
            return effort, c, trace
        effort = min(effort * 2, max_effort)


def convergence_confidence(signal) -> float:
    """Map the recurrent-core extrapolation-error signal (charkha A2, model._last_convergence,
    a (B,T) tensor) to a convergence confidence in (0,1]: small trajectory error => the answer
    stopped moving => high confidence. 1.0 when the signal is absent (don't penalize). Pure +
    testable. NOTE: the 1/(1+error) mapping is uncalibrated (error scales with state norm); it is
    only ever used as a *signal* fused by min() behind a default-off flag, never as a hard gate."""
    if signal is None:
        return 1.0
    mean_err = float(signal.float().mean().item())
    return 1.0 / (1.0 + max(mean_err, 0.0))


def sngp_confidence(var) -> float:
    """Map the SNGP epistemic variance (charkha model._last_sngp_var, a (B,T) tensor) to a
    distance-aware confidence in (0,1]: small variance => input near the training manifold =>
    high confidence; large variance (OOD) => low. 1.0 when absent. Same caveat as
    convergence_confidence — an uncalibrated *signal* fused by min() behind a default-off flag,
    never a hard gate; calibrate the scale on a val set before trusting the magnitude."""
    if var is None:
        return 1.0
    mean_var = float(var.float().mean().item())
    return 1.0 / (1.0 + max(mean_var, 0.0))


@torch.no_grad()
def mean_confidence(
    model, ids, device, effort, start=0, fuse_convergence=False, fuse_sngp=False
) -> float:
    """Mean of the model's per-token calibration head over `ids` at a given effort. `ids` must
    include the prompt context; `start` marks the first generated token to score. When
    fuse_convergence is set, AND it (via min) with the A2 convergence confidence so the effort
    dial only stops early when the model BOTH thinks it's right AND its trajectory has settled.
    When fuse_sngp is set, also AND in the SNGP distance-aware confidence (epistemic / OOD)."""
    if not ids:
        return 1.0
    max_len = model.cfg.max_seq_len
    offset = max(0, len(ids) - max_len)
    x = torch.tensor([ids[-max_len:]], device=device)
    _logits, conf = model(x, r=effort)
    local_start = max(0, start - offset)
    span = conf[:, local_start:] if local_start < conf.size(1) else conf[:, -1:]
    c = float(span.mean().item())
    if fuse_convergence:
        c = min(c, convergence_confidence(getattr(model, "_last_convergence", None)))
    if fuse_sngp:
        c = min(c, sngp_confidence(getattr(model, "_last_sngp_var", None)))
    return c


# --------------------------------------------------------------------------
# Abstention as a *generative behavior*, not a gate. After the effort dial is
# spent, if confidence is still below a *conformal* threshold (calibrated offline
# by pipeline.selective_threshold for a target error rate), we do NOT swap in a
# canned refusal - users hate a model that keeps replying "I'm not sure" verbatim.
# Instead we re-generate in an uncertainty-aware mode: a steering note conditions
# the model to answer in its OWN words using what it does know, name the specific
# gap, and suggest a way forward. The conformal gate chooses between "confident
# answer" and "hedged answer" - both are real model outputs. Escalate-then-abstain:
# compute is spent FIRST; the hedged mode is the fallback, not the first move.
# (The model must LEARN this skill weight-wise - see ARCHITECTURE: uncertainty-aware
# training data + RLVR reward for calibrated hedging + the confidence head. This
# wrapper only elicits the behavior; training makes it good.)
# --------------------------------------------------------------------------

# Injected into the prompt to elicit the learned uncertainty-aware voice.
UNCERTAINTY_NOTE = (
    "[NOTE: your internal confidence here is low. Do not fabricate. Give a genuinely "
    "helpful reply in your own words: say what you do know, name exactly what you are "
    "unsure about and why, and suggest how to pin it down. Be natural and specific, "
    "never formulaic.]"
)


def honest_refusal(conf: float, tau: float) -> str:
    """Last-resort fallback ONLY if uncertainty-aware regeneration yields nothing
    (e.g. an untrained model). The trained model speaks for itself instead."""
    return (
        "I'm not sure enough to answer that confidently "
        f"(confidence {conf:.2f}). Here's what would help: more context, or a "
        f"tool I can use to verify my answer."
    )


_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "you",
    "your",
}


def _terms(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in _STOP]


def _char_ngrams(text: str, n: int = 4):
    s = re.sub(r"\s+", " ", text.lower()).strip()
    if len(s) <= n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _number_sig(text: str):
    return set(re.findall(r"[-+]?\d+(?:\.\d+)?", text))


def answer_similarity(a: str, b: str) -> float:
    """Cheap semantic proxy for council voting: word overlap + character n-gram overlap,
    with a hard penalty when both answers contain numbers but disagree about them."""
    wa, wb = _terms(a), _terms(b)
    ca, cb = _char_ngrams(a), _char_ngrams(b)
    sim = 0.55 * _jaccard(wa, wb) + 0.45 * _jaccard(ca, cb)
    na, nb = _number_sig(a), _number_sig(b)
    if na and nb and na != nb:
        sim *= 0.35
    return max(0.0, min(1.0, sim))


def repetition_health(text: str) -> float:
    """1 is healthy; 0 is degenerate looping. Small LMs often fail by repetition long before
    the confidence head notices, so council scoring treats this as an independent signal."""
    toks = _terms(text)
    if len(toks) < 8:
        return 1.0
    uniq = len(set(toks)) / max(1, len(toks))
    grams = list(zip(toks, toks[1:], toks[2:]))
    gram_uniq = len(set(grams)) / max(1, len(grams))
    longest = run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    return max(0.0, min(1.0, 0.45 * uniq + 0.45 * gram_uniq + 0.10 * (1.0 / longest)))


def grounding_score(answer: str, passages) -> float:
    """Lexical support against retrieved passages. This is not truth, but it catches the common
    failure mode where an answer ignores the context it was handed."""
    if not passages:
        return 1.0
    ans = set(_terms(answer))
    if not ans:
        return 0.5
    ctx = set()
    for p in passages:
        ctx.update(_terms(p.get("text", "")))
    return max(0.0, min(1.0, len(ans & ctx) / max(1, min(len(ans), 24))))


def tool_health(calls) -> float:
    if not calls:
        return 1.0
    return sum(0.0 if "error:" in str(out).lower() else 1.0 for _n, _a, out in calls) / len(calls)


def numeric_claim_health(text: str) -> float:
    """Tool-integrated verifier for simple numeric claims.

    This is the deterministic half of T1-style verification: if a candidate states
    `12 * 7 = 83`, the council should not need the small model to remember arithmetic.
    It should execute the arithmetic and penalize the candidate before self-scoring.
    """
    checks = []
    pat = re.compile(
        r"(?<![\w.])([0-9][0-9\s+\-*/().%]{1,80}[+\-*/%][0-9\s+\-*/().%]{1,80})"
        r"\s*(?:=|is|equals)\s*([-+]?\d+(?:\.\d+)?)"
    )
    for m in pat.finditer(text):
        expr = m.group(1).strip()
        want = float(m.group(2))
        try:
            got = float(safe_calc(expr.replace("%", "/100")))
        except Exception:
            continue
        tol = max(1e-6, abs(want) * 1e-4)
        checks.append(abs(got - want) <= tol)
    if not checks:
        return 1.0
    return sum(1.0 if ok else 0.0 for ok in checks) / len(checks)


def extract_claims(text: str):
    """Atomic-ish claims for claim-level council scoring."""
    claims = []
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        s = s.strip(" -\t\r\n")
        if not s or len(s) < 4:
            continue
        if len(s) > 240:
            parts = [p.strip() for p in re.split(r";|, and ", s) if len(p.strip()) >= 4]
            claims.extend(parts[:4])
        else:
            claims.append(s)
    # Include explicit numeric equalities as claims even if the sentence splitter missed them.
    for m in re.finditer(
        r"\d[0-9\s+\-*/().%]*[+\-*/%][0-9\s+\-*/().%]*\s*(?:=|is|equals)\s*[-+]?\d+(?:\.\d+)?", text
    ):
        claims.append(m.group(0).strip())
    out, seen = [], set()
    for c in claims:
        key = re.sub(r"\s+", " ", c.lower())
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out[:12]


def proof_contract(text: str, passages=None) -> dict:
    """Executable proof obligations for a candidate answer."""
    claims = extract_claims(text)
    numeric = numeric_claim_health(text)
    citation_needed = bool(passages)
    citation_score = grounding_score(text, passages) if citation_needed else 1.0
    file_checks = []
    for m in re.finditer(r"(?:file|path)\s*[:=]\s*([^\s,;]+)", text, re.I):
        file_checks.append(os.path.exists(m.group(1).strip('"')))
    file_score = 1.0 if not file_checks else sum(file_checks) / len(file_checks)
    return {
        "claims": claims,
        "numeric": numeric,
        "citation": citation_score,
        "file": file_score,
        "score": min(numeric, citation_score, file_score),
    }


def mini_verifier_ensemble(text: str, passages=None) -> dict:
    """Cheap specialist verifiers: fluency/form, citation grounding, code-ish sanity."""
    toks = _terms(text)
    fluency = repetition_health(text)
    citation = grounding_score(text, passages)
    code_blocks = len(re.findall(r"```|def |class |import ", text))
    code_form = 1.0 if code_blocks == 0 else (0.7 + 0.3 * ("```" in text or "\n" in text))
    length = 1.0 if 0 < len(toks) < 220 else 0.65
    score = 0.35 * fluency + 0.30 * citation + 0.20 * code_form + 0.15 * length
    return {
        "fluency": fluency,
        "citation": citation,
        "code_form": code_form,
        "length": length,
        "score": score,
    }


def claim_consensus(cands):
    all_claims = []
    for ci, c in enumerate(cands):
        c["claims"] = extract_claims(c.get("answer", ""))
        for cl in c["claims"]:
            all_claims.append((ci, cl))
    reports = []
    for ci, cl in all_claims:
        sims = []
        for cj, other in all_claims:
            if ci == cj:
                continue
            sims.append(answer_similarity(cl, other))
        support = max(sims) if sims else 1.0
        reports.append(
            {"candidate": ci, "claim": cl, "support": support, "numeric": numeric_claim_health(cl)}
        )
    for i, c in enumerate(cands):
        mine = [r for r in reports if r["candidate"] == i]
        c["claim_support"] = (
            sum(r["support"] * r["numeric"] for r in mine) / len(mine) if mine else 1.0
        )
    return reports


def classify_failure(
    answer: str, council=None, retrieval_q=None, world_conflicts=None, calls=None
) -> str:
    if calls and any("error:" in str(out).lower() for _n, _a, out in calls):
        return "tool_error"
    if numeric_claim_health(answer) < 1.0:
        return "arithmetic"
    if world_conflicts:
        return "contradiction"
    if council and council.get("entropy", 0.0) > 0.45:
        return "ambiguity"
    if retrieval_q is not None and retrieval_q < 0.4:
        return "missing_knowledge"
    if repetition_health(answer) < 0.45:
        return "loop_degeneration"
    if council and council.get("consensus", 1.0) < 0.4:
        return "unsupported_claim"
    return "ok"


def council_rank(cands, passages=None, retrieval_q=None):
    """Rank generated candidates by a small-model-oriented outer loop:
    calibrated token confidence, agreement with other attempts, retrieved-context support,
    tool validity, and anti-loop scoring. Returns (best, report)."""
    if not cands:
        return None, {"size": 0, "consensus": 0.0, "entropy": 0.0, "clusters": []}
    n = len(cands)
    sims = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sims[i][j] = sims[j][i] = answer_similarity(cands[i]["answer"], cands[j]["answer"])
    claims = claim_consensus(cands)
    for i, c in enumerate(cands):
        consensus = sum(sims[i][j] for j in range(n) if j != i) / max(1, n - 1)
        ground = grounding_score(c["answer"], passages)
        if retrieval_q is not None:
            ground = min(ground, retrieval_q)
        tools = tool_health(c.get("calls", []))
        numeric = numeric_claim_health(c["answer"])
        proof = proof_contract(c["answer"], passages)
        ver = mini_verifier_ensemble(c["answer"], passages)
        repeat = repetition_health(c["answer"])
        base = float(c.get("base_conf", 1.0))
        claim_sup = c.get("claim_support", 1.0)
        c.update(
            {
                "consensus": consensus,
                "grounding": ground,
                "tool_health": tools,
                "numeric_health": numeric,
                "repeat_health": repeat,
                "proof": proof,
                "verifiers": ver,
            }
        )
        c["score"] = (
            0.27 * base
            + 0.20 * consensus
            + 0.13 * claim_sup
            + 0.13 * ground
            + 0.07 * repeat
            + 0.05 * tools
            + 0.05 * numeric
            + 0.05 * proof["score"]
            + 0.05 * ver["score"]
        )
    # Greedy semantic clusters for an ACSE-like entropy signal without an embedding dependency.
    order = sorted(range(n), key=lambda i: cands[i]["score"], reverse=True)
    clusters = []
    for i in order:
        placed = False
        for cl in clusters:
            if max(sims[i][j] for j in cl) >= 0.72:
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    weights = [sum(cands[i]["score"] for i in cl) for cl in clusters]
    total = sum(weights) or 1.0
    probs = [w / total for w in weights]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    entropy = entropy / math.log(max(2, len(clusters))) if len(clusters) > 1 else 0.0
    best = max(cands, key=lambda c: c["score"])
    market = [
        {
            "i": i,
            "score": round(cands[i]["score"], 4),
            "base_conf": round(float(cands[i].get("base_conf", 0.0)), 4),
            "claim_support": round(float(cands[i].get("claim_support", 1.0)), 4),
            "proof": round(float(cands[i].get("proof", {}).get("score", 1.0)), 4),
        }
        for i in order
    ]
    report = {
        "size": n,
        "consensus": best["consensus"],
        "entropy": entropy,
        "claim_reports": claims[:40],
        "market": market,
        "clusters": [
            {
                "weight": round(probs[k], 4),
                "size": len(cl),
                "representative": cands[cl[0]]["answer"][:180],
            }
            for k, cl in enumerate(clusters)
        ],
    }
    return best, report


__all__ = [
    "ByteTokenizer",
    "UNCERTAINTY_NOTE",
    "answer_similarity",
    "causal_engine",
    "claim_consensus",
    "classify_failure",
    "convergence_confidence",
    "council_rank",
    "escalate",
    "extract_claims",
    "grounding_score",
    "honest_refusal",
    "load_neox",
    "load_tokenizer_for",
    "mean_confidence",
    "mini_verifier_ensemble",
    "numeric_claim_health",
    "proof_contract",
    "reason_engine",
    "repetition_health",
    "resolve_tools",
    "safe_calc",
    "sngp_confidence",
    "split_thinking",
    "system_preamble",
    "tool_health",
    "verify_engine",
]
