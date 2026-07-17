#!/usr/bin/env python3
"""Proof-Carry: deterministic claim ledgers for tiny-model answers.

Murmur and Ply try to improve the model's internal computation. Proof-Carry improves
the surrounding contract: answers are decomposed into typed claims, each claim gets a
machine-checkable verdict where possible, and unsupported factual claims can be removed
before the response is stored or reused.

This is intentionally not a truth oracle. It proves only narrow classes:
  * arithmetic equalities through a safe calculator;
  * personal/world claims against CHARKHA's structured world model facts;
  * retrieved factual claims against the passages already injected into the prompt.
Everything else is either non-factual text or an unsupported claim. The value is not
"the model is smarter"; the value is that unverified claims stop contaminating memory,
replay, and user-facing output when strict mode is enabled.

Usage:
    python src/proofcarry.py --selftest

"""

from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import re
import sys
from dataclasses import asdict, dataclass


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
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
    "its",
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
    "my",
    "me",
    "we",
    "our",
    "can",
    "will",
    "would",
    "should",
}
_FACT_VERBS = re.compile(r"\b(is|are|was|were|has|have|equals|contains|uses|lives?|works?)\b", re.I)
_UNCERTAIN_RE = re.compile(
    r"\b(i do not know|i don't know|not enough information|do not have enough information|cannot verify|unsure)\b",
    re.I,
)
_NUMERIC_RE = re.compile(
    r"(?<![\w.])([0-9][0-9\s+\-*/().%]{1,80}[+\-*/%][0-9\s+\-*/().%]{1,80})"
    r"\s*(?:=|is|equals)\s*([-+]?\d+(?:\.\d+)?)"
)
_WORLD_RE = re.compile(
    r"\b(?:my|your|the user(?:\'s)?)\s+([a-z][a-z0-9 _-]{1,40}?)\s+"
    r"(?:is|are|=)\s+([^.;!?]{1,120})",
    re.I,
)


@dataclass
class Claim:
    kind: str
    text: str
    verdict: str = "unchecked"  # supported | contradicted | unsupported | nonclaim
    score: float = 0.0
    evidence: str = ""
    source: str = ""
    payload: dict | None = None


def safe_calc(expr: str):
    """Evaluate arithmetic with no names, calls, attributes, comprehensions, or indexing."""

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported arithmetic: {ast.dump(node)}")

    return ev(ast.parse(expr.strip(), mode="eval"))


def _terms(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in _STOP]


def _anchors(text: str):
    """Terms that should be present in evidence: numbers and proper-looking words."""
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    caps = [w.lower() for w in re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b", text)]
    return set(nums + [c for c in caps if c not in _STOP])


def _sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip(" \t-") for p in parts if p.strip(" \t-")]


def _norm_object(text: str):
    return " ".join(_terms(text))


def _world_predicate(slot: str):
    slot = re.sub(r"\s+", " ", slot.strip().lower())
    if slot in {"location", "city"}:
        return "location"
    if slot in {"workplace", "employer"}:
        return "workplace"
    return "has_" + slot.replace(" ", "_").replace("-", "_")


def extract_claims(answer: str):
    """Typed, deterministic claim extraction. Non-factual sentences are omitted."""
    claims = []
    seen = set()

    def add(claim: Claim):
        key = (claim.kind, re.sub(r"\s+", " ", claim.text.lower()))
        if key not in seen:
            seen.add(key)
            claims.append(claim)

    for sent in _sentences(answer):
        if _UNCERTAIN_RE.search(sent):
            continue
        numeric_hits = list(_NUMERIC_RE.finditer(sent))
        for m in numeric_hits:
            add(
                Claim(
                    "arithmetic",
                    m.group(0).strip(),
                    payload={"expr": m.group(1).strip(), "expected": m.group(2).strip()},
                )
            )

        world_hits = list(_WORLD_RE.finditer(sent))
        for m in world_hits:
            slot = m.group(1).strip()
            obj = m.group(2).strip()
            add(
                Claim(
                    "world",
                    sent,
                    payload={"subject": "user", "predicate": _world_predicate(slot), "object": obj},
                )
            )

        if numeric_hits or world_hits:
            continue
        terms = _terms(sent)
        if len(terms) >= 3 and (_FACT_VERBS.search(sent) or _anchors(sent)):
            add(Claim("retrieval", sent, payload={"terms": terms}))
    return claims[:24]


def _verify_arithmetic(claim: Claim):
    expr = claim.payload.get("expr", "") if claim.payload else ""
    want = float(claim.payload.get("expected")) if claim.payload else math.nan
    try:
        got = float(safe_calc(expr.replace("%", "/100")))
    except Exception as e:
        claim.verdict = "unsupported"
        claim.evidence = f"calculator error: {e}"
        return claim
    tol = max(1e-6, abs(want) * 1e-4)
    ok = abs(got - want) <= tol
    claim.verdict = "supported" if ok else "contradicted"
    claim.score = 1.0 if ok else 0.0
    claim.evidence = f"{expr} -> {got:g}"
    claim.source = "calculator"
    return claim


def _verify_world(claim: Claim, world_facts):
    payload = claim.payload or {}
    subj = payload.get("subject", "")
    pred = payload.get("predicate", "")
    want = _norm_object(payload.get("object", ""))
    best = None
    for f in world_facts or []:
        if f.get("subject") != subj or f.get("predicate") != pred:
            continue
        got = _norm_object(str(f.get("object", "")))
        overlap = len(set(want.split()) & set(got.split())) / max(1, len(set(want.split())))
        row = (overlap, f, got)
        if best is None or row[0] > best[0]:
            best = row
    if best and best[0] >= 0.66:
        claim.verdict = "supported"
        claim.score = min(1.0, 0.5 + 0.5 * best[0])
        claim.source = best[1].get("source", "world")
        claim.evidence = f"world: {subj}/{pred}/{best[1].get('object', '')}"
    elif best:
        claim.verdict = "contradicted"
        claim.score = 0.0
        claim.source = best[1].get("source", "world")
        claim.evidence = f"world has {subj}/{pred}/{best[1].get('object', '')}"
    else:
        claim.verdict = "unsupported"
        claim.score = 0.0
        claim.evidence = f"no world fact for {subj}/{pred}"
    return claim


def _passage_rows(passages):
    rows = []
    for p in passages or []:
        text = p.get("text", "") if isinstance(p, dict) else str(p)
        source = p.get("source", "") if isinstance(p, dict) else ""
        rows.append((text, source, set(_terms(text))))
    return rows


def _verify_retrieval(claim: Claim, passages):
    cterms = set(_terms(claim.text))
    anchors = _anchors(claim.text)
    if not cterms:
        claim.verdict = "nonclaim"
        claim.score = 1.0
        return claim
    best = (0.0, "", "")
    for text, source, pterms in _passage_rows(passages):
        if anchors and not anchors.issubset(pterms | set(re.findall(r"[-+]?\d+(?:\.\d+)?", text))):
            continue
        denom = max(1, min(len(cterms), 10))
        score = len(cterms & pterms) / denom
        if score > best[0]:
            best = (score, text, source)
    if best[0] >= 0.62:
        claim.verdict = "supported"
        claim.score = min(1.0, best[0])
        claim.source = best[2]
        claim.evidence = best[1][:220]
    else:
        claim.verdict = "unsupported"
        claim.score = best[0]
        claim.source = best[2]
        claim.evidence = best[1][:220] if best[1] else "no retrieved passage matched anchors"
    return claim


def verify_claim(claim: Claim, passages=None, world_facts=None):
    if claim.kind == "arithmetic":
        return _verify_arithmetic(claim)
    if claim.kind == "world":
        return _verify_world(claim, world_facts or [])
    return _verify_retrieval(claim, passages or [])


def _sentence_supported(sentence: str, verified):
    mine = [c for c in verified if c.text == sentence or c.text in sentence]
    return not mine or all(c.verdict == "supported" for c in mine)


def filter_answer(answer: str, verified, *, strict: bool = True):
    """Drop unsupported factual sentences in strict mode; keep nonclaims."""
    if not strict:
        return answer
    kept, dropped = [], 0
    for sent in _sentences(answer):
        if _sentence_supported(sent, verified):
            kept.append(sent)
        else:
            dropped += 1
    out = " ".join(kept).strip()
    if dropped:
        note = f"I omitted {dropped} unsupported claim" + ("s." if dropped != 1 else ".")
        out = (out + " " + note).strip() if out else note
    return out or "I cannot verify the factual claims in that answer."


def proof_carry(answer: str, *, passages=None, world_facts=None, strict: bool = False):
    claims = [
        verify_claim(c, passages=passages, world_facts=world_facts) for c in extract_claims(answer)
    ]
    supported = sum(1 for c in claims if c.verdict == "supported")
    contradicted = sum(1 for c in claims if c.verdict == "contradicted")
    unsupported = sum(1 for c in claims if c.verdict == "unsupported")
    checked = supported + contradicted + unsupported
    score = supported / checked if checked else 1.0
    accepted = contradicted == 0 and unsupported == 0
    filtered = filter_answer(answer, claims, strict=strict)
    return {
        "accepted": accepted,
        "score": score,
        "supported": supported,
        "unsupported": unsupported,
        "contradicted": contradicted,
        "checked": checked,
        "filtered_answer": filtered,
        "claims": [asdict(c) for c in claims],
    }


def replay_record(prompt: str, answer: str, report: dict):
    """Return a verified replay artifact or None if the answer is not fully proved."""
    if not report.get("accepted"):
        return None
    proof_lines = []
    for c in report.get("claims", []):
        if c.get("verdict") == "supported":
            proof_lines.append(
                f"- {c.get('kind')}: {c.get('text')} :: {c.get('source')} :: {c.get('evidence')}"
            )
    return {
        "prompt": prompt,
        "answer": answer,
        "proof": proof_lines,
        "text": "Q: "
        + prompt
        + "\nA: "
        + answer
        + ("\nPROOF:\n" + "\n".join(proof_lines) if proof_lines else ""),
    }


def _selftest():
    checks = 0

    r = proof_carry("12 * 13 = 156.")
    assert r["accepted"] and r["supported"] == 1 and r["claims"][0]["source"] == "calculator"
    checks += 1

    r = proof_carry("12 * 13 = 155.", strict=True)
    assert not r["accepted"] and r["contradicted"] == 1
    assert (
        "omitted" in r["filtered_answer"].lower() or "cannot verify" in r["filtered_answer"].lower()
    )
    checks += 1

    passages = [{"text": "Paris is the capital of France and sits on the Seine.", "source": "doc1"}]
    r = proof_carry("Paris is the capital of France.", passages=passages)
    assert r["accepted"] and r["supported"] == 1 and r["claims"][0]["source"] == "doc1"
    checks += 1

    r = proof_carry("Berlin is the capital of France.", passages=passages, strict=True)
    assert not r["accepted"] and r["unsupported"] == 1
    assert "Berlin" not in r["filtered_answer"]
    checks += 1

    facts = [
        {"subject": "user", "predicate": "has_gpu", "object": "RTX 4060 Ti", "source": "world"}
    ]
    r = proof_carry("Your GPU is an RTX 4060 Ti.", world_facts=facts)
    assert r["accepted"] and r["supported"] == 1
    r2 = proof_carry("Your GPU is an RTX 5090.", world_facts=facts)
    assert not r2["accepted"] and r2["contradicted"] == 1
    checks += 2

    rr = replay_record("What is 12*13?", "12 * 13 = 156.", r)
    assert rr is not None and "PROOF:" in rr["text"]
    assert replay_record("x", "Berlin is the capital of France.", r2) is None
    checks += 2

    clean = proof_carry("I do not have enough information to answer.", strict=True)
    assert clean["accepted"] and clean["checked"] == 0
    checks += 1

    print(f"[selftest] proofcarry.py: all checks passed ({checks} groups)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", type=str, default=None, help="proof-carry this answer and print JSON")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(_selftest())
    if a.json is not None:
        print(json.dumps(proof_carry(a.json), indent=2))
        sys.exit(0)
    ap.print_help()
