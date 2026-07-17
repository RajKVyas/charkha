"""Tiny structured world model for CHARKHA serve.

This is not a video/physics world model. It is the first local substrate for a personal
assistant world state: structured, timestamped beliefs extracted from conversation and
retrieved back into the prompt. Unlike plain memory recall, facts are keyed by
(subject, predicate) so contradictions can be surfaced instead of silently blending.

The extractor is deliberately conservative and stdlib-only. The model can later learn to
emit explicit [[fact: subject | predicate | object]] tool calls, but this file gives the
serving stack a safe deterministic base today.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from dataclasses import dataclass


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
    "my",
    "me",
}


@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    confidence: float = 0.7
    evidence: str = ""
    source: str = "conversation"
    ts: float = 0.0
    trust: float = 1.0


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _terms(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in _STOP]


def extract_facts(text: str, *, source: str = "conversation", ts: float | None = None):
    """Conservative personal/factual triples from natural language.

    Patterns intentionally favor first-person stable facts and simple definitions. They
    avoid questions, conditionals, and long objects because bad world-state writes are
    more harmful than missed writes.
    """
    now = time.time() if ts is None else float(ts)
    raw = re.sub(r"\s+", " ", text.strip())
    if not raw or raw.endswith("?"):
        return []
    facts = []

    def add(subj, pred, obj, conf=0.72):
        subj, pred, obj = _norm(subj), _norm(pred), _norm(obj.rstrip(".! "))
        if not subj or not pred or not obj:
            return
        if len(obj) > 120 or len(subj) > 80 or len(pred) > 48:
            return
        facts.append(Fact(subj, pred, obj, conf, raw[:500], source, now))

    # "my GPU is a 4060 Ti", "my favorite editor is vim"
    for m in re.finditer(
        r"\bmy ([a-z][a-z0-9 _-]{1,40}?) (?:is|are|=) ([^.;!?]{1,120})", raw, re.I
    ):
        add("user", "has_" + _norm(m.group(1)).replace(" ", "_"), m.group(2), 0.78)

    # "I live in Boston", "I work at OpenAI", "I use WSL"
    for verb, pred in (
        ("live in", "location"),
        ("work at", "workplace"),
        ("work for", "workplace"),
        ("use", "uses"),
        ("like", "likes"),
        ("prefer", "prefers"),
    ):
        pat = rf"\bI {verb} ([^.;!?]{{1,120}})"
        for m in re.finditer(pat, raw, re.I):
            add("user", pred, m.group(1), 0.74)

    # "CHARKHA is a recurrent model", "Paris is the capital of France".
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9 _-]{1,60}) (?:is|are) ([^.;!?]{1,120})", raw):
        subj = m.group(1)
        if subj.lower() not in {"i", "you"}:
            add(subj, "is", m.group(2), 0.62)

    # De-duplicate within one utterance.
    seen, out = set(), []
    for f in facts:
        key = (f.subject, f.predicate, f.object)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out


def extract_corrections(text: str, *, source: str = "correction", ts: float | None = None):
    """Parse direct user corrections into replacement facts.

    Examples:
      "no, my GPU is an RTX 5090"
      "correction: my city is Boston"
    """
    t = re.sub(r"\s+", " ", text.strip())
    if not re.match(r"(?i)^(no[, ]|correction:|actually\b|update:)", t):
        return []
    t = re.sub(r"(?i)^(no[, ]+|correction:\s*|actually[, ]*|update:\s*)", "", t)
    return extract_facts(t, source=source, ts=ts)


class WorldModel:
    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS facts ("
            "subject TEXT, predicate TEXT, object TEXT, confidence REAL, "
            "source TEXT, evidence TEXT, ts REAL, hits INTEGER DEFAULT 1, trust REAL DEFAULT 1.0, "
            "PRIMARY KEY(subject, predicate, object))"
        )
        self._migrate()
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS contradictions ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, predicate TEXT, "
            "old_object TEXT, new_object TEXT, source TEXT, evidence TEXT, ts REAL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS observations ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, source TEXT, ts REAL)"
        )
        self.db.commit()

    def _migrate(self):
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(facts)").fetchall()}
        if "trust" not in cols:
            self.db.execute("ALTER TABLE facts ADD COLUMN trust REAL DEFAULT 1.0")

    def close(self):
        self.db.close()

    def remember(
        self,
        text: str,
        *,
        source: str = "conversation",
        ts: float | None = None,
        trust: float | None = None,
    ):
        ts = time.time() if ts is None else float(ts)
        self.db.execute(
            "INSERT INTO observations(text, source, ts) VALUES (?, ?, ?)", (text, source, ts)
        )
        facts = extract_corrections(text, source="correction", ts=ts) or extract_facts(
            text, source=source, ts=ts
        )
        contradictions = []
        trust = (
            self.source_trust(source if source != "conversation" else "user")
            if trust is None
            else float(trust)
        )
        for f in facts:
            f.trust = trust
            found = self.contradictions(f)
            contradictions.extend(found)
            for old in found:
                self.db.execute(
                    "INSERT INTO contradictions(subject, predicate, old_object, new_object, source, evidence, ts) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (f.subject, f.predicate, old["object"], f.object, source, f.evidence, ts),
                )
            self.db.execute(
                "INSERT INTO facts(subject, predicate, object, confidence, source, evidence, ts, hits, trust) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?) "
                "ON CONFLICT(subject, predicate, object) DO UPDATE SET "
                "confidence=max(confidence, excluded.confidence), "
                "evidence=excluded.evidence, ts=excluded.ts, hits=hits+1, trust=max(trust, excluded.trust)",
                (
                    f.subject,
                    f.predicate,
                    f.object,
                    f.confidence,
                    f.source,
                    f.evidence,
                    f.ts,
                    f.trust,
                ),
            )
        self.db.commit()
        return facts, contradictions

    def contradictions(self, fact: Fact):
        cur = self.db.execute(
            "SELECT subject, predicate, object, confidence, source, evidence, ts, hits, trust "
            "FROM facts WHERE subject=? AND predicate=? AND object<>?",
            (fact.subject, fact.predicate, fact.object),
        )
        return [
            dict(
                subject=r[0],
                predicate=r[1],
                object=r[2],
                confidence=r[3],
                source=r[4],
                evidence=r[5],
                ts=r[6],
                hits=r[7],
                trust=r[8],
            )
            for r in cur.fetchall()
        ]

    @staticmethod
    def source_trust(source: str) -> float:
        s = (source or "").lower()
        if s in {"user", "correction"}:
            return 1.0
        if s in {"retrieval", "document", "personal"}:
            return 0.82
        if s in {"assistant", "conversation"}:
            return 0.55
        return 0.65

    def contradiction_docs(self, since_ts: float = 0.0):
        rows = self.db.execute(
            "SELECT subject, predicate, old_object, new_object, source, evidence, ts "
            "FROM contradictions WHERE ts>=? ORDER BY ts",
            (since_ts,),
        ).fetchall()
        return [
            (
                "CONTRADICTION REPLAY: for {s}/{p}, replace {old!r} with {new!r}.\nEvidence: {ev}\n"
            ).format(s=r[0], p=r[1], old=r[2], new=r[3], ev=r[5])
            for r in rows
        ]

    def counterfactual(self, subject: str, predicate: str, new_object: str, query: str = ""):
        """Return a prompt-ready hypothetical state without mutating the ledger."""
        subject, predicate, new_object = _norm(subject), _norm(predicate), _norm(new_object)
        prior = self.db.execute(
            "SELECT object, confidence, trust FROM facts WHERE subject=? AND predicate=?",
            (subject, predicate),
        ).fetchall()
        return {
            "hypothesis": {"subject": subject, "predicate": predicate, "object": new_object},
            "replaces": [{"object": r[0], "confidence": r[1], "trust": r[2]} for r in prior],
            "context": format_world_context(self.retrieve(query or f"{subject} {predicate}", k=8)),
        }

    def retrieve(self, query: str, k: int = 8):
        q = set(_terms(query))
        rows = self.db.execute(
            "SELECT subject, predicate, object, confidence, source, evidence, ts, hits, trust FROM facts"
        ).fetchall()
        scored = []
        for r in rows:
            text = " ".join(str(x) for x in r[:3])
            terms = set(_terms(text))
            overlap = len(q & terms) / max(1, len(q | terms)) if q or terms else 0.0
            recency = 1.0 / (1.0 + max(0.0, time.time() - float(r[6])) / (86400.0 * 90.0))
            score = (
                0.54 * overlap
                + 0.18 * float(r[3])
                + 0.12 * float(r[8])
                + 0.08 * min(1.0, r[7] / 5.0)
                + 0.08 * recency
            )
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            dict(
                score=s,
                subject=r[0],
                predicate=r[1],
                object=r[2],
                confidence=r[3],
                source=r[4],
                evidence=r[5],
                ts=r[6],
                hits=r[7],
                trust=r[8],
            )
            for s, r in scored[:k]
        ]


def format_world_context(facts):
    if not facts:
        return ""
    lines = ["[WORLD MODEL: structured beliefs; use as context, not as guaranteed truth]"]
    for f in facts:
        lines.append(
            f"- {f['subject']} / {f['predicate']} / {f['object']} "
            f"(conf={float(f['confidence']):.2f}, trust={float(f.get('trust', 1.0)):.2f}, "
            f"hits={int(f['hits'])})"
        )
    lines.append("[/WORLD MODEL]")
    return "\n".join(lines)


def _selftest():
    import tempfile

    wm = WorldModel(os.path.join(tempfile.mkdtemp(), "world.sqlite"))
    facts, contra = wm.remember("My GPU is a 4060 Ti. I live in Boston.", source="test", ts=1)
    assert len(facts) >= 2 and not contra
    hits = wm.retrieve("what gpu do I use?", k=3)
    assert hits and "4060" in hits[0]["object"]
    _facts, contra = wm.remember("No, my GPU is an RTX 5090.", source="user", ts=2)
    assert contra and "4060" in contra[0]["object"]
    assert wm.contradiction_docs()
    assert wm.counterfactual("user", "has_gpu", "RTX 6000")
    assert "WORLD MODEL" in format_world_context(hits)
    wm.close()
    print("SELFTEST PASS - world model stores, retrieves, and surfaces contradictions")


if __name__ == "__main__":
    _selftest()
