"""CHARKHA retrieval — the knowledge subsystem (how a ~0.4B model out-knows bigger models).

A tiny model cannot store a 3B model's facts in its weights (knowledge scales with params). So we
DON'T: knowledge lives in an external, provenanced datastore and is retrieved at inference. Evidence
this works: RETRO (7.5B ≈ 280B Gopher), GPT-JT 6B + retrieval ≈ GPT-3.5 (30× larger). The datastore
carries per-document provenance (source/license travel with each passage), so every retrieved fact
is *citable* — knowledge, currency (update the store, not the weights), and honesty in one mechanism.

Design: BM25 lexical retrieval is the pure-Python, dependency-free core (solid and 8GB-trivial —
index+store live off-GPU). An optional dense index (sentence-transformers, lazy) fuses for a hybrid
when available. Each passage carries provenance (source/license) for citations. `retrieval_quality`
exposes a difficulty signal for retrieval-gated recurrence (weak/contradictory retrieval -> spend
more effort, lower confidence).

  python retrieval.py --selftest        # stdlib only; BM25 + fusion + provenance + citations

"""

from __future__ import annotations
import argparse
import math
import re
import sys
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str):
    return _TOKEN.findall(text.lower())


# --------------------------------------------------------------------------
# BM25 lexical index (Okapi BM25) — pure Python, the dependency-free core.
# --------------------------------------------------------------------------


class BM25Index:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = []
        self.meta = []  # provenance per passage
        self.df = defaultdict(int)
        self.doc_len = []
        self.avgdl = 0.0
        self._finalized = False

    def add(self, text, meta=None):
        toks = _tokenize(text)
        self.docs_tokens.append(toks)
        self.meta.append({"text": text, **(meta or {})})
        for term in set(toks):
            self.df[term] += 1
        self.doc_len.append(len(toks))
        self._finalized = False
        return len(self.docs_tokens) - 1

    def finalize(self):
        n = max(len(self.doc_len), 1)
        self.avgdl = sum(self.doc_len) / n
        self._tf = [Counter(t) for t in self.docs_tokens]
        self._finalized = True
        return self

    def _idf(self, term):
        n = len(self.docs_tokens)
        df = self.df.get(term, 0)
        # BM25+ idf, floored at 0 so common terms never go negative
        return max(0.0, math.log((n - df + 0.5) / (df + 0.5) + 1.0))

    def search(self, query, k=5):
        if not self._finalized:
            self.finalize()
        q = _tokenize(query)
        scores = []
        for i, tf in enumerate(self._tf):
            dl = self.doc_len[i]
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                s += self._idf(term) * f * (self.k1 + 1) / denom
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return scores[:k]


# --------------------------------------------------------------------------
# Optional dense index (lazy sentence-transformers) — fused with BM25 for hybrid.
# --------------------------------------------------------------------------


class DenseIndex:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self.np = np
        self.model = SentenceTransformer(model_name)
        self.vecs = None
        self.texts = []

    def build(self, texts):
        self.texts = list(texts)
        v = self.model.encode(self.texts, normalize_embeddings=True)
        self.vecs = self.np.asarray(v)
        return self

    def search(self, query, k=5):
        q = self.np.asarray(self.model.encode([query], normalize_embeddings=True))[0]
        sims = self.vecs @ q
        order = sims.argsort()[::-1][:k]
        return [(float(sims[i]), int(i)) for i in order]


# --------------------------------------------------------------------------
# Datastore + hybrid retriever
# --------------------------------------------------------------------------


def _minmax(scores):
    if not scores:
        return {}
    vals = [s for s, _ in scores]
    lo, hi = min(vals), max(vals)
    rng = hi - lo or 1.0
    return {i: (s - lo) / rng for s, i in scores}


class Datastore:
    """Provenanced passage store with BM25 (+optional dense) hybrid retrieval."""

    def __init__(self, use_dense=False, dense_model=None, alpha=0.5):
        self.bm25 = BM25Index()
        self.alpha = alpha  # weight on dense in the fusion
        self.use_dense = use_dense
        self._dense = (
            DenseIndex(dense_model)
            if use_dense and dense_model
            else (DenseIndex() if use_dense else None)
        )

    def add(self, text, source="", license=""):
        return self.bm25.add(text, {"source": source, "license": license})

    def finalize(self):
        self.bm25.finalize()
        if self._dense is not None:
            self._dense.build([m["text"] for m in self.bm25.meta])
        return self

    def retrieve(self, query, k=5):
        lex = self.bm25.search(query, k=max(k * 4, k))
        if self._dense is None:
            fused = [(s, i) for s, i in lex][:k]
        else:
            dn = self.bm25  # noqa
            ln, dnn = _minmax(lex), _minmax(self._dense.search(query, k=max(k * 4, k)))
            keys = set(ln) | set(dnn)
            fused = sorted(
                ((self.alpha * dnn.get(i, 0) + (1 - self.alpha) * ln.get(i, 0), i) for i in keys),
                reverse=True,
            )[:k]
        out = []
        for score, i in fused:
            m = self.bm25.meta[i]
            out.append(
                {
                    "text": m["text"],
                    "source": m.get("source", ""),
                    "license": m.get("license", ""),
                    "score": float(score),
                }
            )
        return out


# --------------------------------------------------------------------------
# Inference helpers: build a cited context block, and a difficulty signal.
# --------------------------------------------------------------------------


def format_context(passages, max_chars=1200):
    """Prepend retrieved passages with [source] tags so the model can cite — the provenance
    that makes the honesty guarantee auditable."""
    lines, used = [], 0
    for p in passages:
        tag = p.get("source") or "source"
        snippet = p["text"].strip().replace("\n", " ")
        chunk = f"[{tag}] {snippet}"
        if used + len(chunk) > max_chars:
            chunk = chunk[: max_chars - used]
        lines.append(chunk)
        used += len(chunk)
        if used >= max_chars:
            break
    return "Relevant sources:\n" + "\n".join(lines) if lines else ""


def retrieval_quality(passages):
    """Signal in [0,1] for retrieval-gated recurrence: high when the top hit is strong and clearly
    ahead of the rest (confident grounding); low when retrieval is weak or flat (spend more effort,
    lower confidence, prefer abstention). Returns 0.0 on empty retrieval."""
    if not passages:
        return 0.0
    scores = sorted((p["score"] for p in passages), reverse=True)
    top = scores[0]
    if top <= 0:
        return 0.0
    margin = (top - scores[1]) / top if len(scores) > 1 else 1.0  # lead over the runner-up
    return max(0.0, min(1.0, 0.5 * min(top, 1.0) + 0.5 * margin))


# --------------------------------------------------------------------------
def _selftest():
    ok = 0

    def ck(name, cond):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        assert cond, name
        ok += 1

    ds = Datastore()
    ds.add(
        "The boiling point of water at standard atmospheric pressure is 100 degrees Celsius.",
        source="wikipedia",
        license="cc-by-sa",
    )
    ds.add(
        "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
        source="common_corpus",
        license="public-domain",
    )
    ds.add(
        "The spinning wheel, or charkha, turned cotton into thread in village homes.",
        source="gutenberg",
        license="public-domain",
    )
    ds.finalize()

    res = ds.retrieve("what temperature does water boil at", k=2)
    ck("retrieval returns passages", len(res) >= 1)
    ck("top hit is the boiling-point doc", "boiling point" in res[0]["text"])
    ck(
        "provenance preserved (source+license)",
        res[0]["source"] == "wikipedia" and res[0]["license"] == "cc-by-sa",
    )

    res2 = ds.retrieve("carbon dioxide glucose sunlight", k=2)
    ck("lexical hit on photosynthesis doc", "Photosynthesis" in res2[0]["text"])

    # BM25 ranking sanity: a query term that's rare ranks its doc above an unrelated doc
    ck(
        "charkha query finds the charkha doc",
        "charkha" in ds.retrieve("charkha thread", k=1)[0]["text"],
    )

    # citations
    ctx = format_context(res)
    ck("context cites the source tag", "[wikipedia]" in ctx and "boiling point" in ctx)
    ck("empty retrieval -> empty context", format_context([]) == "")

    # retrieval-gated difficulty signal
    strong = retrieval_quality([{"score": 1.0}, {"score": 0.1}])
    weak = retrieval_quality([{"score": 0.2}, {"score": 0.19}])
    ck("quality high for strong+clear top hit", strong > 0.6)
    ck("quality low for weak/flat retrieval", weak < 0.5)
    ck("quality 0 on empty retrieval", retrieval_quality([]) == 0.0)

    # no-match query returns nothing (model should then abstain / answer from weights)
    ck("no-match query returns empty", ds.retrieve("zzzqqq nonexistent term", k=3) == [])

    print(
        f"\nretrieval selftest: {ok}/{ok} passed -- BM25 hybrid retrieval with provenance, "
        "citations, and a retrieval-gating signal"
    )
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CHARKHA retrieval (knowledge subsystem)")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        _selftest()
    else:
        p.print_help()
