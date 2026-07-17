"""CHARKHA deduplication — MinHash LSH + exact/near-dedup pipeline."""

import hashlib
import random
import re


class MinHashLSH:
    def __init__(self, num_perm=128, bands=32, seed=1):
        self.num_perm, self.bands = num_perm, bands
        self.rows = num_perm // bands
        self.MOD = (1 << 61) - 1
        rnd = random.Random(seed)
        self.a = [rnd.randrange(1, self.MOD) for _ in range(num_perm)]
        self.b = [rnd.randrange(0, self.MOD) for _ in range(num_perm)]
        self.buckets = [dict() for _ in range(bands)]

    def _shingles(self, text, k=5):
        w = text.split()
        if len(w) < k:
            return {hash(text) & 0xFFFFFFFF}
        return {hash(" ".join(w[i : i + k])) & 0xFFFFFFFF for i in range(len(w) - k + 1)}

    def _sig(self, text):
        sh = self._shingles(text)
        return [min((a * x + b) % self.MOD for x in sh) for a, b in zip(self.a, self.b)]

    def is_dup_then_add(self, text):
        sig = self._sig(text)
        keys = [tuple(sig[i * self.rows : (i + 1) * self.rows]) for i in range(self.bands)]
        if any(k in self.buckets[i] for i, k in enumerate(keys)):
            return True
        for i, k in enumerate(keys):
            self.buckets[i][k] = True
        return False


class Deduper:
    """mode='near' = exact sha1 + MinHash-LSH near-dedup (default; correct but pure-Python
    MinHash is ~88% of pipeline CPU AND its LSH buckets grow unbounded — at tens of millions
    of docs that OOMs a 16GB box). mode='exact' = sha1 only (cheap, bounded, catches verbatim
    repeats — the right choice for already-deduped curated corpora at billions of tokens).
    mode='none' = no dedup."""

    def __init__(self, mode="near"):
        self.mode = mode
        self.seen = set()
        self.lsh = MinHashLSH() if mode == "near" else None

    def check(self, text):
        if self.mode == "none":
            return None
        h = hashlib.sha1(text.encode("utf-8")).digest()
        if h in self.seen:
            return "exact"
        self.seen.add(h)
        if self.mode == "near":
            return "near" if self.lsh.is_dup_then_add(text) else None
        return None


# --------------------------------------------------------------------------
# Stage 5 - PII scrub
# --------------------------------------------------------------------------
_PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<EMAIL>"),
    (
        re.compile(
            r"\b(?:\+?\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3}[ .-]?\d{3,4}[ .-]?\d{0,4}\b"
        ),
        "<PHONE>",
    ),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<IP>"),
    (re.compile(r"\b(?:sk|pk|ghp|xox[baprs])[-_][A-Za-z0-9]{16,}\b"), "<KEY>"),
    (re.compile(r"\b[A-Fa-f0-9]{32,}\b"), "<HASH>"),
]

__all__ = ["Deduper", "MinHashLSH"]
