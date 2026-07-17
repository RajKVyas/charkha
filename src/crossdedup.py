"""
CHARKHA crossdedup - exact document-level cross-dedup of tokenized shard dirs.
=============================================================================
mv-0 and mv-1 are both MixtureVitae-v1, deduped *within* each run but never against
each other. This removes from a CANDIDATE dir every document whose exact token sequence
already appears in one or more REFERENCE dirs (or earlier in the candidate itself), and
writes a fresh deduped shard dir + index.json. Exact (blake2b of the token bytes) to stay
consistent with dataprep's `exact` (sha1) dedup - the mode the real corpus was built with;
near-dup/MinHash OOMs at billions of tokens and would be inconsistent with mv-0's own dedup.

Docs are recovered by splitting the concatenated uint16 stream on the EOS token
(gpt-neox '<|endoftext|>' = id 0). Doc boundaries do NOT align to 100M-token shard
boundaries, so each dir is read as ONE continuous stream with carry-over across shards.

Usage:
  python crossdedup.py --candidate data/new --ref data/reference --out data/new-dedup
  python crossdedup.py --selftest

Exit 0 on success.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import tempfile

import numpy as np

from dataprep import shard_format_for_vocab

EOS_DEFAULT = 0  # gpt-neox '<|endoftext|>' id in real shards
HASH_BYTES = 16  # blake2b digest size: 16B -> ~negligible collision risk at <1e9 docs


def _iter_docs(d, eos=EOS_DEFAULT):
    """Yield each document (uint16/uint32 array, INCLUDING its trailing EOS) by reading all of dir
    d's shards as one continuous stream and splitting on `eos`. Carries the partial doc at
    each shard boundary into the next shard (boundaries don't align to docs). dtype ('<u2' vs
    '<u4') is decided by d's own index.json vocab_size -- the single source of truth shared with
    dataprep.py's writer, so a >65535-vocab dir is read at its true element width."""
    idx = json.load(open(os.path.join(d, "index.json")))
    _, bytes_per_token = shard_format_for_vocab(idx.get("vocab_size"))
    dt = "<u2" if bytes_per_token == 2 else "<u4"
    carry = np.empty(0, dtype=dt)
    for s in idx["shards"]:
        a = np.fromfile(os.path.join(d, s["file"]), dtype=dt)
        if carry.size:
            a = np.concatenate([carry, a])
        zeros = np.flatnonzero(a == eos)
        prev = 0
        for z in zeros:
            yield a[prev : z + 1]
            prev = int(z) + 1
        carry = a[prev:].copy()
    if carry.size:  # trailing doc with no EOS (stream tail)
        yield carry


def _h(doc):
    return hashlib.blake2b(doc.tobytes(), digest_size=HASH_BYTES).digest()


class _Writer:
    """Re-shard the kept docs into 100M-token .bin files + index.json. dtype is decided by
    vocab via shard_format_for_vocab -- the same source of truth as dataprep.py's writer and
    this module's own _iter_docs read side -- so output bytes always match the vocab_size this
    same instance writes into index.json via close(). Without this, a >65535-vocab dir run
    through crossdedup/chain_dedup would have its token ids silently truncated mod 65536 while
    index.json still (wrongly) claimed the large vocab -- a corrupted, mismatched output."""

    def __init__(self, out, shard_tokens=100_000_000, vocab=None):
        os.makedirs(out, exist_ok=True)
        self.out, self.cap = out, shard_tokens
        _, bytes_per_token = shard_format_for_vocab(vocab)
        self.dtype = "<u2" if bytes_per_token == 2 else "<u4"
        self.buf, self.n, self.shards, self.i = [], 0, [], 0

    def add(self, doc):
        self.buf.append(doc)
        self.n += int(doc.size)
        if self.n >= self.cap:
            self._flush()

    def _flush(self):
        if not self.n:
            return
        arr = np.concatenate(self.buf) if len(self.buf) > 1 else self.buf[0]
        fn = f"shard_{self.i:05d}.bin"
        arr.astype(self.dtype).tofile(os.path.join(self.out, fn))
        self.shards.append({"file": fn, "tokens": int(arr.size)})
        self.i += 1
        self.buf, self.n = [], 0

    def close(self, vocab):
        self._flush()
        total = sum(s["tokens"] for s in self.shards)
        with open(os.path.join(self.out, "index.json"), "w") as f:
            json.dump(
                {"vocab_size": vocab, "total_tokens": total, "shards": self.shards}, f, indent=2
            )
        return total


def crossdedup(candidate, refs, out, eos=EOS_DEFAULT, quiet=False):
    vocab = json.load(open(os.path.join(candidate, "index.json")))["vocab_size"]
    seen = set()
    ref_docs = 0
    for rd in refs:
        for doc in _iter_docs(rd, eos):
            seen.add(_h(doc))
            ref_docs += 1
    if not quiet:
        print(
            f"  reference: {ref_docs:,} docs from {len(refs)} dir(s) -> {len(seen):,} unique hashes"
        )
    w = _Writer(out, vocab=vocab)
    cand = dup = kept = kept_tok = 0
    for doc in _iter_docs(candidate, eos):
        cand += 1
        h = _h(doc)
        if h in seen:
            dup += 1
            continue
        seen.add(h)  # also kills self-dups within the candidate
        w.add(doc)
        kept += 1
        kept_tok += int(doc.size)
    total = w.close(vocab)
    if not quiet:
        print(f"  candidate: {cand:,} docs | dup-of-ref-or-self: {dup:,} | KEPT: {kept:,}")
        print(f"  kept tokens: {kept_tok:,} ({kept_tok / 1e9:.3f}B)")
        print(f"  -> {out}  ({total:,} tokens, vocab {vocab})")
    return {
        "ref_docs": ref_docs,
        "cand_docs": cand,
        "dup": dup,
        "kept": kept,
        "kept_tokens": kept_tok,
        "out_total": total,
    }


def chain_dedup(dirs, suffix="-dd", in_place=False, eos=EOS_DEFAULT, quiet=False):
    """GLOBAL dedup across many dirs with ONE shared seen-set, processed in the given order
    (put the cleanest / most-canonical source FIRST so verbatim copies in dirtier later sources
    are the ones dropped). Writes each dir D to D+suffix; reports per-dir + grand total."""
    vocabs = {json.load(open(os.path.join(d, "index.json")))["vocab_size"] for d in dirs}
    if len(vocabs) > 1:
        raise SystemExit(f"vocab mismatch across dirs: {vocabs} (cannot share a doc space)")
    vocab = vocabs.pop()
    seen = set()
    rows, gtot, gdup = [], 0, 0
    import shutil

    for d in dirs:
        out = d.rstrip("/\\") + suffix
        if in_place:
            out = d.rstrip("/\\") + "-xd-tmp"
        w = _Writer(out, vocab=vocab)
        docs = dup = kept = kt = 0
        for doc in _iter_docs(d, eos):
            docs += 1
            h = _h(doc)
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            w.add(doc)
            kept += 1
            kt += int(doc.size)
        tot = w.close(vocab)
        if in_place:
            old = d.rstrip("/\\") + "-old"
            if os.path.exists(old):
                shutil.rmtree(old)
            try:
                os.rename(d, old)
                try:
                    os.rename(out, d)
                    out = d
                    shutil.rmtree(old)
                except Exception as e2:
                    os.rename(old, d)
                    raise RuntimeError(f"Failed to move new dir {out} into place: {e2}")
            except Exception as e:
                print(
                    f"  [ERROR] In-place swap failed for {d} ({e}). Deduped data safely retained in {out}"
                )
        rows.append((os.path.basename(d), docs, dup, kept, kt))
        gtot += kt
        gdup += dup
        if not quiet:
            print(
                f"  [{os.path.basename(d):<10}] docs {docs:>10,} | cross/self-dup {dup:>9,} "
                f"| kept {kept:>10,} = {kt / 1e9:6.3f}B  -> {out}"
            )
    if not quiet:
        print(
            f"\n  GLOBAL: {gdup:,} duplicate docs removed | corpus now {gtot / 1e9:.3f}B tokens "
            f"across {len(dirs)} dirs (vocab {vocab})"
        )
    return rows, gtot


def main(argv):
    p = argparse.ArgumentParser(
        description="Exact document-level cross-dedup of tokenized shard dirs"
    )
    p.add_argument("--candidate", help="shard dir to dedup (kept docs written to --out)")
    p.add_argument("--ref", action="append", default=[], help="reference shard dir (repeatable)")
    p.add_argument("--out", help="output deduped shard dir")
    p.add_argument(
        "--chain",
        nargs="+",
        help="GLOBAL mode: ordered dirs (cleanest first); "
        "each dir D is deduped against all earlier dirs + itself, written to D+suffix",
    )
    p.add_argument("--suffix", default="-dd", help="output suffix for --chain mode (default -dd)")
    p.add_argument(
        "--in-place",
        action="store_true",
        help="replace the original directories with the deduped output to save disk space",
    )
    p.add_argument(
        "--eos", type=int, default=EOS_DEFAULT, help="EOS/doc-separator token id (default 0)"
    )
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.chain:
        print(f"crossdedup GLOBAL | {len(a.chain)} dirs (order = priority): {a.chain}")
        chain_dedup(a.chain, suffix=a.suffix, in_place=a.in_place, eos=a.eos)
        print("done")
        return 0
    if not (a.candidate and a.ref and a.out):
        p.error("need --candidate + --ref + --out, or --chain, or --selftest")
    print(f"crossdedup | candidate={a.candidate} refs={a.ref}")
    crossdedup(a.candidate, a.ref, a.out, eos=a.eos)
    print("done")
    return 0


def selftest():
    print("crossdedup self-test")
    tmp = tempfile.mkdtemp(prefix="crossdedup_")

    def write_dir(d, docs, eos=0, vocab=300):
        os.makedirs(d)
        stream = []
        for doc in docs:
            stream += list(doc) + [eos]
        _, bpt = shard_format_for_vocab(vocab)
        dt = "<u2" if bpt == 2 else "<u4"
        arr = np.array(stream, dtype=dt)
        half = len(arr) // 2  # split mid-doc to exercise carry-over
        arr[:half].tofile(os.path.join(d, "shard_00000.bin"))
        arr[half:].tofile(os.path.join(d, "shard_00001.bin"))
        with open(os.path.join(d, "index.json"), "w") as f:
            json.dump(
                {
                    "vocab_size": vocab,
                    "total_tokens": int(arr.size),
                    "shards": [
                        {"file": "shard_00000.bin", "tokens": int(half)},
                        {"file": "shard_00001.bin", "tokens": int(arr.size - half)},
                    ],
                },
                f,
            )

    A, B, C, D = [10, 11, 12], [20, 21], [30, 31, 32, 33], [40, 41]
    ref, cand, out = os.path.join(tmp, "ref"), os.path.join(tmp, "cand"), os.path.join(tmp, "out")
    write_dir(ref, [A, B, C])  # reference has A, B, C
    write_dir(cand, [B, C, D, D])  # candidate: B/C dup ref, D new, 2nd D self-dup

    checks = {}
    # split-on-eos recovers docs across the shard boundary
    rd = [x.tolist() for x in _iter_docs(ref)]
    checks["iter_docs recovers docs across shard boundary"] = rd == [A + [0], B + [0], C + [0]]

    r = crossdedup(cand, [ref], out, quiet=True)
    checks["ref docs counted"] = r["ref_docs"] == 3
    checks["dup-of-ref + self-dup dropped"] = r["dup"] == 3  # B, C (ref) + 2nd D (self)
    checks["only the unique doc kept"] = r["kept"] == 1
    got = [x.tolist() for x in _iter_docs(out)]
    checks["kept doc is exactly D (+eos)"] = got == [D + [0]]
    checks["output verifies (index total matches)"] = r["out_total"] == len(D) + 1

    # all-duplicate candidate -> empty but valid output
    out2 = os.path.join(tmp, "out2")
    r2 = crossdedup(cand, [ref, cand], out2, quiet=True)  # ref+cand as refs => everything dup
    checks["all-dup candidate -> 0 kept, valid index"] = r2["kept"] == 0 and os.path.isfile(
        os.path.join(out2, "index.json")
    )

    # chain (global) mode: 3 dirs sharing docs; first keeps all, later drop cross/self dups
    c1, c2, c3 = os.path.join(tmp, "c1"), os.path.join(tmp, "c2"), os.path.join(tmp, "c3")
    write_dir(c1, [A, B])  # owns A, B
    write_dir(c2, [B, C])  # B dup-of-c1 -> drop; C new
    write_dir(c3, [A, C, D])  # A,C already seen -> drop; D new
    rows, gtot = chain_dedup([c1, c2, c3], suffix="-dd", quiet=True)
    kept_by = {name: kept for (name, _d, _dup, kept, _kt) in rows}
    checks["chain: first dir keeps all"] = kept_by["c1"] == 2
    checks["chain: 2nd drops cross-dup"] = kept_by["c2"] == 1
    checks["chain: 3rd drops all earlier"] = kept_by["c3"] == 1
    uniq = sum(len(x) for x in [A, B, C, D]) + 4  # 4 unique docs + their eos
    checks["chain: global total = unique docs only"] = gtot == uniq

    # uint32 round-trip: a token id > 65535 must survive crossdedup's write side intact, not get
    # silently truncated mod 65536 by a write path that ignores vocab (the bug this guards).
    BIG = [99999, 70000, 1]
    refb, candb, outb = (os.path.join(tmp, n) for n in ("refb", "candb", "outb"))
    write_dir(refb, [[1, 2, 3]], vocab=100000)
    write_dir(candb, [BIG], vocab=100000)
    rb = crossdedup(candb, [refb], outb, quiet=True)
    checks["uint32: candidate kept (no false dup)"] = rb["kept"] == 1
    gotb = [x.tolist() for x in _iter_docs(outb)]
    checks["uint32: token ids survive intact (no mod-65536 truncation)"] = gotb == [BIG + [0]]
    checks["uint32: output shard is 4 bytes/token"] = (
        os.path.getsize(os.path.join(outb, "shard_00000.bin")) == (len(BIG) + 1) * 4
    )

    ok = True
    for n, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {n}")
        ok &= v
    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    sys.exit(main(sys.argv[1:]))
