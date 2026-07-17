#!/usr/bin/env python3
"""Rebuild a shard-dir index.json from its .bin files.

Used when a dir has valid tokenized shards but a missing/corrupt index.json (the
training loader and verify_shards read the index, not the shards, so an unindexed
dir is invisible even though the data is fine). Sutra-131k/v8 shards are uint32 (vocab
131072 > 65535) -> 4 bytes/token; older uint16 dirs -> 2 bytes/token, auto-detected
from the max token id in the first shard.

Usage:
    python scripts/recover_index.py data/corpus-dd [data/other-dd ...]
    python scripts/recover_index.py --tokenizer charkha_tokenizer.json data/*-dd
"""

import argparse
import glob
import json
import os

import numpy as np


def detect_dtype(first_shard):
    """uint16 if every id fits, else uint32. Probe as uint32 and check the range."""
    a = np.memmap(first_shard, dtype=np.uint32, mode="r")
    sample = a[:1_000_000]
    return np.uint16 if sample.max() < 65536 else np.uint32


def rebuild(d, tokenizer):
    shards_glob = sorted(glob.glob(os.path.join(d, "shard_*.bin")))
    if not shards_glob:
        return None
    dtype = detect_dtype(shards_glob[0])
    itemsize = np.dtype(dtype).itemsize
    vocab = 131072 if dtype == np.uint32 else 50277
    shards, total = [], 0
    for f in shards_glob:
        tok = os.path.getsize(f) // itemsize
        shards.append({"file": os.path.basename(f), "tokens": tok})
        total += tok
    idx = {
        "vocab_size": vocab,
        "total_tokens": total,
        "tokenizer": tokenizer,
        "shards": shards,
        "accounting": {"recovered_index": True},
    }
    with open(os.path.join(d, "index.json"), "w") as f:
        json.dump(idx, f, indent=1)
    return total, len(shards), dtype


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dirs", nargs="+", help="shard dirs to (re)index")
    ap.add_argument(
        "--tokenizer",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "charkha_tokenizer.json"
        ),
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing index.json")
    a = ap.parse_args()
    grand = 0
    for d in a.dirs:
        if os.path.exists(os.path.join(d, "index.json")) and not a.force:
            print(f"[skip] {d} already has index.json (use --force)")
            continue
        r = rebuild(d, a.tokenizer)
        if r is None:
            print(f"[warn] {d}: no shard_*.bin found")
            continue
        total, n, dtype = r
        grand += total
        print(f"[ok]   {d}: {n} shards, {total / 1e9:.2f}B tokens ({np.dtype(dtype).name})")
    print(f"[done] recovered {grand / 1e9:.2f}B tokens total")
