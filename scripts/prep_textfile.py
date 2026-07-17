#!/usr/bin/env python3
"""Tokenize a raw text corpus (documents separated by <|endoftext|> lines) into CHARKHA
Sutra-131k/v8 training shards — same conventions as src/dataprep.py (per-doc encode + EOS append,
ShardWriter uint16/uint32 format, index.json layout), so the output dir drops straight
into `train.py --data`.

Built for TinyStoriesV2 (the fast-fluency corpus for the mini -> grow-init pipeline),
but any <|endoftext|>-separated .txt works:

  python scripts/prep_textfile.py \
      --in data/raw/tinystories/TinyStoriesV2-GPT4-train.txt \
      --out data/corpus-dd --tokenizer charkha_tokenizer.json

"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataprep import load_tokenizer, ShardWriter, shard_format_for_vocab, normalize  # noqa: E402

SEP = "<|endoftext|>"


def iter_docs(path):
    """Yield documents from a SEP-delimited text file, streaming (no full-file load)."""
    buf = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == SEP:
                doc = "".join(buf).strip()
                if doc:
                    yield doc
                buf = []
            else:
                buf.append(line)
    doc = "".join(buf).strip()
    if doc:
        yield doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="inp", required=True, help="raw .txt (docs separated by <|endoftext|>)"
    )
    ap.add_argument("--out", required=True, help="output shard dir (e.g. data/corpus-dd)")
    ap.add_argument("--tokenizer", default=os.path.join(ROOT, "charkha_tokenizer.json"))
    ap.add_argument(
        "--threads", type=int, default=8, help="tokenizer threads (Rust tokenizers release the GIL)"
    )
    ap.add_argument("--batch", type=int, default=2048, help="docs per tokenize batch")
    a = ap.parse_args()

    tok_path = os.path.abspath(a.tokenizer)
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"tokenizer not found: {tok_path}")
    encode, vocab = load_tokenizer(tok_path)
    fmt, itemsize = shard_format_for_vocab(vocab)
    print(
        f"[prep] tokenizer={tok_path} vocab={vocab} shard-dtype={'uint32' if itemsize == 4 else 'uint16'}"
    )

    writer = ShardWriter(a.out, vocab_size=vocab)
    t0, docs, toks = time.time(), 0, 0
    batch = []

    def flush_batch():
        nonlocal docs, toks
        if not batch:
            return
        with ThreadPoolExecutor(max_workers=a.threads) as ex:
            for ids in ex.map(lambda d: encode(normalize(d)), batch):
                writer.add(ids)
                toks += len(ids)
        docs += len(batch)
        batch.clear()
        el = time.time() - t0
        print(
            f"\r[prep] {docs:,} docs | {toks / 1e6:.1f}M tok | {toks / max(el, 1e-6) / 1e3:,.0f}K tok/s",
            end="",
            flush=True,
        )

    for doc in iter_docs(a.inp):
        batch.append(doc)
        if len(batch) >= a.batch:
            flush_batch()
    flush_batch()
    writer.close()

    import json

    index = {
        "vocab_size": vocab,
        "total_tokens": writer.total,
        "tokenizer": tok_path,
        "shards": writer.shards,
    }
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(
        f"\n[prep] DONE: {writer.total:,} tokens in {len(writer.shards)} shards -> {a.out} "
        f"({time.time() - t0:,.0f}s)"
    )


if __name__ == "__main__":
    main()
