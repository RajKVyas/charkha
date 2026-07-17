#!/usr/bin/env python3
"""Train CHARKHA's custom tokenizer — a code/math/assembly-aware ByteLevel BPE.

Design (proven pieces, not a mashup of redundant ones):
  • ByteLevel BPE with the full 256-byte initial alphabet  -> NEVER emits <unk>; raw hex, x86
    registers, Ghidra dumps, binary, and unicode all round-trip as bytes. (This is what Qwen/NeoX/
    GPT-2 actually do; "byte_fallback" is the SentencePiece equivalent and is redundant here.)
  • Digits(individual_digits=True) pre-tokenizer  -> every digit is its own token, so 4096 -> 4 0 9 6.
    This gives the NeuralMath/Calculus modules native place-value alignment and RETIRES the
    dataprep --digit-split string hack. Numbers tokenize identically regardless of magnitude.
  • vocab_size 65535  -> fits uint16 shards exactly (no uint32 migration, no overflow risk).

Trains from RAW TEXT (the -dd dirs are already tokenized, so point this at raw samples or stream a
representative slice from HF). ~2-5 GB of mixed text is plenty for a world-class BPE.

  # from local raw text (txt/jsonl/md), recursive:
  python scripts/train_tokenizer.py --input /path/to/raw_samples --out tokenizer.json
  # or stream a representative HF sample (needs `datasets`):
  python scripts/train_tokenizer.py --hf your-org/your-dataset:config \
      --hf-bytes 3_000_000_000 --out tokenizer.json

"""

import argparse
import glob
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

VOCAB = 65535  # default: fits uint16 with headroom (max uint16 = 65535).
# --vocab-size can go above this (e.g. 100000) for better
# compression at the cost of uint32 shards (src/dataprep.py's
# shard_format_for_vocab picks the format automatically) and a
# bigger tied embedding table (vocab_size * d_model params).
SPECIALS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|thinking|>",
    "<|answer|>",
    "<|search|>",
    "<|result|>",
    "<|tool|>",
]

# GPT-2's own ByteLevel split regex, with ONE surgical addition: a `0[xX][0-9a-fA-F]+` branch
# inserted before the letter-run branch so hex literals ("0x7fffffffe3a0") match and isolate as
# ONE piece instead of falling through to the bare `[0-9]` (place-value digit) branch. Decimal
# digits still isolate one-at-a-time (NeuralMath place-value alignment is unaffected — this only
# carves out the one digit class, hex addresses/opcodes, where place-value is meaningless and
# isolating it was pure compression loss on asm/RE corpora vs every GPT-family tokenizer, which
# merges hex freely). Must be applied as a SINGLE Split step, not chained before ByteLevel: a
# Sequence's later steps re-split every piece the earlier ones produced (ByteLevel's own regex
# would re-shred an already-isolated hex literal into alternating digit/letter runs), so ByteLevel
# here runs with use_regex=False (byte-remap only, no re-splitting).
GPT2_HEX_REGEX = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?0[xX][0-9a-fA-F]+"
    r"| ?\p{L}+"
    r"| ?[0-9]"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def build(vocab_size=VOCAB, hex_merge=True):
    from tokenizers import Tokenizer, Regex
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Sequence, ByteLevel, Digits, Split
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    tok = Tokenizer(BPE(unk_token=None))  # ByteLevel => no unk is ever needed
    if hex_merge:
        tok.pre_tokenizer = Sequence(
            [
                Split(pattern=Regex(GPT2_HEX_REGEX), behavior="isolated"),
                ByteLevel(
                    add_prefix_space=False, use_regex=False
                ),  # byte remap ONLY (see comment above)
            ]
        )
    else:
        tok.pre_tokenizer = Sequence(
            [
                Digits(individual_digits=True),  # isolate every digit (place-value math)
                ByteLevel(
                    add_prefix_space=False, use_regex=True
                ),  # GPT-2 byte mapping + word split
            ]
        )
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )
    return tok, trainer


def _iter_local(paths, max_bytes):
    seen = 0
    exts = ("*.txt", "*.jsonl", "*.json", "*.md", "*.py", "*.rs", "*.c", "*.asm")
    files = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        else:
            for e in exts:
                files.extend(glob.glob(os.path.join(p, "**", e), recursive=True))
    import random

    random.seed(0)
    random.shuffle(files)
    for fn in files:
        try:
            with open(fn, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if fn.endswith((".jsonl", ".json")):
                        try:
                            obj = json.loads(line)
                            line = obj.get("text") or obj.get("content") or ""
                        except Exception:
                            pass
                    if not line:
                        continue
                    yield line
                    seen += len(line.encode("utf-8"))
                    if seen >= max_bytes:
                        return
        except Exception:
            continue


def _iter_hf(specs, max_bytes):
    from datasets import load_dataset

    per = max_bytes // max(1, len(specs))
    for spec in specs:
        name, _, cfg = spec.partition(":")
        seen = 0
        try:
            ds = load_dataset(name, cfg or None, split="train", streaming=True)
        except Exception as e:
            print(f"[hf] skip {spec}: {e}")
            continue
        for row in ds:
            txt = row.get("text") or row.get("content") or ""
            if not txt:
                continue
            yield txt
            seen += len(txt.encode("utf-8"))
            if seen >= per:
                break
        print(f"[hf] {spec}: ~{seen / 1e6:.0f}MB sampled")


def verify(path):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(path)
    print(f"\n[verify] vocab={tok.get_vocab_size()} (uint16 ok: {tok.get_vocab_size() <= 65535})")
    probes = {
        "math": "Calculate 4096 * 2.5 = 10240 and 14523+98124",
        "asm": "mov eax, DWORD PTR [rbp-0x4]; lea rdi, 0x7fffffffe3a0; call 0x401136",
        "code": "def f(x):\n    return x ** 2 + 1  # comment",
        "prose": "The quick brown fox jumps over the lazy dog.",
    }
    for k, s in probes.items():
        ids = tok.encode(s).ids
        back = tok.decode(ids)
        digits = [tok.decode([i]) for i in ids if tok.decode([i]).strip().isdigit()]
        allsingle = all(len(d.strip()) == 1 for d in digits) if digits else True
        rt = "✓" if back.replace(" ", "") == s.replace(" ", "") or back == s else "≈"
        print(f"  {k:5} {len(ids):3d} tok  digits-isolated={allsingle}  roundtrip={rt}")
    # hex-merge probe: a single hex literal should NOT explode into one-token-per-character
    # (that's the asm/RE compression we're recovering); decimal digits must still be one-per-token.
    hx = "lea rdi, 0x7fffffffe3a0"
    hx_ids = tok.encode(hx).ids
    hx_back = tok.decode(hx_ids)
    print(
        f"  hex   {len(hx_ids):3d} tok  (0x7fffffffe3a0 is 14 chars -> "
        f"{'merged' if len(hx_ids) < 10 else 'still char-split'})  roundtrip="
        f"{'✓' if hx_back.replace(' ', '') == hx.replace(' ', '') or hx_back == hx else '≈'}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", nargs="*", default=[], help="raw text dirs/files (txt/jsonl/md/code)"
    )
    ap.add_argument(
        "--hf", action="append", default=[], help="HF dataset spec name[:config] (repeatable)"
    )
    ap.add_argument(
        "--hf-bytes", type=int, default=3_000_000_000, help="total bytes to stream from HF"
    )
    ap.add_argument(
        "--local-bytes", type=int, default=5_000_000_000, help="max bytes from local files"
    )
    ap.add_argument("--out", default="charkha_tokenizer.json")
    ap.add_argument(
        "--vocab-size",
        type=int,
        default=VOCAB,
        help=f"target vocab size (default {VOCAB}, fits uint16 shards). Going above "
        "65535 needs uint32 shards (src/dataprep.py picks this up automatically "
        "via shard_format_for_vocab) and a bigger tied embedding table.",
    )
    ap.add_argument(
        "--hex-merge",
        dest="hex_merge",
        action="store_true",
        default=True,
        help="isolate hex literals (0x...) whole instead of digit-by-digit (default on: "
        "place-value math only needs DECIMAL digits isolated; merging hex recovers "
        "compression on asm/RE corpora where every other tokenizer already merges it)",
    )
    ap.add_argument("--no-hex-merge", dest="hex_merge", action="store_false")
    ap.add_argument("--verify-only", action="store_true")
    a = ap.parse_args()
    if a.verify_only:
        verify(a.out)
        return 0
    if not a.input and not a.hf:
        print("give --input <raw text dir> and/or --hf <dataset spec>")
        return 2

    tok, trainer = build(a.vocab_size, hex_merge=a.hex_merge)

    def corpus():
        if a.input:
            yield from _iter_local(a.input, a.local_bytes)
        if a.hf:
            yield from _iter_hf(a.hf, a.hf_bytes)

    mode = "hex-merge + decimal-isolate" if a.hex_merge else "individual-digit"
    print(f"[train] target vocab {a.vocab_size}, ByteLevel BPE + {mode} pre-tokenizer ...")
    tok.train_from_iterator(corpus(), trainer)
    tok.save(a.out)
    print(f"[done] saved {a.out}  (vocab {tok.get_vocab_size()})")
    verify(a.out)
    shard_note = (
        "uint16"
        if tok.get_vocab_size() <= 65535
        else "uint32 (requires re-tokenizing existing shards)"
    )
    print(
        f"\nNEXT: point dataprep.py --tokenizer {a.out} at this tokenizer and DROP --digit-split "
        f"(now native). Shard format: {shard_note}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
