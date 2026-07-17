#!/usr/bin/env python3
"""Post-process a freshly-trained tokenizer.json into a serve-ready CHARKHA tokenizer, matching the
known-good v4 exactly. The SuperBPE fork saves tokenizer.json with `decoder: null`, so raw decode
emits ByteLevel artifacts (Ġ, Ċ). We do NOT use superbpe's construct_hf_tokenizer(): it appends a
fresh EOS/PAD at ids vocab_size / vocab_size+1, but CHARKHA already bakes <|endoftext|>=0 and
<|pad|>=1 as real vocab entries, so those computed ids are invalid and decode to ''.

This just: (1) attaches the ByteLevel decoder v4 uses, (2) verifies specials/round-trip/digit
isolation, (3) writes the output. Encode (hence compression) is unchanged by the decoder, so a
pre/post benchmark should match to the token; this script prints a round-trip check to prove decode
is now clean.

  python scripts/postprocess_tokenizer.py IN.json OUT.json
"""

import sys
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

EXPECT_SPECIALS = {
    0: "<|endoftext|>",
    1: "<|pad|>",
    2: "<|system|>",
    3: "<|user|>",
    4: "<|assistant|>",
}
SAMPLES = [
    "The quick brown fox jumps over the lazy dog.",
    "def add(a, b):\n    return a + b  # 0xDEADBEEF and 12345",
    "Les résultats étaient très bons — 100% d'accord.",
]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    inp, outp = sys.argv[1], sys.argv[2]
    tok = Tokenizer.from_file(inp)
    # match v4: ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True)
    tok.decoder = ByteLevelDecoder(add_prefix_space=True, trim_offsets=True, use_regex=True)

    vocab = tok.get_vocab()
    print(f"vocab size: {tok.get_vocab_size()}")
    ok = True
    for i, s in EXPECT_SPECIALS.items():
        got = tok.id_to_token(i)
        mark = "OK" if got == s else "MISMATCH"
        if got != s:
            ok = False
        print(f"  id {i}: {got!r} (expect {s!r}) {mark}")

    print("round-trip decode (must have NO leading-space / Ġ / Ċ artifacts):")
    for s in SAMPLES:
        ids = tok.encode(s).ids
        dec = tok.decode(ids)
        clean = "Ġ" not in dec and "Ċ" not in dec
        # decode drops the model's byte-level prefix space; compare on stripped text
        match = dec.strip() == s.strip()
        print(
            f"  [{'clean' if clean else 'ARTIFACTS'} | {'match' if match else 'DIFF'}] "
            f"{len(ids)} toks  {dec!r}"
        )
        if not clean:
            ok = False

    # digit isolation: multi-digit runs must split into single-digit tokens (place value)
    dig = tok.encode("year 2024 value 130806 pi 3141592").ids
    pieces = [tok.decode([i]).strip() for i in dig]
    multidigit = [p for p in pieces if p.isdigit() and len(p) > 1]
    print(
        f"digit isolation: {'OK (all single)' if not multidigit else 'FAIL multi-digit ' + str(multidigit)}"
    )

    tok.save(outp)
    print(f"{'WROTE' if ok else 'WROTE (WITH WARNINGS)'} -> {outp}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
