# Tokenizer provenance

The checked-in `charkha_tokenizer.json` was rebuilt for the corrected `v0.1.1`
release from one recorded, license-identified source:

- Dataset: [`Salesforce/wikitext`](https://huggingface.co/datasets/Salesforce/wikitext)
- Configuration: `wikitext-103-raw-v1`
- Dataset license: CC BY-SA 3.0
- Local source snapshot: `data/tok_corpus_v8s2b/general_wikitext.txt`
- Source snapshot SHA-256: `6e8b17ec3a208d60b407a45a6b89bf4b3b1689eb0a7470177616775332b2414c`
- Tokenizer SHA-256: `d9dfcd5b3523cfeee1403a2fc6066dd49c1feca2e222bfc19cfbe32d4af40cf6`
- Vocabulary: 65,535 entries (fits uint16 token shards)

It was generated with:

```text
python scripts/train_tokenizer.py \
  --input data/tok_corpus_v8s2b/general_wikitext.txt \
  --out charkha_tokenizer.json \
  --vocab-size 65535
```

The source snapshot is not distributed in this repository. To reproduce the
artifact, download the cited dataset, apply the repository's text preparation
pipeline, and verify the recorded hashes. The previous bundled tokenizer had
no reproducible corpus record and was replaced rather than relabeled.
