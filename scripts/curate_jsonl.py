#!/usr/bin/env python3
"""CHARKHA curator: JSONL -> model-selected JSONL by quality + learnability band.

This is the practical RHO-style diet gate. The current checkpoint scores candidate documents; very low
loss means "already learned / redundant", very high loss usually means garbage or too hard right now,
and the middle band is the useful training diet. Output stays as raw JSONL so src/dataprep.py still
performs the canonical data gates before token sharding.

Usage:
  python scripts/curate_jsonl.py --inp data/news/news.jsonl --out data/news/news_curated.jsonl \
    --model runs/main/ckpt.pt --tokenizer charkha_tokenizer.json --loss-min 1.5 --loss-max 7.0
  python scripts/curate_jsonl.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from statistics import mean

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dataprep import normalize, pii_scrub, quality_filter, load_tokenizer  # noqa: E402


def load_model(path: str, device: str):
    from charkha import Charkha, CharkhaConfig

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = CharkhaConfig.from_dict(ck["cfg"]) if isinstance(ck.get("cfg"), dict) else ck["cfg"]
    model = Charkha(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


def score_ids(model, ids: list[int], device: str, max_tokens: int = 1024) -> float:
    if len(ids) < 3:
        return math.inf
    ids = ids[: max_tokens + 1]
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        _logits, loss = model(x, y, r=1)
    return float(loss.detach().cpu())


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[curate] skip malformed {path}:{ln}: {e}")


def curate(
    inp: str,
    out: str,
    tokenizer: str | None = None,
    model_path: str | None = None,
    device: str = "cpu",
    loss_min: float = 1.5,
    loss_max: float = 7.0,
    max_tokens: int = 1024,
    limit: int | None = None,
    digit_split: bool = False,
):
    encode, vocab = load_tokenizer(tokenizer, digit_split=digit_split)
    model = None
    if model_path:
        model, cfg = load_model(model_path, device)
        if cfg.vocab_size != vocab:
            raise ValueError(f"tokenizer vocab {vocab} != checkpoint vocab {cfg.vocab_size}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    kept = seen = drop_quality = drop_band = 0
    scores = []
    with open(out, "w", encoding="utf-8") as fo:
        for _ln, doc in iter_jsonl(inp):
            if limit and seen >= limit:
                break
            seen += 1
            text = normalize(str(doc.get("text") or doc.get("content") or ""))
            text, _ = pii_scrub(text)
            ok, reason = quality_filter(text, min_words=25, target_langs=["en"])
            if not ok:
                drop_quality += 1
                continue
            loss = None
            if model is not None:
                ids = encode(text)
                loss = score_ids(model, ids, device=device, max_tokens=max_tokens)
                scores.append(loss)
                if not (loss_min <= loss <= loss_max):
                    drop_band += 1
                    continue
            doc["text"] = text
            doc["curator"] = {
                "loss": loss,
                "loss_min": loss_min,
                "loss_max": loss_max,
                "reason": "quality+learnability" if model is not None else "quality",
            }
            fo.write(json.dumps(doc, ensure_ascii=False) + "\n")
            kept += 1
    msg = f"[curate] kept={kept} seen={seen} drop_quality={drop_quality} drop_band={drop_band}"
    if scores:
        msg += (
            f" loss_mean={mean(scores):.3f} loss_min={min(scores):.3f} loss_max={max(scores):.3f}"
        )
    print(msg)
    return kept


def selftest():
    print("CHARKHA curator self-test")
    tmp = os.path.join("/tmp", "charkha_curate_selftest")
    os.makedirs(tmp, exist_ok=True)
    inp = os.path.join(tmp, "in.jsonl")
    out = os.path.join(tmp, "out.jsonl")
    docs = [
        {
            "text": "The small model reads the clean story and learns ordinary English from careful examples. The story says that the model can use the words and make a clear answer for the reader. This document has enough normal words and the sentences are simple enough for the quality gate."
        },
        {"text": "x y z"},
    ]
    with open(inp, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    kept = curate(inp, out, model_path=None)
    rows = [json.loads(x) for x in open(out, encoding="utf-8")]
    checks = {
        "quality gate kept one doc": kept == 1 and len(rows) == 1,
        "curator metadata written": rows[0]["curator"]["reason"] == "quality",
    }
    ok = True
    for name, passed in checks.items():
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(
        "\nSELFTEST",
        "PASS - JSONL docs can be quality/model curated before dataprep" if ok else "FAIL",
    )
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(
        description="Curate JSONL docs by quality and optional model NLL band"
    )
    ap.add_argument("--inp", help="input JSONL")
    ap.add_argument("--out", help="output JSONL")
    ap.add_argument("--tokenizer", default=None, help="tokenizer path/name for model scoring")
    ap.add_argument("--model", default=None, help="optional CHARKHA checkpoint to score docs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--loss-min", type=float, default=1.5)
    ap.add_argument("--loss-max", type=float, default=7.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--digit-split", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.inp or not args.out:
        ap.error("--inp and --out are required")
    curate(
        args.inp,
        args.out,
        tokenizer=args.tokenizer,
        model_path=args.model,
        device=args.device,
        loss_min=args.loss_min,
        loss_max=args.loss_max,
        max_tokens=args.max_tokens,
        limit=args.limit,
        digit_split=args.digit_split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
