"""Private prompt benchmark harness for CHARKHA.

JSONL input format:
  {"prompt": "...", "contains": "expected substring", "answer": "..."}
  {"prompt": "...", "equals": "exact answer", "answer": "..."}
  {"prompt": "...", "regex": "pattern", "answer": "..."}

The scoring core is model-agnostic so it can be used in fast tests and by serve.py
launchers that fill in answers before scoring.
"""

from __future__ import annotations
import argparse
import json
import re
import sys


def score_one(record):
    answer = str(record.get("answer", ""))
    if "equals" in record:
        ok = answer.strip() == str(record["equals"]).strip()
    elif "contains" in record:
        ok = str(record["contains"]).lower() in answer.lower()
    elif "regex" in record:
        ok = re.search(str(record["regex"]), answer, re.I | re.S) is not None
    else:
        ok = bool(answer.strip())
    return {"prompt": record.get("prompt", ""), "ok": bool(ok), "answer": answer}


def score_records(records):
    rows = [score_one(r) for r in records]
    acc = sum(1 for r in rows if r["ok"]) / max(1, len(rows))
    return {"n": len(rows), "accuracy": acc, "rows": rows}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def selftest():
    records = [
        {"prompt": "2+2", "answer": "4", "equals": "4"},
        {"prompt": "gpu", "answer": "You use a 4060 Ti.", "contains": "4060"},
        {"prompt": "shape", "answer": "abc-123", "regex": r"[a-z]+-\d+"},
        {"prompt": "miss", "answer": "no", "contains": "yes"},
    ]
    out = score_records(records)
    print(json.dumps({"n": out["n"], "accuracy": out["accuracy"]}))
    return 0 if out["n"] == 4 and abs(out["accuracy"] - 0.75) < 1e-9 else 1


def main():
    ap = argparse.ArgumentParser(description="Score private prompt/answer eval JSONL")
    ap.add_argument("--jsonl")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.jsonl:
        ap.error("--jsonl required unless --selftest")
    out = score_records(load_jsonl(args.jsonl))
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
