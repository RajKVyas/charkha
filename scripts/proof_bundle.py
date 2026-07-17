#!/usr/bin/env python3
"""Run CHARKHA's CPU-safe proof bundle.

This is a reproducible evidence bundle for code review/hardening passes. It intentionally hides CUDA
so it cannot steal VRAM from a live training run, then runs syntax checks and selftests that cover
checkpoint merge, embedding Carding, freshness ingest, curation, distillation, and fast regressions.

Usage:
  python scripts/proof_bundle.py                # full bundle (pre-push gate)
  python scripts/proof_bundle.py --syntax-only  # static guards + py_compile only (pre-commit gate)
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def python_files() -> list[str]:
    files = []
    for root in ("src", "scripts", "tests"):
        for path in (ROOT / root).rglob("*.py"):
            if "__pycache__" not in path.parts:
                files.append(str(path.relative_to(ROOT)))
    return sorted(files)


PY_FILES = python_files()

FORBIDDEN_PATTERNS = {
    "direct CUDA checkpoint load": "map_location=device",
}

COMMANDS = [
    ["-m", "py_compile", *PY_FILES],
    ["scripts/check_compile_grads.py"],
    ["scripts/loom_merge.py", "--selftest"],
    ["scripts/card_embeddings.py", "--selftest"],
    ["scripts/news_ingest.py", "--selftest"],
    ["scripts/curate_jsonl.py", "--selftest"],
    ["scripts/dependency_audit.py", "--selftest"],
    ["scripts/cache_logits.py", "--selftest"],
    ["src/murmur.py", "--selftest"],
    ["src/murmur_micro.py", "--selftest"],
    ["src/accordion_micro.py", "--selftest"],
    ["src/vdepth_micro.py", "--selftest"],
    ["src/fold.py", "--selftest"],
    ["scripts/build_re_dataset.py", "--selftest"],
    ["src/ply.py", "--selftest"],
    ["src/ply_micro.py", "--selftest"],
    ["src/latent_memory.py", "--selftest"],
    ["src/proofcarry.py", "--selftest"],
    ["src/verify_shards.py", "--selftest"],
    ["src/retrieval.py", "--selftest"],
    ["src/distill.py", "--selftest"],
    ["src/serve.py", "--selftest"],
    ["tests/run_tests.py"],
]


def run(cmd: list[str]) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    print("\n[proof]", " ".join([sys.executable, *cmd]), flush=True)
    proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env)
    print(f"[proof] exit={proc.returncode}", flush=True)
    return proc.returncode == 0


def static_guards() -> bool:
    ok = True
    search_roots = [ROOT / "src", ROOT / "scripts", ROOT / "tests"]
    for label, needle in FORBIDDEN_PATTERNS.items():
        hits = []
        for root in search_roots:
            for path in root.rglob("*.py"):
                text = path.read_text(encoding="utf-8", errors="ignore")
                if needle in text and path.name != "proof_bundle.py":
                    hits.append(path.relative_to(ROOT))
        if hits:
            ok = False
            print(f"[proof-static] FAIL {label}: {needle!r} in {hits}", flush=True)
        else:
            print(f"[proof-static] PASS {label}", flush=True)

    # Runtime guards in model code must survive python -O; selftests may use asserts.
    charkha = ROOT / "src" / "charkha" / "_model.py"
    all_lines = charkha.read_text(encoding="utf-8", errors="ignore").splitlines()
    prod_asserts = [
        (i, line.strip())
        for i, line in enumerate(all_lines, 1)
        if "assert " in line and "device-side assert" not in line
    ]
    if prod_asserts:
        ok = False
        print(f"[proof-static] FAIL production asserts survive -O: {prod_asserts[:5]}", flush=True)
    else:
        print("[proof-static] PASS production runtime guards avoid assert", flush=True)

    train_text = (ROOT / "src" / "train.py").read_text(encoding="utf-8", errors="ignore")
    has_r_off = re.search(r"[\"']r-off[\"']", train_text) is not None
    has_recurrence_flag = (
        re.search(r"[\"']recurrence[\"']\s*:\s*bool\(model\.cfg\.use_recurrence\)", train_text)
        is not None
    )
    if has_r_off and has_recurrence_flag:
        print("[proof-static] PASS recurrence-off logging is explicit", flush=True)
    else:
        ok = False
        print("[proof-static] FAIL recurrence-off logging is not explicit", flush=True)
    return ok


def main() -> int:
    syntax_only = "--syntax-only" in sys.argv[1:]
    ok = static_guards()
    print(f"[proof-static] Python files syntax-checked: {len(PY_FILES)}", flush=True)
    for cmd in COMMANDS[:1] if syntax_only else COMMANDS:
        ok = run(cmd) and ok
    print("\nPROOF_BUNDLE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
