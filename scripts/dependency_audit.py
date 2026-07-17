#!/usr/bin/env python3
"""Audit CHARKHA's dependency lock without mutating the active training environment.

The production stack intentionally tracks near-latest packages, but CUDA kernel packages are coupled:
Torch, Triton, flash-linear-attention, and CUDA wheels must move together and must be proven on GPU.
This script makes those constraints executable so an eager upgrade cannot quietly invalidate the 8GB path.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements.txt"

LOCKED = {
    "torch": "2.6.0",
    "triton": "3.2.0",
    "flash-linear-attention": "0.5.1",
    "fla-core": "0.5.1",
    "sympy": "1.13.1",
    "mpmath": "1.3.0",
    "datasets": "5.0.0",
    "fsspec": "2026.4.0",
    "transformers": "5.12.1",
    "tokenizers": "0.22.2",
    "typer": "0.25.1",
}

RISKY_MAJOR_STACK = {"torch", "triton", "flash-linear-attention", "fla-core", "bitsandbytes"}


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


def parse_requirements(path: Path = REQ) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"([A-Za-z0-9_.-]+)==([^;\s]+)", line)
        if m:
            pins[_norm(m.group(1))] = m.group(2)
    return pins


def installed_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def run_pip(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def static_guards(pins: dict[str, str]) -> list[str]:
    errors: list[str] = []
    for name, want in LOCKED.items():
        got = pins.get(name)
        if got != want:
            errors.append(
                f"{name} must stay pinned to {want} in the production lock; found {got!r}"
            )

    if pins.get("torch") == "2.6.0" and pins.get("triton") != "3.2.0":
        errors.append("torch 2.6.0 requires triton==3.2.0")
    if pins.get("datasets") == "5.0.0" and pins.get("fsspec") != "2026.4.0":
        errors.append("datasets 5.0.0 requires fsspec<=2026.4.0; keep the proven exact pin")
    if pins.get("transformers") == "5.13.0":
        errors.append(
            "transformers 5.13.0 dry-run conflicted with stable tokenizers 0.23.1; prove first"
        )
    return errors


def report_outdated() -> int:
    code, out = run_pip(["list", "--outdated", "--format=json"])
    if code != 0:
        print(out, end="")
        return code
    rows = json.loads(out or "[]")
    pins = parse_requirements()
    print("name current latest action")
    for row in rows:
        name = _norm(row["name"])
        current = row["version"]
        latest = row["latest_version"]
        if name in RISKY_MAJOR_STACK or name.startswith("nvidia-"):
            action = "isolate+GPU-proof"
        elif name in LOCKED and pins.get(name) == LOCKED[name]:
            action = "held-by-production-lock"
        else:
            action = "candidate-after-pip-check"
        print(f"{name} {current} {latest} {action}")
    return 0


def selftest() -> int:
    pins = parse_requirements()
    errors = static_guards(pins)
    required = {"torch", "triton", "flash-linear-attention", "datasets", "fsspec", "typer"}
    missing = sorted(required - pins.keys())
    if missing:
        errors.append(f"requirements.txt missing explicit pins: {missing}")
    if errors:
        for err in errors:
            print(f"[deps] FAIL {err}")
        return 1
    print(f"[deps] PASS production dependency guards ({len(pins)} pins)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdated", action="store_true", help="show outdated packages with CHARKHA risk labels"
    )
    ap.add_argument(
        "--pip-check", action="store_true", help="run pip check in the active environment"
    )
    ap.add_argument(
        "--selftest", action="store_true", help="check requirements.txt against CHARKHA lock rules"
    )
    args = ap.parse_args()

    rc = 0
    if args.selftest or not (args.outdated or args.pip_check):
        rc = selftest() or rc
    if args.pip_check:
        code, out = run_pip(["check"])
        print(out, end="")
        rc = code or rc
    if args.outdated:
        rc = report_outdated() or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
