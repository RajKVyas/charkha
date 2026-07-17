#!/usr/bin/env bash
# CHARKHA — canonical environment bootstrap (WSL2 / Linux + CUDA).
# ============================================================================
# CHARKHA's production Gated-DeltaNet path is the flash-linear-attention (fla)
# Triton kernel, which only exists on Linux. So the canonical training/benchmark
# environment is WSL2 + CUDA + triton. On native Windows GDN falls back to the
# pure-PyTorch chunked scan (correct + selftest-exact, but slower and numerically
# distinct) — fine for selftests and inference, NOT where real training belongs.
#
# This script builds that env once. Run it from the repo root INSIDE WSL:
#   bash scripts/setup_wsl.sh
#
# Recommended layout (avoids the slow /mnt/d FS and its >21KB write-truncation
# gotcha): keep code + venv + checkpoints on the Linux side (e.g. ~/charkha) and
# point --data at the big uint16 shards on /mnt/d (read-mostly).
#
# Env knobs:  VENV (default .venv)   CUDA (default cu124)   PY (default python3)
# set -euo pipefail

VENV="${VENV:-.venv}"
CUDA="${CUDA:-cu124}"
PY="${PY:-python3}"

echo "[setup] python: $($PY --version 2>&1)"
if ! grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
  echo "[setup] WARNING: this does not look like WSL/Linux. The fla/triton kernel"
  echo "[setup]          only installs on Linux; on native Windows use the fallback."
fi

case "$(pwd -P)" in
  /mnt/*)
    if [ "${ALLOW_MNT:-0}" != "1" ]; then
      echo "[setup] ERROR: repo is under /mnt/* ($(pwd -P))."
      echo "[setup]        Keep code + venv on the Linux filesystem, e.g.:"
      echo "[setup]        cd ~ && git clone <repo-url> charkha && cd ~/charkha"
      echo "[setup]        Keep big data on /mnt/d and pass --data /data/data/..."
      echo "[setup]        Override only for diagnostics: ALLOW_MNT=1 bash scripts/setup_wsl.sh"
      exit 2
    fi
    ;;
esac

if [ -n "${VIRTUAL_ENV:-}" ] && [ "$(realpath "$VIRTUAL_ENV")" != "$(realpath -m "$PWD/$VENV")" ]; then
  echo "[setup] ERROR: another virtualenv is active:"
  echo "[setup]        VIRTUAL_ENV=$VIRTUAL_ENV"
  echo "[setup]        Expected:   $PWD/$VENV"
  echo "[setup]        Run 'deactivate' first, then rerun setup."
  exit 2
fi

if [ ! -d "$VENV" ]; then
  echo "[setup] creating venv at $VENV"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

echo "[setup] installing torch ($CUDA) ..."
pip install torch --index-url "https://download.pytorch.org/whl/${CUDA}"

echo "[setup] installing the production GDN kernel (flash-linear-attention + triton) ..."
pip install triton flash-linear-attention

echo "[setup] installing data/serve/eval deps ..."
pip install tokenizers transformers datasets pyyaml safetensors lm-eval numpy ftfy bitsandbytes

echo "[setup] verifying the production path is live ..."
python - <<'PY'
import torch
try:
    import fla.ops.gated_delta_rule  # noqa: F401
    have_fla = True
except Exception as e:
    have_fla = False
    print(f"  fla import FAILED: {type(e).__name__}: {e}")
print(f"  torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)} | bf16: {torch.cuda.is_bf16_supported()}")
print(f"  fla/Triton GDN kernel: {'OK (production path)' if have_fla else 'MISSING (fallback only)'}")
if not (have_fla and torch.cuda.is_available()):
    raise SystemExit("[setup] environment is NOT production-ready (need CUDA + fla).")
print("[setup] GREEN — canonical CHARKHA environment ready.")
PY

echo "[setup] done. Activate with:  source $VENV/bin/activate"
echo "[setup] sanity:  python src/preflight.py        (full suite + GPU benchmark)"
