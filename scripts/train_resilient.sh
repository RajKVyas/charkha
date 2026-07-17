#!/usr/bin/env bash
# Resilient CHARKHA training launcher (WSL/GPU box).
# Starts training and auto-restarts with --resume on ANY non-zero exit (crash, CUDA OOM,
# driver reset, power blip). The first run starts fresh; every restart appends --resume so it
# picks up <out>/ckpt.pt at the last saved step. A clean finish (exit 0) or Ctrl-C (130) stops.
#
# Usage (pass the normal train.py args through):
#   bash /data/scripts/train_resilient.sh --data /data/data/wiki --data /data/data/sc \
#       --out runs/model --steps 200000 --batch-size 2 --seq-len 4096 --accum-steps 4 \
#       --grad-checkpoint --val-frac 0.01 --eval-every 500 --save-every 500 --snapshot-every 10000 \
#       --sample-every 500 --sample-prompt "The "
#
# Env knobs:  MAX_RESTARTS (default 1000)   RESTART_DELAY seconds (default 10)
#   PY (default: .venv/bin/python)   TRAIN (default: /data/src/train.py)
set -u
PY="${PY:-.venv/bin/python}"
TRAIN="${TRAIN:-/data/src/train.py}"
MAX_RESTARTS="${MAX_RESTARTS:-1000}"
RESTART_DELAY="${RESTART_DELAY:-10}"
args=("$@")
# Enforce 8GB VRAM limits (Batch Size 1, 8-bit Optimizers, Grad Checkpointing)
case " ${args[*]} " in *" --batch-size "*) ;; *) args+=(--batch-size 1) ;; esac
case " ${args[*]} " in *" --8bit-optim "*) ;; *) args+=(--8bit-optim) ;; esac
case " ${args[*]} " in *" --grad-checkpoint "*) ;; *) args+=(--grad-checkpoint) ;; esac
i=0
while :; do
  echo "[launcher] launch (restart $i): $PY $TRAIN ${args[*]}"
  "$PY" "$TRAIN" "${args[@]}"
  code=$?
  if [ $code -eq 0 ]; then echo "[launcher] training finished cleanly (exit 0)."; break; fi
  if [ $code -eq 130 ]; then echo "[launcher] interrupted (Ctrl-C); not restarting."; break; fi
  i=$((i + 1))
  if [ $i -ge "$MAX_RESTARTS" ]; then
    echo "[launcher] hit MAX_RESTARTS=$MAX_RESTARTS after exit $code; giving up."; exit $code
  fi
  echo "[launcher] crash (exit $code); restart #$i in ${RESTART_DELAY}s with --resume ..."
  sleep "$RESTART_DELAY"
  # ensure --resume is present exactly once for subsequent attempts
  case " ${args[*]} " in *" --resume "*) ;; *) args+=(--resume) ;; esac
done
