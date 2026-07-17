#!/usr/bin/env bash
# Spot-aware cloud training launcher for CHARKHA.
#
# Wraps train_resilient.sh with DURABLE remote state so a run survives not just an in-process
# crash, but a whole-instance kill (spot preemption) or a credit pool running dry. It:
#   1. pulls the latest checkpoint from a remote bucket on startup,
#   2. resumes automatically if one exists (even on a brand-new box / different provider),
#   3. pushes checkpoints + metrics back to the bucket every SYNC_EVERY seconds, and
#   4. does a best-effort final push on SIGTERM (the signal spot instances get before reclaim).
#
# So: rent a spot GPU on provider A, let it train, get preempted; rent another on provider B,
# run the SAME command with the SAME REMOTE, and it picks up where it left off. Ideal for
# stitching together free student credits across GCP / Azure / Thunder / etc.
#
# Usage:
#   REMOTE=s3://my-bucket/charkha-run1 \
#   bash scripts/train_cloud.sh --data /data/corpus-dd --out /runs/model \
#       --steps 100000 --batch-size 1 --seq-len 2048 --accum-steps 8 --ce-chunk 512 \
#       --grad-checkpoint --offload-optim --val-frac 0.01 --eval-every 500 \
#       --save-every 500 --snapshot-every 20000
#
# REMOTE backends (pick what your provider/credits use):
#   s3://bucket/path   -> awscli (`aws s3 sync`)  or rclone fallback
#   gs://bucket/path   -> gcloud (`gcloud storage rsync`) or gsutil fallback
#   <rclone-remote>:path -> rclone (e.g. r2:charkha/run1, b2:charkha/run1)
#
# Env knobs: REMOTE (required)  SYNC_EVERY (sec, default 300)
#   PY (python)  TRAIN (train.py path)  RESILIENT (train_resilient.sh path)
#
set -u

REMOTE="${REMOTE:-}"
SYNC_EVERY="${SYNC_EVERY:-300}"
PY="${PY:-.venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
RESILIENT="${RESILIENT:-$HERE/train_resilient.sh}"
export PY TRAIN="${TRAIN:-/data/src/train.py}"

[ -z "$REMOTE" ] && { echo "[cloud] set REMOTE=s3://… | gs://… | rclone-remote:path"; exit 2; }

# --- locate the run dir from the passed-through train args (--out), default runs/charkha ---
args=("$@")
RUN="runs/charkha"
for i in "${!args[@]}"; do
  [ "${args[$i]}" = "--out" ] && RUN="${args[$((i+1))]}"
done

# --- choose a sync tool based on the REMOTE scheme + what's installed ---
case "$REMOTE" in
  s3://*) if command -v aws >/dev/null 2>&1; then TOOL=aws
          elif command -v rclone >/dev/null 2>&1; then TOOL=rclone
          else TOOL=""; fi ;;
  gs://*) if command -v gcloud >/dev/null 2>&1; then TOOL=gcloud
          elif command -v gsutil >/dev/null 2>&1; then TOOL=gsutil
          else TOOL=""; fi ;;
  *)      command -v rclone >/dev/null 2>&1 && TOOL=rclone || TOOL="" ;;
esac
[ -z "$TOOL" ] && { echo "[cloud] no sync tool for REMOTE=$REMOTE (need awscli / gcloud / gsutil / rclone)"; exit 2; }

push() {  # local RUN -> REMOTE
  case "$TOOL" in
    aws)    aws s3 sync "$RUN" "$REMOTE" --no-progress ;;
    gcloud) gcloud storage rsync -r "$RUN" "$REMOTE" ;;
    gsutil) gsutil -m rsync -r "$RUN" "$REMOTE" ;;
    rclone) rclone sync "$RUN" "$REMOTE" ;;
  esac
}
pull() {  # REMOTE -> local RUN (ok if remote is empty: first run)
  case "$TOOL" in
    aws)    aws s3 sync "$REMOTE" "$RUN" --no-progress ;;
    gcloud) gcloud storage rsync -r "$REMOTE" "$RUN" ;;
    gsutil) gsutil -m rsync -r "$REMOTE" "$RUN" ;;
    rclone) rclone sync "$REMOTE" "$RUN" ;;
  esac
}

mkdir -p "$RUN"
echo "[cloud] REMOTE=$REMOTE  RUN=$RUN  TOOL=$TOOL  SYNC_EVERY=${SYNC_EVERY}s"
echo "[cloud] pulling latest checkpoint (if any)…"
pull 2>/dev/null || echo "[cloud] (nothing to pull — fresh run)"

# A pulled checkpoint means we're continuing: make the FIRST launch resume too.
if [ -f "$RUN/ckpt.pt" ]; then
  case " ${args[*]} " in *" --resume "*) ;; *) args+=(--resume); echo "[cloud] found ckpt.pt -> resuming"; esac
fi

# --- background syncer: durable state every SYNC_EVERY seconds ---
syncer() { while true; do sleep "$SYNC_EVERY"; push >/dev/null 2>&1 || true; done; }
syncer & SYNC_PID=$!

cleanup() {
  echo "[cloud] shutting down — final push to $REMOTE …"
  kill "$SYNC_PID" 2>/dev/null || true
  push >/dev/null 2>&1 || true
  echo "[cloud] state saved. Re-run the same command anywhere to resume."
}
# SIGTERM is what spot instances send before reclaim; INT is Ctrl-C.
trap 'cleanup; exit 0' TERM INT

echo "[cloud] launching training…"
bash "$RESILIENT" "${args[@]}"
code=$?

cleanup
echo "[cloud] training exited with code $code"
exit $code
