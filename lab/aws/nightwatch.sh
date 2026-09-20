#!/usr/bin/env bash
# Local durable driver: every INT seconds — "$SKY" status; if the cluster is gone (spot reclaim) relaunch it and
# re-submit the capture job (which resumes from S3); if it is UP with no job running and the capture is not
# done, submit the job. Hard wall-clock cap on total box hours. Logs to lab/aws/nightwatch-log.txt.
#   CONFIRM_GPU_SPEND=1 nohup bash lab/aws/nightwatch.sh 720 > /dev/null 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
export AWS_PROFILE="${AWS_PROFILE_LAUNCH:-cairn-skypilot}"; CLUSTER="${CLUSTER:-localmind-dflash}"; INT="${1:-720}"; CAP_H="${CAP_HOURS:-8}"
SKY="${SKY:-$HOME/.cairn-sky-venv/bin/sky}"; LOG=lab/aws/nightwatch-log.txt; START=$(date +%s); BUDGET="${CAPTURE_BUDGET:-4000000}"; S3="s3://skypilot-cairn-artifacts/localmind-dflash"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
while true; do
  hrs=$(( ($(date +%s) - START) / 3600 )); if [ "$hrs" -ge "$CAP_H" ]; then say "wall-clock cap ${CAP_H}h: "$SKY" down"; "$SKY" down "$CLUSTER" -y >> "$LOG" 2>&1; exit 0; fi
  st="$("$SKY" status "$CLUSTER" 2>/dev/null | grep -E "^$CLUSTER" | grep -oE '\b(UP|INIT|STOPPED|AUTOSTOPPING)\b' | head -1 || true)"
  done_flag="$(aws s3 cp "$S3/shards/progress.json" - 2>/dev/null | python3 -c 'import json,sys; d=json.load(sys.stdin); print("done" if d.get("done") else d.get("tokens",0))' 2>/dev/null || echo 0)"
  case "${st:-GONE}" in
    UP)
      jobs_running="$("$SKY" queue "$CLUSTER" 2>/dev/null | grep -c -E 'RUNNING|PENDING|SETTING_UP' || echo 0)"
      gpu="$(timeout 40 ssh -o ConnectTimeout=10 "$CLUSTER" 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null; df -h / | tail -1 | awk "{print \$4}"; tail -1 ~/capture_job.log 2>/dev/null' 2>/dev/null | tr '\n' ' ' || echo ssh-fail)"
      say "UP jobs=$jobs_running s3_progress=$done_flag gpu=[$gpu]"
      if [ "$jobs_running" = "0" ] && [ "$done_flag" != "done" ]; then say "no job running and capture not done: submitting capture_job"; "$SKY" exec "$CLUSTER" -d "bash /work/lab/aws/capture_job.sh $BUDGET" >> "$LOG" 2>&1; fi ;;
    INIT) say "INIT (provisioning/setup)";;
    GONE|STOPPED|"")
      if [ "$done_flag" = "done" ]; then say "cluster ${st:-gone} and capture done; nothing to do"; else say "cluster ${st:-gone} and capture at $done_flag tokens: relaunching"; "$SKY" down "$CLUSTER" -y --purge >> "$LOG" 2>&1 || true; CONFIRM_GPU_SPEND=1 bash lab/aws/launch.sh >> "$LOG" 2>&1 || say "relaunch failed"; fi ;;
    *) say "status=$st";;
  esac
  sleep "$INT"
done
