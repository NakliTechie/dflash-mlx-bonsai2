#!/usr/bin/env bash
# infra/aws Mandate 13: while the box exists, poll on a <=12-minute interval; log status, GPU, jobs, disk;
# alert (stdout + log) on DOWN/INIT-stuck/idle-with-nothing-running.  bash lab/aws/watchdog.sh [interval_s]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
export AWS_PROFILE="${AWS_PROFILE_LAUNCH:-cairn-skypilot}"; CLUSTER="${CLUSTER:-localmind-dflash}"; INT="${1:-600}"
SKY="${SKY:-$HOME/.cairn-sky-venv/bin/sky}"; LOG=lab/aws/watchdog-log.txt; START=$(date +%s); CAP_H="${CAP_HOURS:-8}"   # hard cost cap: ~$0.89/h x 14 h < $13
while true; do
  st="$("$SKY" status "$CLUSTER" 2>/dev/null | grep -E "^$CLUSTER" | grep -oE '\b(UP|INIT|STOPPED|AUTOSTOPPING)\b' | head -1 || true)"
  q="$("$SKY" queue "$CLUSTER" 2>/dev/null | grep -c -E 'RUNNING|PENDING|SETTING_UP' || echo 0)"
  gpu="$(timeout 40 ssh -o ConnectTimeout=10 "$CLUSTER" 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h / | tail -1 | awk "{print \$4\" free\"}"' 2>/dev/null | tr '\n' ' ' || echo 'ssh-fail')"
  echo "[$(date '+%F %T')] status=${st:-GONE} jobs=$q gpu=[$gpu]" | tee -a "$LOG"
  if [ $(( ($(date +%s) - START) / 3600 )) -ge "$CAP_H" ]; then echo "[watchdog] wall-clock cap ${CAP_H}h reached; "$SKY" down $CLUSTER" | tee -a "$LOG"; "$SKY" down "$CLUSTER" -y >> "$LOG" 2>&1; exit 0; fi
  case "${st:-GONE}" in GONE|STOPPED) echo "[watchdog] cluster $CLUSTER is ${st:-gone}; exiting" | tee -a "$LOG"; exit 0;; esac
  sleep "$INT"
done
