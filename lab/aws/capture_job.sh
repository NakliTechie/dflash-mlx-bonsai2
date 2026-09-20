#!/bin/bash
# Box-side, durable capture: pull prior shards from S3, resume from progress.json, run tap-dump, push each new
# shard to S3 in a 5-minute sync loop. Idempotent: re-running after a reclaim continues where S3 left off.
#   bash /work/lab/aws/capture_job.sh [budget_tokens]
set -uo pipefail
export PATH="$HOME/.local/bin:$HOME/venv/bin:/usr/local/cuda/bin:$PATH"; export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"; export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
BUDGET="${1:-4000000}"; S3="s3://skypilot-cairn-artifacts/localmind-dflash"; OUT=~/shards; CORPUS=/work/lab/aws/corpus/mix.jsonl
MODEL=~/models/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf; BIN=~/llama.cpp/build/bin/llama-tap-dump
log() { echo "[$(date '+%F %T')] $*" | tee -a ~/capture_job.log; }
mkdir -p "$OUT"; log "pull prior shards from $S3/shards"; aws s3 sync "$S3/shards" "$OUT" --only-show-errors
SD=0; SS=0; if [ -f "$OUT/progress.json" ]; then SD=$(python3 -c "import json;print(json.load(open('$OUT/progress.json'))['next_doc'])"); SS=$(python3 -c "import json;print(json.load(open('$OUT/progress.json'))['next_shard'])"); fi
if python3 -c "import json,sys; sys.exit(0 if json.load(open('$OUT/progress.json')).get('done') else 1)" 2>/dev/null; then log "capture already complete per progress.json"; exit 0; fi
log "resume at doc $SD shard $SS (budget $BUDGET)"
( while true; do sleep 300; aws s3 sync "$OUT" "$S3/shards" --only-show-errors && echo "[$(date '+%F %T')] s3 sync ok ($(ls $OUT | grep -c '\.json$') shard headers)" >> ~/capture_job.log; done ) & SYNC=$!
"$BIN" -m "$MODEL" --corpus "$CORPUS" --out "$OUT" --taps 5,19,33,47,61 --budget "$BUDGET" --start-doc "$SD" --start-shard "$SS" -c 2048 -b 512 -ub 512 -ngl 99 -fa on 2>&1 | grep --line-buffered -E 'shard|done|doc [0-9]+/|error|failed|offloaded|CUDA|ggml_backend|device' | tee -a ~/capture_job.log
RC=${PIPESTATUS[0]}; kill $SYNC 2>/dev/null; aws s3 sync "$OUT" "$S3/shards" --only-show-errors; log "tap-dump exit $RC; final s3 sync done"; exit $RC
