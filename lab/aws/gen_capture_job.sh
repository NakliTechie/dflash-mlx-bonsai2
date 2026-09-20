#!/bin/bash
# Job: self-generation capture. Build the prompt set (once), then tap-dump --gen 512 with 16 sequences in flight,
# shards to ~/shards_gen and S3 shards_gen/. Run under sky exec --gpus L40S:0.5 (fits beside a training job).
export PATH="$HOME/.local/bin:$HOME/venv/bin:$PATH"
log() { echo "[$(date '+%F %T')] $*" | tee -a ~/capture_job.log; }
P=/work/lab/aws/corpus/gen_prompts.jsonl
if [ ! -s "$P" ]; then log "building gen prompts"; ~/venv/bin/python /work/lab/aws/build_gen_prompts.py "$P" 2>&1 | grep -E 'wrote|sample|Error|Traceback' | tee -a ~/capture_job.log; fi
[ -s "$P" ] || { log "no prompt file; abort"; exit 1; }
GEN=512 SEQS=16 PROMPT_MAX=768 OUT=~/shards_gen CORPUS="$P" S3SUB=shards_gen bash /work/lab/aws/capture_job.sh 2500000
