#!/bin/bash
# Round 2: full drafter fine-tune on teacher-forced general + chat shards, stride 8, 1 epoch (fits the box cap), lr 2e-5,
# initialised from round 1's best checkpoint; eval before/after; S3 sync every 10 min and at the end.
export PATH="$HOME/.local/bin:$HOME/venv/bin:$PATH"; S3=s3://skypilot-cairn-artifacts/localmind-dflash
log() { echo "[$(date '+%F %T')] $*" | tee -a ~/post_capture.log; }
cd /work/lab/aws
INIT=~/adapter-full/adapter_final.safetensors; [ -f "$INIT" ] || INIT=~/adapter/adapter_final.safetensors
log "round 2: full fine-tune on general + chat shards, stride 8, 1 epoch, init $INIT"
( while true; do sleep 600; aws s3 sync ~/adapter-r2 "$S3/adapter-r2" --only-show-errors; done ) & SYNC=$!
~/venv/bin/python train_adapter_torch.py --full --init "$INIT" --shards ~/shards,~/shards_chat --epochs 1 --stride 8 --lr 2e-5 --out ~/adapter-r2 2>&1 | grep -E 'eval|train:|step [0-9]*000/|saved|Error|Traceback' | tee -a ~/post_capture.log
kill $SYNC 2>/dev/null
# same proxy as round 3: score the round-2 drafter on held-out GENERATED blocks
log "round 2: gen-only eval on ~/shards_gen"
~/venv/bin/python train_adapter_torch.py --eval-only --gen-only --init ~/adapter-r2/adapter_final.safetensors --shards ~/shards_gen --window 160 --eval-docs 60 --out ~/adapter-r2/eval-gen 2>&1 | grep -E 'eval|Error|Traceback' | tee -a ~/post_capture.log
aws s3 sync ~/adapter-r2 "$S3/adapter-r2" --only-show-errors; log "round 2 synced to $S3/adapter-r2"; log "ROUND2 DONE"
