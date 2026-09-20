#!/bin/bash
# Round 2 (the L40S trains ~10x faster than budgeted): full drafter fine-tune, stride 8, 2 epochs, lr 2e-5,
# initialised from round 1's best checkpoint; eval before/after; S3 sync every 10 min and at the end.
export PATH="$HOME/.local/bin:$HOME/venv/bin:$PATH"; S3=s3://skypilot-cairn-artifacts/localmind-dflash
log() { echo "[$(date '+%F %T')] $*" | tee -a ~/post_capture.log; }
cd /work/lab/aws
INIT=~/adapter-full/adapter_final.safetensors; [ -f "$INIT" ] || INIT=~/adapter/adapter_final.safetensors
log "round 2: full fine-tune stride 8, 2 epochs, init $INIT"
( while true; do sleep 600; aws s3 sync ~/adapter-r2 "$S3/adapter-r2" --only-show-errors; done ) & SYNC=$!
~/venv/bin/python train_adapter_torch.py --full --init "$INIT" --epochs 2 --stride 8 --lr 2e-5 --out ~/adapter-r2 2>&1 | grep -E 'eval|train:|step [0-9]*000/|saved|Error|Traceback' | tee -a ~/post_capture.log
kill $SYNC 2>/dev/null; aws s3 sync ~/adapter-r2 "$S3/adapter-r2" --only-show-errors; log "round 2 synced to $S3/adapter-r2"; log "ROUND2 DONE"
