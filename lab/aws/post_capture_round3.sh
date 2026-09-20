#!/bin/bash
# Round 3: full fine-tune on the model's OWN generations (gen-only blocks), init from round 2 (or round 1),
# eval before/after on held-out generated blocks — the proxy that matches what greedy verification sees.
export PATH="$HOME/.local/bin:$HOME/venv/bin:$PATH"; S3=s3://skypilot-cairn-artifacts/localmind-dflash
log() { echo "[$(date '+%F %T')] $*" | tee -a ~/post_capture.log; }
cd /work/lab/aws
INIT=~/adapter-r2/adapter_final.safetensors; [ -f "$INIT" ] || INIT=~/adapter-full/adapter_final.safetensors
log "round 3: full fine-tune on generated blocks (gen-only), stride 4, 2 epochs, init $INIT"
( while true; do sleep 600; aws s3 sync ~/adapter-r3 "$S3/adapter-r3" --only-show-errors; done ) & SYNC=$!
~/venv/bin/python train_adapter_torch.py --full --gen-only --init "$INIT" --shards ~/shards_gen --epochs 2 --stride 4 --lr 2e-5 --eval-docs 60 --out ~/adapter-r3 2>&1 | grep -E 'eval|train:|shards in|step [0-9]*000/|saved|Error|Traceback' | tee -a ~/post_capture.log
kill $SYNC 2>/dev/null; aws s3 sync ~/adapter-r3 "$S3/adapter-r3" --only-show-errors; log "round 3 synced to $S3/adapter-r3"; log "ROUND3 DONE"
