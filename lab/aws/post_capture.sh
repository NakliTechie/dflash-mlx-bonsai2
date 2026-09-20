#!/bin/bash
# After the capture: dequantize the pack's embed/lm_head, baseline eval, fc+hidden_norm re-fit, eval, S3 upload.
export PATH="$HOME/.local/bin:$HOME/venv/bin:$PATH"; S3=s3://skypilot-cairn-artifacts/localmind-dflash
log() { echo "[$(date '+%F %T')] $*" | tee -a ~/post_capture.log; }
cd /work/lab/aws
[ -f ~/models/pack_embed_lmhead.safetensors ] || { log "dequant pack embed/lm_head"; ~/venv/bin/python dequant_pack.py ~/models/bonsai2-mlx-pack ~/models/pack_embed_lmhead.safetensors 2>&1 | tail -3 | tee -a ~/post_capture.log; }
log "baseline eval (all shards)"; ~/venv/bin/python train_adapter_torch.py --eval-only --out ~/adapter 2>&1 | grep -E 'eval|shards|drafter|Error|Traceback' | tee -a ~/post_capture.log
( while true; do sleep 600; aws s3 sync ~/adapter "$S3/adapter" --only-show-errors; done ) & SYNC=$!
log "train fc+hidden_norm 1 epoch (stride 32, ~2 h on L40S)"; ~/venv/bin/python train_adapter_torch.py --epochs 1 --stride 32 --out ~/adapter 2>&1 | grep -E 'eval|train:|step [0-9]*00/|saved|Error|Traceback' | tee -a ~/post_capture.log
kill $SYNC 2>/dev/null; aws s3 sync ~/adapter "$S3/adapter" --only-show-errors; log "adapter synced to $S3/adapter"
# Chirag 2026-09-20 20:55: time can be extended -> continue into the full drafter fine-tune from the fitted adapter.
( while true; do sleep 600; aws s3 sync ~/adapter-full "$S3/adapter-full" --only-show-errors; done ) & SYNC2=$!
log "full drafter fine-tune 1 epoch (stride 32, lr 2e-5, init from adapter_final)"; ~/venv/bin/python train_adapter_torch.py --full --init ~/adapter/adapter_final.safetensors --epochs 1 --stride 32 --lr 2e-5 --out ~/adapter-full 2>&1 | grep -E 'eval|train:|step [0-9]*00/|saved|Error|Traceback|mem' | tee -a ~/post_capture.log
kill $SYNC2 2>/dev/null; aws s3 sync ~/adapter-full "$S3/adapter-full" --only-show-errors; log "full fine-tune synced to $S3/adapter-full"
log "POST_CAPTURE DONE"
