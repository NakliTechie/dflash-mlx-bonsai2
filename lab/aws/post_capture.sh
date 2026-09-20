#!/bin/bash
# After the capture: dequantize the pack's embed/lm_head, baseline eval, fc+hidden_norm re-fit, eval, S3 upload.
export PATH="$HOME/.local/bin:$HOME/venv/bin:$PATH"; S3=s3://skypilot-cairn-artifacts/localmind-dflash
log() { echo "[$(date '+%F %T')] $*" | tee -a ~/post_capture.log; }
cd /work/lab/aws
[ -f ~/models/pack_embed_lmhead.safetensors ] || { log "dequant pack embed/lm_head"; ~/venv/bin/python dequant_pack.py ~/models/bonsai2-mlx-pack ~/models/pack_embed_lmhead.safetensors 2>&1 | tail -3 | tee -a ~/post_capture.log; }
log "baseline eval (all shards)"; ~/venv/bin/python train_adapter_torch.py --eval-only --out ~/adapter 2>&1 | grep -E 'eval|shards|drafter|Error|Traceback' | tee -a ~/post_capture.log
log "train fc+hidden_norm 1 epoch"; ~/venv/bin/python train_adapter_torch.py --epochs 1 --out ~/adapter 2>&1 | grep -E 'eval|train:|step [0-9]*00/|saved|Error|Traceback' | tee -a ~/post_capture.log
aws s3 sync ~/adapter "$S3/adapter" --only-show-errors; log "adapter synced to $S3/adapter"
log "POST_CAPTURE DONE"
