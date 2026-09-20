#!/bin/zsh
cd ~/Code/dflash-mlx-bonsai2 && source .venv/bin/activate
D=~/Code/models/bonsai2-drafter-data
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $D/train.log; }
log "waiting for capture to finish"
while pgrep -f capture_corpus >/dev/null; do sleep 30; done
log "capture finished: $(tail -1 $D/capture.log)"
log "smoke: baseline eval on 2 shards"
python lab/leg6_acceptance/train_adapter.py --shards 2 --eval-only --eval-docs 8 2>&1 | grep -v transformers | tail -4 | tee -a $D/train.out
log "train: 1 epoch over all shards"
python lab/leg6_acceptance/train_adapter.py --epochs 1 2>&1 | grep -v transformers | tee -a $D/train.out | grep -E 'eval|train:|step [0-9]*00/|saved|checkpoint|Error|Traceback' 
log "done"
