#!/bin/zsh
cd ~/Code/dflash-mlx-bonsai2 && source .venv/bin/activate
log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "capture pack"; python lab/leg6_acceptance/capture_hidden.py pack lab/leg6_acceptance/hidden_pack.npz 2>&1 | grep -v transformers | tail -2
log "capture ref";  python lab/leg6_acceptance/capture_hidden.py ref  lab/leg6_acceptance/hidden_ref.npz  2>&1 | grep -v transformers | tail -2
log "compare";      python lab/leg6_acceptance/compare_hidden.py 2>&1 | tail -12
log "done"
