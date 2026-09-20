#!/bin/zsh
cd ~/Code/dflash-mlx-bonsai2 && source .venv/bin/activate
D=~/Code/models/Qwen3.8-27B-DFlash2; PACK=~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit
P='Write a detailed, well-structured essay of about 800 words on how speculative decoding works, why block-diffusion drafters help, and what limits acceptance on quantized targets. Use headings and end with a short summary.'
log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "dflash generate (spec, v4b)"
DFLASH_PRISM_VERIFY=v4b dflash generate --model $PACK --draft $D --max-tokens 1024 --prompt "$P" > lab/leg4/spec.txt 2> lab/leg4/spec.err; log "spec exit $?"
log "plain greedy"
python lab/leg4/plain_generate.py "$P" 1024 lab/leg4/plain.json 2> lab/leg4/plain.err | tail -1
log "done"
