#!/bin/bash
# Sync the repo and the three model artefacts to the Studio over Tailscale. Idempotent (rsync --partial).
set -euo pipefail
H=ompatnaik@100.89.200.65; R=code/chirag/dflash-mlx-bonsai2
ssh -o BatchMode=yes -o ConnectTimeout=10 $H 'mkdir -p ~/code/chirag ~/.cache/huggingface/hub ~/models' || { echo "Studio unreachable (Tailscale up on both ends?)"; exit 3; }
rsync -a --partial --exclude .git --exclude .venv --exclude '*.log' --exclude 'lab/webgpu/oracle' --exclude 'lab/webgpu/drafter/oracle' ~/Code/dflash-mlx-bonsai2/ $H:$R/
PACK=$(readlink -f ~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit)
rsync -a --partial "$PACK/" $H:.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit/
rsync -a --partial --exclude '*.gguf' ~/Code/models/Qwen3.8-27B-DFlash2-r3/ $H:models/Qwen3.8-27B-DFlash2-r3/
rsync -a --partial ~/Code/models/bonsai2-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf $H:models/
echo "synced: repo, pack (8.6 GB), drafter (3.85 GB), PTQ1_0 GGUF (5.95 GB)"
