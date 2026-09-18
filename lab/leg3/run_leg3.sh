#!/bin/zsh
# Unattended leg 3: wait for the drafter, verify sha256, then benchmark DFlash2 on the pack in two modes.
cd ~/Code/dflash-mlx-bonsai2 && source .venv/bin/activate
D=~/Code/models/Qwen3.8-27B-DFlash2; PACK=~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit
WANT=67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c
log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "waiting for drafter"
while pgrep -f 'fetch.sh' >/dev/null; do sleep 60; done
log "fetch.sh finished: $(tail -2 /tmp/fetch.log | tr '\n' ' ')"
[ -f $D/model.safetensors ] || { log "no model.safetensors"; exit 1; }
HAVE=$(shasum -a 256 $D/model.safetensors | cut -d' ' -f1)
if [ "$HAVE" != "$WANT" ]; then log "CHECKSUM MISMATCH $HAVE"; exit 1; fi
log "checksum ok; removing parts"; rm -f $D/part* $D/fetch.sh
for mode in v4b off; do
  log "benchmark mode=$mode"
  DFLASH_PRISM_VERIFY=$mode dflash benchmark --model $PACK --draft $D --block-tokens 8 --max-tokens 512 --repeat 2 --no-eos \
    --out ~/Code/dflash-mlx-bonsai2/lab/leg3/bench-$mode > ~/Code/dflash-mlx-bonsai2/lab/leg3/bench-$mode.log 2>&1
  log "mode=$mode exit $? ; tail:"; tail -25 ~/Code/dflash-mlx-bonsai2/lab/leg3/bench-$mode.log
done
log "done"
