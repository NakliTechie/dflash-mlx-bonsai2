#!/bin/zsh
cd ~/Code/dflash-mlx-bonsai2 && source .venv/bin/activate
D=~/Code/models/Qwen3.8-27B-DFlash2; PACK=~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit
log() { echo "[$(date '+%H:%M:%S')] $*"; }
while pgrep -f run_leg3b.sh >/dev/null; do sleep 15; done
run() { name=$1; mode=$2; shift 2
  log "benchmark $name (DFLASH_PRISM_VERIFY=$mode $*)"
  DFLASH_PRISM_VERIFY=$mode dflash benchmark --model $PACK --draft $D --only-dflash --block-tokens 8 --max-tokens 512 --repeat 2 --no-eos "$@" \
    --out ~/Code/dflash-mlx-bonsai2/lab/leg3/bench-$name > ~/Code/dflash-mlx-bonsai2/lab/leg3/bench-$name.log 2>&1
  log "$name exit $?"; grep -E 'generation_tps|acceptance|peak memory' ~/Code/dflash-mlx-bonsai2/lab/leg3/bench-$name.log | tail -4
}
run fixed8-v4b v4b --verify-mode dflash
run bf16draft-v4b v4b --draft-quant none
run fixed8-bf16draft-v4b v4b --verify-mode dflash --draft-quant none
log "done"
