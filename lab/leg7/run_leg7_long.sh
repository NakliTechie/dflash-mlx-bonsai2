#!/bin/zsh
# Leg 7b: long generations (up to 2048 new tokens, natural EOS) and one long-input prompt, round-3 vs shipped
# drafter. Greedy verification makes both drafters emit identical text, so lengths match and only cycles differ.
cd ~/Code/dflash-mlx-bonsai2 && source .venv/bin/activate
R3=~/Code/models/Qwen3.8-27B-DFlash2-r3; D0=~/Code/models/Qwen3.8-27B-DFlash2; PACK=~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit
OUT=~/Code/dflash-mlx-bonsai2/lab/leg7
log() { echo "[$(date '+%H:%M:%S')] $*"; }
while pgrep -f 'run_leg7.sh' >/dev/null; do sleep 15; done
log "long runs start: $(top -l 1 -s 0 | grep PhysMem | sed 's/PhysMem: //') swap $(sysctl -n vm.swapusage | awk '{print $6}')"
run() { name=$1; draft=$2; shift 2
  log "benchmark $name ($draft)"
  DFLASH_PRISM_VERIFY=v4b dflash benchmark --model $PACK --draft $draft --only-dflash --block-tokens 8 --max-tokens 2048 --repeat 1 --verify-mode dflash --cache-limit 1GB --wired-limit none "$@" \
    --out $OUT/long-$name > $OUT/long-$name.log 2>&1
  log "$name exit $? | $(grep -E 'dflash generation_tps|acceptance median' $OUT/long-$name.log | awk '{print $NF}' | tr '\n' ' ') | swap $(sysctl -n vm.swapusage | awk '{print $6}')"
}
STORY='Write a complete short story of about 1500 words: a lighthouse keeper on a remote island receives a letter that changes everything. Include dialogue, a turning point, and a resolution.'
MODULE='Write a complete, production-quality Python package for a small task-queue: a Task dataclass, a priority scheduler with retries and exponential backoff, a SQLite-backed persistence layer, a CLI using argparse, and a pytest suite. Include full docstrings and type hints for every public function.'
DOC="Summarise the following engineering notes as 12 concise bullet points, then list the three biggest open risks.

$(head -c 6000 ~/Code/naklios-universe/LocalMind/plan/2026-09-18-dflash-bonsai2-plan.md)"
for pair in "r3:$R3" "z0:$D0"; do tag=${pair%%:*}; draft=${pair#*:}
  run story-$tag $draft --prompt "$STORY"
  run module-$tag $draft --prompt "$MODULE"
  run doc-$tag $draft --prompt "$DOC"
done
log "LEG7-LONG DONE"
