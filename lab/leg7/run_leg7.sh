#!/bin/zsh
# Leg 7: live acceptance with the round-3 drafter (re-fit on the ternary model's own generations) vs the shipped
# z-lab drafter, same prompts as leg 3d, same session, back to back. Waits for the S3 download to assemble first.
cd ~/Code/dflash-mlx-bonsai2 && source .venv/bin/activate
R3=~/Code/models/Qwen3.8-27B-DFlash2-r3; D0=~/Code/models/Qwen3.8-27B-DFlash2; PACK=~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit
OUT=~/Code/dflash-mlx-bonsai2/lab/leg7
log() { echo "[$(date '+%H:%M:%S')] $*"; }
while [ ! -f $R3/model.safetensors ] || [ $(stat -f %z $R3/model.safetensors) -ne 3848817904 ]; do sleep 20; done
log "r3 file present: $(stat -f %z $R3/model.safetensors) bytes; sha256 $(shasum -a 256 $R3/model.safetensors | cut -c1-16)…"
log "swap before: $(sysctl -n vm.swapusage)"
run() { name=$1; draft=$2; shift 2
  log "benchmark $name ($draft) $*"
  DFLASH_PRISM_VERIFY=v4b dflash benchmark --model $PACK --draft $draft --only-dflash --block-tokens 8 --max-tokens 512 --repeat 1 --no-eos --verify-mode dflash --cache-limit 1GB --wired-limit none "$@" \
    --out $OUT/bench-$name > $OUT/bench-$name.log 2>&1
  log "$name exit $? | $(grep -E 'dflash generation_tps|acceptance median' $OUT/bench-$name.log | awk '{print $NF}' | tr '\n' ' ') | swap $(sysctl -n vm.swapusage | awk '{print $6}')"
}
CHAT='Write a warm, two-paragraph email to a friend describing a weekend hike in the hills, the weather, and what you cooked afterwards.'
CODE='Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings, type hints, and a small pytest test file at the end.'
MATH='Question: The function f satisfies f(x) + f(y) = f(x + y) - xy - 1 for all real x, y, and f(1) = 1. Find all integers n with f(n) = n.
Answer: Let us reason step by step.'
CODERAW='def lru_cache_class():
    """Return a class LRUCache(capacity) with O(1) get(key) and put(key, value), implemented with a dict and a doubly linked list."""
'
for pair in "r3:$R3" "z0:$D0"; do tag=${pair%%:*}; draft=${pair#*:}
  run chat-$tag $draft --prompt "$CHAT"
  run code-$tag $draft --prompt "$CODE"
  run math-raw-$tag $draft --no-chat-template --prompt "$MATH"
  run code-raw-$tag $draft --no-chat-template --prompt "$CODERAW"
done
log "swap after: $(sysctl -n vm.swapusage)"
log "LEG7 DONE"
