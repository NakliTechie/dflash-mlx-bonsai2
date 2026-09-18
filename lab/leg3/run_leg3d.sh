#!/bin/zsh
cd ~/Code/dflash-mlx-bonsai2 && source .venv/bin/activate
D=~/Code/models/Qwen3.8-27B-DFlash2; PACK=~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit
log() { echo "[$(date '+%H:%M:%S')] $*"; }
run() { name=$1; shift 1
  log "benchmark $name ($*)"
  DFLASH_PRISM_VERIFY=v4b dflash benchmark --model $PACK --draft $D --only-dflash --block-tokens 8 --max-tokens 512 --repeat 1 --no-eos --verify-mode dflash "$@" \
    --out ~/Code/dflash-mlx-bonsai2/lab/leg3/bench-$name > ~/Code/dflash-mlx-bonsai2/lab/leg3/bench-$name.log 2>&1
  log "$name exit $?"; grep -E 'generation_tps|acceptance median' ~/Code/dflash-mlx-bonsai2/lab/leg3/bench-$name.log | tail -2
}
run chat --prompt 'Write a warm, two-paragraph email to a friend describing a weekend hike in the hills, the weather, and what you cooked afterwards.'
run code --prompt 'Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings, type hints, and a small pytest test file at the end.'
run math-raw --no-chat-template --prompt 'Question: The function f satisfies f(x) + f(y) = f(x + y) - xy - 1 for all real x, y, and f(1) = 1. Find all integers n with f(n) = n.
Answer: Let us reason step by step.'
run code-raw --no-chat-template --prompt 'def lru_cache_class():
    """Return a class LRUCache(capacity) with O(1) get(key) and put(key, value), implemented with a dict and a doubly linked list."""
'
log "done"
