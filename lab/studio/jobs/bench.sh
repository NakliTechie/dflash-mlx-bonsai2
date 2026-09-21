#!/bin/zsh
# Leg-7 style A/B on the Studio: plain decode (mlx_lm-free: dflash generate with --verify-mode off) vs DFlash v7 round-3.
R3=~/models/Qwen3.8-27B-DFlash2-r3; PACK=~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit; OUT=lab/studio/results; mkdir -p $OUT
wrap() { printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' "$1"; }
P_CHAT=$(wrap 'Write a warm, two-paragraph email to a friend describing a weekend hike in the hills, the weather, and what you cooked afterwards.')
P_CODE=$(wrap 'Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings, type hints, and a small pytest test file at the end.')
P_MATH='Question: The function f satisfies f(x) + f(y) = f(x + y) - xy - 1 for all real x, y, and f(1) = 1. Find all integers n with f(n) = n.
Answer: Let us reason step by step.'
for rep in 1 2; do for pair in "chat:$P_CHAT" "code:$P_CODE" "math:$P_MATH"; do n=${pair%%:*}; p=${pair#*:}
  for mode in dflash off; do dflash benchmark --model $PACK --draft $R3 --only-dflash --block-tokens 8 --max-tokens 512 --repeat 1 --no-eos --verify-mode $mode --cache-limit 1GB --no-chat-template --prompt "$p" --out $OUT/bench-$n$rep-$mode > $OUT/bench-$n$rep-$mode.log 2>&1; echo "$n$rep $mode $(grep -E 'dflash generation_tps|acceptance median' $OUT/bench-$n$rep-$mode.log | awk '{print $NF}' | tr '\n' ' ')"; done; done; done
echo BENCH DONE
