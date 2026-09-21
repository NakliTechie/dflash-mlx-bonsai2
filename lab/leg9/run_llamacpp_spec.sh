#!/bin/zsh
# Leg 9: llama.cpp (PrismML fork, Metal) — plain decode vs DFlash2 speculative (z-lab Q4_K_M drafter) on the
# PTQ1_0 target, same hand-templated prompts (thinking OFF) as leg 7. Prints tok/s and accept rate per run.
BIN=~/Code/llama.cpp-prism/build-metal/bin; M=~/Code/models/bonsai2-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf; MD=~/Code/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
OUT=~/Code/dflash-mlx-bonsai2/lab/leg9; N=${N:-512}
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/runner.log; }
wrap() { printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' "$1"; }
EMAIL=$(wrap 'Write a warm, two-paragraph email to a friend describing a weekend hike in the hills, the weather, and what you cooked afterwards.')
CODE=$(wrap 'Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings, type hints, and a small pytest test file at the end.')
STORY=$(wrap 'Write a complete short story of about 1500 words: a lighthouse keeper on a remote island receives a letter that changes everything. Include dialogue, a turning point, and a resolution.')
log "leg9 start: $(top -l 1 -s 0 | grep PhysMem | sed 's/PhysMem: //')"
for pair in "email:$EMAIL" "code:$CODE" "story:$STORY"; do name=${pair%%:*}; prompt=${pair#*:}
  for mode in plain dflash; do
    if [ $mode = plain ]; then extra=(); else extra=(-md $MD --spec-type draft-dflash --spec-draft-n-max 7 -ngld 999); fi
    $BIN/llama-speculative-simple -m $M "${extra[@]}" -ngl 999 -fa on -c 8192 -n $N --temp 0 -e -p "$prompt" > $OUT/$name-$mode.log 2>&1; rc=$?
    log "$name $mode exit $rc | $(grep -E 'decoded|accept|n_draft|n_accept|tokens per second|t/s' $OUT/$name-$mode.log | tail -4 | tr '\n' ' ' | cut -c1-200)"
  done
done
log "LEG9 DONE"
