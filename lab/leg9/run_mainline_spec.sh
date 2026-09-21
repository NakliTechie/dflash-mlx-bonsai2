#!/bin/zsh
# Leg 9c: mainline llama.cpp (Metal) on the Q2_0-requantized Bonsai 2: plain tg128 via llama-bench, then DFlash2
# with z-lab's drafter and with the round-3 drafter on the email/code/story prompts (thinking OFF, greedy).
BIN=~/Code/llama.cpp-dev/build-metal/bin; M=~/Code/models/bonsai2-gguf/Ternary-Bonsai-2-27B-Q2_0.gguf
MD0=~/Code/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf; MD3=~/Code/models/Qwen3.8-27B-DFlash2-r3/Qwen3.8-27B-DFlash2-r3-Q4_K_M.gguf
OUT=~/Code/dflash-mlx-bonsai2/lab/leg9; N=${N:-512}
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/runner.log; }
until [ -s $M ] && [ -s $MD3 ] && ! pgrep -f llama-quantize >/dev/null; do sleep 30; done
until ! pgrep -f 'dflash benchmark' >/dev/null; do sleep 30; done
log "leg9c start (mainline $(cd ~/Code/llama.cpp-dev && git log -1 --format=%h)): $(top -l 1 -s 0 | grep PhysMem | sed 's/PhysMem: //')"
$BIN/llama-bench -m $M -p 512 -n 128 -ngl 99 -fa 1 -r 3 > $OUT/bench-plain-q2_0.log 2>&1; log "llama-bench plain exit $? | $(grep -E 'tg128|pp512' $OUT/bench-plain-q2_0.log | awk -F'|' '{print $(NF-2), $(NF-1)}' | tr '\n' ' ' | tr -s ' ')"
wrap() { printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' "$1"; }
EMAIL=$(wrap 'Write a warm, two-paragraph email to a friend describing a weekend hike in the hills, the weather, and what you cooked afterwards.')
CODE=$(wrap 'Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings, type hints, and a small pytest test file at the end.')
STORY=$(wrap 'Write a complete short story of about 1500 words: a lighthouse keeper on a remote island receives a letter that changes everything. Include dialogue, a turning point, and a resolution.')
for pair in "email:$EMAIL" "code:$CODE" "story:$STORY"; do name=${pair%%:*}; prompt=${pair#*:}
  for d in "zlab:$MD0" "r3:$MD3"; do tag=${d%%:*}; md=${d#*:}
    $BIN/llama-speculative-simple -m $M -md $md --spec-type draft-dflash --spec-draft-n-max 7 -ngl 99 -ngld 99 -fa on -c 8192 -n $N --temp 0 -e -p "$prompt" > $OUT/ml-$name-$tag.log 2>&1; rc=$?
    log "$name $tag exit $rc | $(grep -E 'decoded|accept|n_draft|n_accept|t/s' $OUT/ml-$name-$tag.log | tail -5 | tr '\n' ' ' | tr -s ' ' | cut -c1-260)"
  done
done
log "LEG9C DONE"
