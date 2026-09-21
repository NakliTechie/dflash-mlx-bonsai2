#!/bin/zsh
# Leg 9d: PrismML fork + ported DFlash2 (branch dflash2-port), Metal, PTQ1_0 target: plain tg128 (llama-bench),
# then DFlash2 with z-lab's drafter and the round-3 drafter on email/code/story (thinking OFF, greedy).
BIN=~/Code/llama.cpp-prism/build-metal/bin; M=~/Code/models/bonsai2-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf
MD0=~/Code/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf; MD3=~/Code/models/Qwen3.8-27B-DFlash2-r3/Qwen3.8-27B-DFlash2-r3-Q4_K_M.gguf
OUT=~/Code/dflash-mlx-bonsai2/lab/leg9; N=${N:-512}
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/runner.log; }
until ! pgrep -x llama-bench >/dev/null && ! pgrep -x llama-speculative-simple >/dev/null && ! pgrep -fx '.*dflash benchmark.*' >/dev/null; do sleep 30; done
log "leg9d start (fork $(cd ~/Code/llama.cpp-prism && git log -1 --format=%h) dflash2-port): $(top -l 1 -s 0 | grep PhysMem | sed 's/PhysMem: //')"
$BIN/llama-bench -m $M -p 512 -n 128 -ngl 99 -fa 1 -r 3 > $OUT/fork-bench-plain.log 2>&1; log "llama-bench plain PTQ1_0 exit $? | $(grep -E 'tg128|pp512' $OUT/fork-bench-plain.log | awk -F'|' '{print $(NF-2), $(NF-1)}' | tr '\n' ' ' | tr -s ' ')"
wrap() { printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' "$1"; }
EMAIL=$(wrap 'Write a warm, two-paragraph email to a friend describing a weekend hike in the hills, the weather, and what you cooked afterwards.')
CODE=$(wrap 'Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings, type hints, and a small pytest test file at the end.')
STORY=$(wrap 'Write a complete short story of about 1500 words: a lighthouse keeper on a remote island receives a letter that changes everything. Include dialogue, a turning point, and a resolution.')
for pair in "email:$EMAIL" "code:$CODE" "story:$STORY"; do name=${pair%%:*}; prompt=${pair#*:}
  for d in "zlab:$MD0" "r3:$MD3"; do tag=${d%%:*}; md=${d#*:}
    $BIN/llama-speculative-simple -m $M -md $md --spec-type draft-dflash --spec-draft-n-max 7 -ngl 99 -ngld 99 -fa on -c 8192 -n $N --temp 0 -e -p "$prompt" > $OUT/fork-$name-$tag.log 2>&1; rc=$?
    log "$name $tag exit $rc | $(grep -E 'decoded|n_drafted|n_accept|accept =|t/s' $OUT/fork-$name-$tag.log | tail -5 | sed -E 's/^[0-9.]+ I //' | tr '\n' ' ' | tr -s ' ' | cut -c1-220)"
  done
done
log "LEG9D DONE"
