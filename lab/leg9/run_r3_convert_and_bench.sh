#!/bin/zsh
# After leg 9: convert round 3 -> GGUF (bf16) -> Q4_K_M, then rerun the three DFlash2 prompts with the r3 drafter.
OUT=~/Code/dflash-mlx-bonsai2/lab/leg9; BIN=~/Code/llama.cpp-prism/build-metal/bin; M=~/Code/models/bonsai2-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf
R3DIR=~/Code/models/Qwen3.8-27B-DFlash2-r3; REF=~/Code/models/Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/runner.log; }
until grep -q 'LEG9 DONE' $OUT/runner.log 2>/dev/null; do sleep 30; done
log "converting round-3 drafter to GGUF"
~/Code/llama.cpp-prism/.venv/bin/python $OUT/r3_to_gguf.py $REF $R3DIR $R3DIR/r3-bf16.gguf 2>&1 | tail -3 | tee -a $OUT/runner.log
$BIN/llama-quantize $R3DIR/r3-bf16.gguf $R3DIR/Qwen3.8-27B-DFlash2-r3-Q4_K_M.gguf Q4_K_M > $OUT/quantize.log 2>&1; log "quantize exit $? : $(ls -la $R3DIR/Qwen3.8-27B-DFlash2-r3-Q4_K_M.gguf 2>/dev/null | awk '{printf "%.2f GB", $5/1e9}')"
MD=$R3DIR/Qwen3.8-27B-DFlash2-r3-Q4_K_M.gguf
wrap() { printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' "$1"; }
EMAIL=$(wrap 'Write a warm, two-paragraph email to a friend describing a weekend hike in the hills, the weather, and what you cooked afterwards.')
CODE=$(wrap 'Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings, type hints, and a small pytest test file at the end.')
STORY=$(wrap 'Write a complete short story of about 1500 words: a lighthouse keeper on a remote island receives a letter that changes everything. Include dialogue, a turning point, and a resolution.')
for pair in "email:$EMAIL" "code:$CODE" "story:$STORY"; do name=${pair%%:*}; prompt=${pair#*:}
  $BIN/llama-speculative-simple -m $M -md $MD --spec-type draft-dflash --spec-draft-n-max 7 -ngld 999 -ngl 999 -fa on -c 8192 -n ${N:-512} --temp 0 -e -p "$prompt" > $OUT/$name-dflash-r3.log 2>&1; rc=$?
  log "$name dflash-r3 exit $rc | $(grep -E 'decoded|accept|n_draft|n_accept|tokens per second|t/s' $OUT/$name-dflash-r3.log | tail -4 | tr '\n' ' ' | cut -c1-200)"
done
log "LEG9B DONE"
