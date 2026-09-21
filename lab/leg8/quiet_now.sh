#!/bin/zsh
# Wait for a quiet window (unused RAM > 10 GB, compressor < 1.5 GB, CPU idle > 85 %) then run the v7 vs v4b
# A/B (code 1024 + story 2048, both kernels, twice) with the round-3 drafter, thinking OFF.
cd ~/Code/dflash-mlx-bonsai2 && source .venv/bin/activate
R3=~/Code/models/Qwen3.8-27B-DFlash2-r3; PACK=~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit; OUT=lab/leg8/quiet; mkdir -p $OUT
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/runner.log; }
quiet() { local pm=$(top -l 2 -s 1 | tail -20); local unused=$(echo "$pm" | grep PhysMem | sed -E 's/.*, ([0-9]+)([MG]) unused.*/\1 \2/' | awk '{print ($2=="G")?$1*1024:$1}'); local comp=$(echo "$pm" | grep PhysMem | sed -E 's/.*\(.*, ([0-9]+)M compressor\).*/\1/'); local idle=$(echo "$pm" | grep 'CPU usage' | tail -1 | sed -E 's/.*, ([0-9.]+)% idle.*/\1/'); echo "unused ${unused}M comp ${comp}M idle ${idle}%"; ! pgrep -f "llama-speculative-simple|llama-server|llama-bench" >/dev/null && [ "${unused:-0}" -gt 9000 ] && [ "${comp:-9999}" -lt 4000 ] && [ "${idle%.*}" -gt 80 ]; }
log "watching for a quiet window"
true
log "quiet window: $(quiet)"
wrap() { printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' "$1"; }
STORY=$(wrap 'Write a complete short story of about 1500 words: a lighthouse keeper on a remote island receives a letter that changes everything. Include dialogue, a turning point, and a resolution.')
CODE=$(wrap 'Write a Python module with a class LRUCache(capacity) supporting get(key) and put(key, value) in O(1), with docstrings, type hints, and a small pytest test file at the end.')
run() { name=$1; mode=$2; maxt=$3; prompt=$4; DFLASH_PRISM_VERIFY=$mode dflash benchmark --model $PACK --draft $R3 --only-dflash --block-tokens 8 --max-tokens $maxt --repeat 1 --verify-mode dflash --cache-limit 1GB --wired-limit none --no-chat-template --prompt "$prompt" --out $OUT/$name-$mode > $OUT/$name-$mode.log 2>&1; log "$name $mode exit $? | $(grep -E 'dflash generation_tps' $OUT/$name-$mode.log | awk '{print $NF}') tok/s | $(quiet)"; }
for rep in 1 2; do for mode in v7 v4b; do run code$rep $mode 1024 "$CODE"; run story$rep $mode 2048 "$STORY"; done; done
python3 - <<'PY' | tee -a $OUT/runner.log
import json
print(f"{'run':8s} {'mode':5s} {'tok/cycle':>9s} {'tok/s':>6s} {'ms/cycle':>8s}")
for n in ['code1','story1','code2','story2']:
    for m in ['v7','v4b']:
        x = json.loads(open(f'lab/leg8/quiet/{n}-{m}/runs.jsonl').readline())['dflash']; gen = round(x['tokens_per_cycle']*x['cycles'])
        print(f"{n:8s} {m:5s} {x['tokens_per_cycle']:9.2f} {x['generation_tps']:6.1f} {gen/x['generation_tps']*1000/x['cycles']:8.0f}")
PY
log "QUIET A/B DONE"
