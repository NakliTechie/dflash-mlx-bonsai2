#!/bin/zsh
# 100k-token cold-prefill test (MiaAI-Lab protocol) on (1) the PrismML llama.cpp fork, Metal, plain, and (2) dflash serve (MLX).
OUT=~/Code/dflash-mlx-bonsai2/lab/leg9; PY=~/Code/dflash-mlx-bonsai2/.venv/bin/python; log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/100k.log; }
MODE=${MODE:-both}
log "start: $(top -l 1 -s 0 | grep PhysMem | sed 's/PhysMem: //') swap $(sysctl -n vm.swapusage | awk '{print $6}')"
# (1) llama.cpp fork, plain decode, q8 KV to fit 110k context
BIN=~/Code/llama.cpp-prism/build-metal/bin/llama-server; M=~/Code/models/bonsai2-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf
$BIN -m $M -ngl 99 -fa on -c 110000 -b 2048 -ub 512 -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port 8793 -np 1 > $OUT/100k-llamacpp-server.log 2>&1 & SRV=$!
for i in $(seq 1 120); do curl -s --max-time 3 http://127.0.0.1:8793/v1/models >/dev/null 2>&1 && break; sleep 5; done
log "llama.cpp server up (pid $SRV): $(top -l 1 -s 0 | grep PhysMem | sed 's/PhysMem: //')"
for T in 8000 100000; do log "llama.cpp cold prefill target $T"; $PY $OUT/cold_prefill_100k.py http://127.0.0.1:8793/v1 bonsai2 $T --think-off 2>&1 | grep -vE 'transformers\]' | tee -a $OUT/100k.log; log "mem after: $(top -l 1 -s 0 | grep PhysMem | sed 's/PhysMem: //') swap $(sysctl -n vm.swapusage | awk '{print $6}')"; done
kill $SRV; sleep 5
[ "$MODE" = llamacpp ] && { log "100K DONE (llama.cpp only)"; exit 0; }
# (2) dflash serve (MLX), round-3 drafter, v7 kernel
cd ~/Code/dflash-mlx-bonsai2 && source .venv/bin/activate
DFLASH_TOOL_PARSER=off DFLASH_PRISM_VERIFY=v7 dflash serve --model ~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit --draft-model ~/Code/models/Qwen3.8-27B-DFlash2-r3 --host 127.0.0.1 --port 8790 --prefill-step-size 2048 --cache-limit 1GB --dflash-max-ctx 120000 > $OUT/100k-dflash-server.log 2>&1 & SRV=$!
for i in $(seq 1 120); do curl -s --max-time 3 http://127.0.0.1:8790/v1/models >/dev/null 2>&1 && break; sleep 5; done
log "dflash serve up (pid $SRV): $(top -l 1 -s 0 | grep PhysMem | sed 's/PhysMem: //')"
MODEL=$(curl -s http://127.0.0.1:8790/v1/models | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data'] if 'onsai' in m['id']][0])")
for T in 8000 100000; do log "dflash serve cold prefill target $T"; $PY $OUT/cold_prefill_100k.py http://127.0.0.1:8790/v1 "$MODEL" $T --think-off 2>&1 | grep -vE 'transformers\]' | tee -a $OUT/100k.log; log "mem after: $(top -l 1 -s 0 | grep PhysMem | sed 's/PhysMem: //') swap $(sysctl -n vm.swapusage | awk '{print $6}')"; done
kill $SRV; log "100K DONE"
