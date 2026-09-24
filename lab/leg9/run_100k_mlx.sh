#!/bin/zsh
# MLX half of the 100k cold-prefill test: waits until no Chrome holds the GPU and >= 12 GB are unused, then runs
# dflash serve (round-3 drafter, v7) and the protocol at 8k and 100k tokens. Holds a marker file so agents wait.
OUT=~/Code/dflash-mlx-bonsai2/lab/leg9; PY=~/Code/dflash-mlx-bonsai2/.venv/bin/python; log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a $OUT/100k.log; }
unused() { top -l 1 -s 0 | grep PhysMem | sed -E 's/.*, ([0-9]+)([MG]) unused.*/\1 \2/' | awk '{print ($2=="G")?$1*1024:$1}'; }
until ! pgrep -f 'user-data-dir=.*chrome-profile' >/dev/null && [ "$(unused)" -gt 12000 ]; do sleep 60; done
cd ~/Code/dflash-mlx-bonsai2 && source .venv/bin/activate
log "MLX 100k start: $(top -l 1 -s 0 | grep PhysMem | sed 's/PhysMem: //')"
DFLASH_TOOL_PARSER=off DFLASH_PRISM_VERIFY=v7 dflash serve --model ~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit --draft-model ~/Code/models/Qwen3.8-27B-DFlash2-r3 --host 127.0.0.1 --port 8790 --prefill-step-size 512 --cache-limit 1GB --dflash-max-ctx 40000 > $OUT/100k-dflash-server.log 2>&1 & SRV=$!
for i in $(seq 1 120); do curl -s --max-time 3 http://127.0.0.1:8790/v1/models >/dev/null 2>&1 && break; sleep 5; done
MODEL=$(curl -s http://127.0.0.1:8790/v1/models | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data'] if 'onsai' in m['id']][0])")
for T in 8000 32000; do log "dflash serve cold prefill target $T"; $PY $OUT/cold_prefill_100k.py http://127.0.0.1:8790/v1 "$MODEL" $T --think-off 2>&1 | grep -vE 'transformers\]' | tee -a $OUT/100k.log; log "mem after: $(top -l 1 -s 0 | grep PhysMem | sed 's/PhysMem: //') swap $(sysctl -n vm.swapusage | awk '{print $6}')"; grep -E 'Insufficient Memory|Traceback' $OUT/100k-dflash-server.log | tail -1; done
kill $SRV; log "MLX 100K DONE"
