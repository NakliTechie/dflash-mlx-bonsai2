#!/bin/zsh
R3=~/models/Qwen3.8-27B-DFlash2-r3; PACK=~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit; OUT=lab/studio/results
DFLASH_TOOL_PARSER=off DFLASH_PRISM_VERIFY=v7 dflash serve --model $PACK --draft-model $R3 --host 127.0.0.1 --port 8790 --prefill-step-size 2048 --cache-limit 4GB --dflash-max-ctx 120000 > $OUT/prefill-server.log 2>&1 & SRV=$!
for i in $(seq 1 120); do curl -s --max-time 3 http://127.0.0.1:8790/v1/models >/dev/null 2>&1 && break; sleep 5; done
MODEL=$(curl -s http://127.0.0.1:8790/v1/models | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data'] if 'onsai' in m['id']][0])")
for T in 8000 32000 100000; do echo "target $T"; python lab/leg9/cold_prefill_100k.py http://127.0.0.1:8790/v1 "$MODEL" $T --think-off 2>&1 | grep -vE 'transformers\]'; done
kill $SRV; echo PREFILL DONE
