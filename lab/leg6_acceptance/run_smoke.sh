#!/bin/zsh
cd ~/Code/dflash-mlx-bonsai2 && source .venv/bin/activate
D=~/Code/models/bonsai2-drafter-data
echo "[$(date '+%H:%M:%S')] smoke: baseline eval on captured shards" >> $D/train.log
python lab/leg6_acceptance/train_adapter.py --eval-only --eval-docs 16 > $D/smoke.out 2>&1 &
PY=$!
while kill -0 $PY 2>/dev/null; do
  used=$(sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/')
  if [ "${used%.*}" -gt 16000 ]; then echo "[$(date '+%H:%M:%S')] WATCHDOG: swap ${used}M > 16 GB, killing smoke" >> $D/train.log; kill $PY; break; fi
  sleep 5
done
echo "[$(date '+%H:%M:%S')] smoke exit; tail:" >> $D/train.log; grep -v transformers $D/smoke.out | tail -6 >> $D/train.log
