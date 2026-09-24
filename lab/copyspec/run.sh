#!/bin/zsh
# Wait for GPU lock, no Chrome profile, >= 12 GB unused; run the bench; release the lock on exit.
cd "$(dirname "$0")"
free_mb() { memory_pressure | awk '/free percentage/ {gsub("%","",$NF); print int($NF*24*1024/100)}'; }
webgpu_chrome() { ps -axo comm=,args= | awk '$0 ~ /^\/Applications\/Google Chrome.app\/Contents\/MacOS\/Google Chrome/ && /enable-unsafe-webgpu|--headless/' | grep -q .; }
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a run.log; }
log "waiting for lock/chrome/memory"
while true; do
  until mkdir /tmp/mac-gpu.lock 2>/dev/null; do sleep 30; done
  if [ "$(free_mb)" -gt 12000 ]; then break; fi
  rmdir /tmp/mac-gpu.lock; sleep 60
done
trap 'rmdir /tmp/mac-gpu.lock 2>/dev/null' EXIT INT TERM
log "lock taken: $(top -l 1 -s 0 | grep PhysMem)"
REPS=1 OUT=results_rep1.jsonl DFLASH_PRISM_VERIFY=v7 DFLASH_TOOL_PARSER=off PYTHONPATH=/Users/chiragpatnaik/Code/dflash-mlx-bonsai2-stack \
  /Users/chiragpatnaik/Code/dflash-mlx-bonsai2/.venv/bin/python -u copyspec_bench.py >> run.log 2>&1
log "exit $?; swap $(sysctl -n vm.swapusage | awk '{print $6}')"
