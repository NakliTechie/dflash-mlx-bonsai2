#!/bin/zsh
# GPU-sharing gate: no llama-server, no dflash serve, no other chrome-profile Chrome, >= $1 MB unused (default 7000). Polls every 60 s.
need=${1:-7000}
while true; do
  ls=$(pgrep -x llama-server | tr '\n' ' '); ds=$(pgrep -f 'dflash serve' | tr '\n' ' '); ch=$(pgrep -f 'user-data-dir=.*chrome-profile' | tr '\n' ' ')
  un=$(top -l 1 | grep PhysMem | sed -E 's/.* ([0-9]+)([MG]) unused.*/\1 \2/' | awk '{print ($2=="G")? $1*1024 : $1}')
  if [ -z "$ls" ] && [ -z "$ds" ] && [ -z "$ch" ] && [ "$un" -ge "$need" ]; then echo "gate clear: unused=${un}M $(date +%H:%M:%S)"; exit 0; fi
  echo "waiting: llama-server=[$ls] dflash=[$ds] chrome=[$ch] unused=${un}M $(date +%H:%M:%S)"; sleep 60
done
