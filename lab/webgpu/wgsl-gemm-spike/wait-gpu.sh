#!/bin/bash
# GPU-sharing rule: wait until no llama-server / dflash serve process and >= 7 GB unused before a model-loading Chrome launch.
while :; do
  L=$(pgrep -x llama-server); D=$(pgrep -f '[d]flash serve'); U=$(top -l 1 | grep PhysMem | sed -E 's/.*, ([0-9]+)([MG]) unused.*/\1 \2/' | awk '{print ($2=="G") ? $1*1024 : $1}')
  if [ -z "$L" ] && [ -z "$D" ] && [ "${U:-0}" -ge 7000 ]; then echo "gpu free: unused=${U}M $(date +%T)"; exit 0; fi
  echo "waiting: llama-server=${L:-none} dflash=${D:-none} unused=${U}M $(date +%T)"; sleep 30
done
