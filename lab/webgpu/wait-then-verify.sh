#!/bin/bash
# GPU-sharing gate: no llama-server, no real `dflash serve` process (zsh -c wrappers that merely quote the string are
# excluded), no other Chrome on a *chrome-profile* user-data-dir, and >= 7 GB unused; poll every 60 s; then one run.
cd "$(dirname "$0")"
while :; do
  L=$(pgrep -x llama-server | tr '\n' ' ')
  D=$(for p in $(pgrep -f 'dflash serve'); do c=$(ps -o comm= -p "$p"); if ! echo "$c" | grep -qE '(zsh|bash|/sh)$'; then echo -n "$p($c) "; fi; done)
  C=$(pgrep -f 'user-data-dir=.*chrome-profile' | tr '\n' ' ')
  U=$(top -l 1 | grep PhysMem | sed -E 's/.*, ([0-9]+)M unused.*/\1/')
  if [ -z "$L" ] && [ -z "$D" ] && [ -z "$C" ] && [ "${U:-0}" -ge 7000 ]; then echo "gate clear: unused=${U}M $(date +%T)"; break; fi
  echo "waiting: llama-server=[${L}] dflash=[${D}] chrome=[${C}] unused=${U}M $(date +%T)"; sleep 60
done
echo "launch $(date +%T)"
node cdp-drive-verify.mjs 'http://127.0.0.1:8795/index-verify.html?model=/model/Ternary-Bonsai-2-27B-PTQ1_0.gguf' 1500
echo "exit=$? $(date +%T)"
