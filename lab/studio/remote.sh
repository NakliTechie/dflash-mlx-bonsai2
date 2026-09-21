#!/bin/bash
# Run a named job on the Studio under the cooperative lock and pull the results back.
set -euo pipefail
H=ompatnaik@100.89.200.65; R='~/code/chirag/dflash-mlx-bonsai2'; LOCK=~/Code/infra/remote-studio/studio-lock.sh; JOB=${1:?setup|bench|prefill100k|mtp}
mkdir -p "$(dirname "$0")/results"
case $JOB in
  setup) ssh $H "cd $R && ~/.local/bin/uv venv .venv -q && ~/.local/bin/uv pip install --python .venv/bin/python -q -e . 'huggingface_hub>=1.0' && .venv/bin/dflash doctor --model ~/.cache/huggingface/hub/Ternary-Bonsai-2-27B-mlx-2bit 2>&1 | tail -5"; exit 0;;
  bench) EST=60; CMD="cd $R && source .venv/bin/activate && DFLASH_PRISM_VERIFY=v7 zsh lab/studio/jobs/bench.sh";;
  prefill100k) EST=90; CMD="cd $R && source .venv/bin/activate && zsh lab/studio/jobs/prefill100k.sh";;
  mtp) EST=60; CMD="cd $R && zsh lab/studio/jobs/mtp.sh";;
esac
out=$($LOCK acquire "dflash-$JOB" $EST) || { echo "$out"; exit 0; }
TOKEN=$(echo "$out" | sed -n 's/^TOKEN=//p')
ssh $H "nohup bash -c '$CMD' > $R/lab/studio/results/$JOB.log 2>&1 & echo \$!" | { read PID; $LOCK setpid "$TOKEN" "$PID"; echo "started $JOB on the Studio (pid $PID, lock $TOKEN)"; }
echo "poll: ssh $H 'tail -5 $R/lab/studio/results/$JOB.log'; then: rsync -a $H:$R/lab/studio/results/ lab/studio/results/ && $LOCK release $TOKEN"
