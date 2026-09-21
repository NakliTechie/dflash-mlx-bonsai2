#!/bin/bash
# Verify-session matrix: block lengths x small-M route, one headless Chrome at a time behind the GPU gate.
# Usage: bash run-verify-matrix.sh [engine-dir] ; writes verify-b<block>-<smallm>.log next to this script.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ENG="${1:-/private/tmp/claude-501/-Users-chiragpatnaik-Code-naklios-universe-LocalMind/4a85d733-7706-45bd-bd5e-98f4de49b3f7/scratchpad/engine}"
cp "$HERE/index-verify.html" "$HERE/verify-qwen35-harness.js" "$HERE/cdp-drive-verify.mjs" "$ENG/"
( cd "$ENG" && node "$HERE/patch-internals.mjs" && node --check engine.dflash.js ) || exit 1
for spec in ${MATRIX:-"8:off 8:f16 4:off 4:f16 5:off 5:f16"}; do
  b=${spec%%:*}; sm=${spec##*:}
  "$HERE/wgsl-gemm-spike/wait-gpu.sh" | tail -1
  ( cd "$ENG" && node cdp-drive-verify.mjs "http://127.0.0.1:8795/index-verify.html?model=/model/Ternary-Bonsai-2-27B-PTQ1_0.gguf&block=$b&smallm=$sm&taps=${TAPS:-6,20,34,48,62}" 900 ) > "$HERE/verify-b$b-$sm.log" 2>&1
  node -e "const t=require('fs').readFileSync('$HERE/verify-b$b-$sm.log','utf8');const i=t.indexOf('RESULTS');const J=JSON.parse(t.slice(i+7));const r=J.results||{};console.log('block=$b smallm=$sm', J.error?('ERROR '+J.error.slice(0,300)):JSON.stringify({matches:r.matches,of:$b,vt:r.verify_tokens,expect:r.expect,next:r.next_token_from_verify,exp:r.next_token_expected,stepMs:r.stepMs,stepMin:r.stepMinMs,decodeMs:r.decodeMs,ratio:r.ratio,nodes:r.graph&&r.graph.verifyNodes,minMargin:r.minMargin,margins:r.margins&&r.margins.map(m=>m.margin)}))"
done
