#!/usr/bin/env bash
# Start `dflash serve` for Ternary-Bonsai-2-27B + the re-fitted DFlash2 drafter.
#
#   bash scripts/serve-bonsai2.sh [extra dflash serve args...]
#
# Uses the venv and models that scripts/setup-bonsai2.sh installed; run that first.
# Defaults: port 8790, --prefill-step-size 512, DFLASH_TOOL_PARSER=off, DFLASH_PRISM_VERIFY=v7
# (override any of them through the environment; see scripts/bonsai2-common.sh).
# Extra arguments are appended to the dflash serve command, e.g. `--diagnostics basic`.
set -euo pipefail

# shellcheck source=scripts/bonsai2-common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bonsai2-common.sh"

[ -x "$VENV/bin/dflash" ] || die "no dflash in $VENV; run: bash scripts/setup-bonsai2.sh"

PACK_DIR="$(find_model_dir "$PACK_REPO" || true)"
[ -n "$PACK_DIR" ]  || die "$PACK_REPO not found in $HF_HUB_DIR; run: bash scripts/setup-bonsai2.sh"
DRAFT_DIR="$(find_model_dir "$DRAFT_REPO" || true)"
[ -n "$DRAFT_DIR" ] || die "$DRAFT_REPO not found in $HF_HUB_DIR; run: bash scripts/setup-bonsai2.sh"

log "pack:    $PACK_DIR"
log "drafter: $DRAFT_DIR"
log "http://127.0.0.1:$PORT/v1  (use temperature 0 for speculative decoding)"

export DFLASH_TOOL_PARSER="$TOOL_PARSER" DFLASH_PRISM_VERIFY="$VERIFY_MODE"
exec "$VENV/bin/dflash" serve \
  --model "$PACK_DIR" \
  --draft "$DRAFT_DIR" \
  --port "$PORT" \
  --prefill-step-size "$PREFILL_STEP" \
  "$@"
