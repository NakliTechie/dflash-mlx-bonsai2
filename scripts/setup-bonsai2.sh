#!/usr/bin/env bash
# One-command setup for DFlash 2 on Ternary-Bonsai-2-27B (Apple Silicon, MLX).
#
#   bash scripts/setup-bonsai2.sh [--dry-run] [--skip-download]
#
# Idempotent: creates the venv only if missing, installs this repo only if `dflash` is not
# importable from the venv, and downloads each model only if no complete checkout is
# already in the Hugging Face cache. Ends by printing the exact `dflash serve` command.
#
# Environment (all optional):
#   HF_HOME / HF_HUB_CACHE   where models go (default ~/.cache/huggingface)
#   DFLASH_VENV              venv path (default <repo>/.venv)
#   DFLASH_PORT              serve port (default 8790)
#   DFLASH_PREFILL_STEP      --prefill-step-size (default 512; caps 2 K-token prefill peak at ~14 GB)
#   DFLASH_PRISM_VERIFY      verify kernel (default v7)
#   DFLASH_TOOL_PARSER       default off (clients parse <tool_call> text themselves)
#   BONSAI2_PACK_REPO / BONSAI2_DRAFT_REPO   override the HF repo ids
set -euo pipefail

# shellcheck source=scripts/bonsai2-common.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/bonsai2-common.sh"

DRY_RUN=0
SKIP_DOWNLOAD=0
for arg in "$@"; do
  case "$arg" in
    --dry-run)       DRY_RUN=1 ;;
    --skip-download) SKIP_DOWNLOAD=1 ;;
    -h|--help)       sed -n '2,18p' "$0"; exit 0 ;;
    *) die "unknown argument: $arg (use --dry-run, --skip-download)" ;;
  esac
done

run() {                       # run "description" cmd args...  (prints instead under --dry-run)
  local what="$1"; shift
  if [ "$DRY_RUN" = 1 ]; then log "would $what:"; printf '    '; printf '%q ' "$@"; printf '\n'; return 0; fi
  log "$what"
  "$@"
}

# ---------------------------------------------------------------- 1. preflight
[ "$(uname -s)" = Darwin ] || die "this runtime needs macOS (MLX); found $(uname -s)"
[ "$(uname -m)" = arm64 ]  || die "this runtime needs Apple Silicon; found $(uname -m)"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null || die "python3 not found on PATH (need Python 3.11+)"
"$PY" - <<'EOF' || die "Python 3.11+ required"
import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)
EOF

mem_gb=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
if [ "$mem_gb" -lt 24 ]; then
  log "warning: ${mem_gb} GB unified memory; 24 GB is the practical minimum (pack + drafter + 2 K prompt peaks at ~15 GB)"
fi

# ---------------------------------------------------------------- 2. venv
if [ -x "$VENV/bin/python" ]; then
  log "venv present: $VENV"
elif command -v uv >/dev/null; then
  run "create venv with uv at $VENV" uv venv --python "$PY" "$VENV"
else
  run "create venv with $PY -m venv at $VENV" "$PY" -m venv "$VENV"
fi

VPY="$VENV/bin/python"
pip_install() {              # pip_install pkg... — uv when available (fast), else the venv's pip
  if command -v uv >/dev/null; then
    run "install $*" uv pip install --python "$VPY" "$@"
  else
    run "install $*" "$VPY" -m pip install "$@"
  fi
}

# ---------------------------------------------------------------- 3. this repo, editable
if [ "$DRY_RUN" = 0 ] && "$VPY" -c "import dflash_mlx.runtime.prism_pack" 2>/dev/null && [ -x "$VENV/bin/dflash" ]; then
  log "dflash-mlx already installed in the venv"
else
  pip_install -e "$REPO_ROOT"
fi

# `hf` (huggingface_hub >= 1.0 CLI) for the downloads.
if [ -x "$VENV/bin/hf" ]; then
  HF="$VENV/bin/hf"
elif command -v hf >/dev/null; then
  HF="$(command -v hf)"
else
  pip_install "huggingface_hub>=1.0"
  HF="$VENV/bin/hf"
fi

# ---------------------------------------------------------------- 4. models
# ensure_model REPO_ID [hf download args...] -> sets MODEL_DIR
ensure_model() {
  local repo="$1"; shift
  MODEL_DIR="$(find_model_dir "$repo" || true)"
  if [ -n "$MODEL_DIR" ]; then
    log "$repo already present: $MODEL_DIR"
    return 0
  fi
  if [ "$SKIP_DOWNLOAD" = 1 ]; then
    log "$repo not in the cache and --skip-download given"
    MODEL_DIR="<$repo: not downloaded>"
    return 0
  fi
  if [ "$DRY_RUN" = 1 ]; then
    log "would download $repo into $HF_HUB_DIR:"
    printf '    '; printf '%q ' "$HF" download "$repo" "$@"; printf '\n'
    MODEL_DIR="$HF_HUB_DIR/$(hf_cache_dirname "$repo")/snapshots/<rev>"
    return 0
  fi
  log "downloading $repo into $HF_HUB_DIR (resumable; re-run on interruption)"
  # `hf download --quiet` prints the snapshot path as its last stdout line.
  MODEL_DIR="$(HF_HOME="$HF_HOME" "$HF" download "$repo" --quiet "$@" | tail -n 1)"
  model_dir_complete "$MODEL_DIR" || die "download of $repo finished but $MODEL_DIR lacks config.json/model.safetensors"
  log "$repo -> $MODEL_DIR"
}

ensure_model "$PACK_REPO";  PACK_DIR="$MODEL_DIR"       # 8.6 GB; the whole pack incl. runtime/
ensure_model "$DRAFT_REPO" --exclude "*.gguf"; DRAFT_DIR="$MODEL_DIR"   # 3.85 GB bf16 safetensors

# ---------------------------------------------------------------- 5. the serve command
cat >&2 <<EOF

[bonsai2] setup complete. Start the server with:

$(serve_command "$PACK_DIR" "$DRAFT_DIR")

or simply:  bash scripts/serve-bonsai2.sh

Then point an OpenAI-compatible client at http://127.0.0.1:$PORT/v1 with temperature 0
(speculation only engages on greedy requests).
EOF
