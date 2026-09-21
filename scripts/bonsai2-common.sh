# shellcheck shell=bash
# Shared by scripts/setup-bonsai2.sh and scripts/serve-bonsai2.sh. Source, do not run.
#
# Resolves the venv and the two model directories (PrismML pack + re-fitted drafter)
# from the Hugging Face cache. Every value can be overridden through the environment.

# shellcheck disable=SC2034  # consumed by the sourcing scripts
PACK_REPO="${BONSAI2_PACK_REPO:-prism-ml/Ternary-Bonsai-2-27B-mlx-2bit}"
DRAFT_REPO="${BONSAI2_DRAFT_REPO:-naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2}"

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_HUB_DIR="${HF_HUB_CACHE:-$HF_HOME/hub}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${DFLASH_VENV:-$REPO_ROOT/.venv}"

PORT="${DFLASH_PORT:-8790}"
PREFILL_STEP="${DFLASH_PREFILL_STEP:-512}"
VERIFY_MODE="${DFLASH_PRISM_VERIFY:-v7}"
TOOL_PARSER="${DFLASH_TOOL_PARSER:-off}"

log()  { printf '[bonsai2] %s\n' "$*" >&2; }
die()  { printf '[bonsai2] error: %s\n' "$*" >&2; exit 1; }

# hf_cache_dirname prism-ml/X  ->  models--prism-ml--X
hf_cache_dirname() { printf 'models--%s' "${1//\//--}"; }

# A directory counts as a complete model checkout when it holds both files.
model_dir_complete() { [ -f "$1/config.json" ] && [ -f "$1/model.safetensors" ]; }

# find_model_dir REPO_ID -> prints the local directory, or nothing.
# Looks at (1) the standard cache layout $HF_HUB_DIR/models--org--name/snapshots/<rev>/
# (newest snapshot with a complete checkout wins) and (2) a plain local-dir checkout at
# $HF_HUB_DIR/<name> (what `hf download --local-dir` produces).
find_model_dir() {
  local repo="$1" name="${1##*/}" snap best="" snaps
  snaps="$HF_HUB_DIR/$(hf_cache_dirname "$repo")/snapshots"
  if [ -d "$snaps" ]; then
    # newest complete snapshot wins
    for snap in "$snaps"/*/; do
      [ -d "$snap" ] || continue
      if model_dir_complete "$snap" && { [ -z "$best" ] || [ "$snap" -nt "$best" ]; }; then best="${snap%/}"; fi
    done
  fi
  if [ -z "$best" ] && model_dir_complete "$HF_HUB_DIR/$name"; then
    best="$HF_HUB_DIR/$name"
  fi
  [ -n "$best" ] && printf '%s\n' "$best"
}

# serve_command PACK_DIR DRAFT_DIR -> prints the exact serve command, one flag per line.
serve_command() {
  cat <<EOF
DFLASH_TOOL_PARSER=$TOOL_PARSER DFLASH_PRISM_VERIFY=$VERIFY_MODE \\
"$VENV/bin/dflash" serve \\
  --model "$1" \\
  --draft "$2" \\
  --port $PORT \\
  --prefill-step-size $PREFILL_STEP
EOF
}
