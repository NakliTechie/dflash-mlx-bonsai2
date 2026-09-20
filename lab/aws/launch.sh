#!/usr/bin/env bash
# Guarded, detached launch of the leg-6 box under the cairn-skypilot identity (infra/aws Gotchas 9, 10, 13).
#   CONFIRM_GPU_SPEND=1 bash lab/aws/launch.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
[ -n "${CONFIRM_GPU_SPEND:-}" ] || { echo "[launch] ABORT: set CONFIRM_GPU_SPEND=1 (Chirag's explicit go) to spend." >&2; exit 1; }
PROFILE="${AWS_PROFILE_LAUNCH:-cairn-skypilot}"
EXPECT_ARN="arn:aws:iam::615809814090:user/cairn-skypilot"
CLUSTER="${CLUSTER:-localmind-dflash}"; IDLE="${IDLE_MIN:-90}"
got="$(aws sts get-caller-identity --profile "$PROFILE" --query Arn --output text)"
[ "$got" = "$EXPECT_ARN" ] || { echo "[launch] ABORT: $PROFILE resolves to $got" >&2; exit 1; }
export AWS_PROFILE="$PROFILE"
sky api stop >/dev/null 2>&1 || true            # Gotcha 9: the API server provisions with the identity it was STARTED with
sky status >/dev/null 2>&1 || true              # restarts the server under $PROFILE
echo "[launch] $(date '+%F %T') cluster=$CLUSTER profile=$PROFILE idle-autodown=${IDLE}m" | tee -a lab/aws/launch-log.txt
exec sky launch -c "$CLUSTER" lab/aws/localmind-dflash.sky.yaml -y -d -i "$IDLE" --down "$@"
