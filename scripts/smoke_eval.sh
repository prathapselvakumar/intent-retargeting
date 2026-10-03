#!/usr/bin/env bash
# Baseline: pretrained SmolVLA on the target LIBERO task (libero_goal #8, "put the bowl on the plate").
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export MUJOCO_GL=egl

lerobot-eval \
  --policy.path="${POLICY:-HuggingFaceVLA/smolvla_libero}" \
  --env.type=libero --env.task=libero_goal --env.task_ids="[${TASK_ID:-8}]" \
  --eval.batch_size=1 --eval.n_episodes="${N_EPISODES:-10}" \
  --output_dir="outputs/eval_$(date +%Y%m%d_%H%M%S)"
