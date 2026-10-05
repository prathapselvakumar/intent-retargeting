#!/usr/bin/env bash
# Scenario 2 — put the bowl on the plate (LIBERO libero_goal #8). Hand-held phone clip.
# Input: data/raw/scenario2_plate.mov
set -euo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# 1. Video → hand landmarks, camera-stabilised bowl-relative-to-plate track  (.venv-extract)
.venv-extract/bin/python src/extract/hands.py data/raw/scenario2_plate.mov
PYTHONPATH=src/extract .venv-extract/bin/python src/extract/scene_plate.py data/raw/scenario2_plate.mov

# 2. Baseline: replay the human hand motion in LIBERO's official scene     (.venv)
PYTHONPATH=src/sim .venv/bin/python src/sim/replay_hand_plate.py --init 0 1 2 3 4 5 6 7 8 9
