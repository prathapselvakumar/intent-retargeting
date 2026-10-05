#!/usr/bin/env bash
# Scenario 1 — slide the bowl. Phone clips → intent → planner → demos → policy → evaluation.
# Inputs: data/raw/episode_000_right.mov, data/raw/episode_050_left.mov (top-down iPhone clips).
set -euo pipefail
cd "$(dirname "$0")/.."
CLIPS=(episode_000_right episode_050_left)
export MUJOCO_GL=egl

# 1. Video → hand landmarks, metric bowl track, contact intent           (.venv-extract)
VIDEOS=$(printf 'data/raw/%s.mov ' "${CLIPS[@]}")
.venv-extract/bin/python src/extract/hands.py $VIDEOS
.venv-extract/bin/python src/extract/bowl.py $VIDEOS --diameter-cm 15
.venv-extract/bin/python src/extract/contact.py "${CLIPS[@]}"

# 2. Baseline: replay the human hand motion on the Panda                  (.venv)
PYTHONPATH=src/sim .venv/bin/python src/sim/replay_hand.py "${CLIPS[@]}"

# 3. Intent planner with / without the human contact prior
PYTHONPATH=src/sim .venv/bin/python src/sim/run_planner.py "${CLIPS[@]}" --seeds 10

# 4. Planner → 300 demos per condition → distilled policy (3 training seeds) → evaluation
for prior in strategy none; do
  PYTHONPATH=src/sim .venv/bin/python src/sim/gen_demos.py --prior $prior --n 300
  for s in 0 1 2; do
    run=runs/policy_${prior}$([ $s = 0 ] || echo _s$s)
    .venv/bin/python src/policy/train.py --demos data/demos/demos_$prior.npz --out $run --seed $s
    .venv/bin/python src/policy/evaluate.py --policy $run --n 50
  done
done
.venv/bin/python src/policy/evaluate.py --hand-replay --n 25

# 5. Vision policy: camera dataset (LeRobot format) → SmolVLA fine-tune → closed-loop evaluation
PYTHONPATH=src/sim:src/vision .venv/bin/python src/vision/render_dataset.py \
  --demos data/demos/demos_strategy.npz --root data/lerobot/bowl_vision_strategy
.venv/bin/lerobot-train --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/bowl_vision_strategy --dataset.root=data/lerobot/bowl_vision_strategy \
  --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
  --batch_size=32 --steps=15000 --output_dir=runs/smolvla_strategy --policy.push_to_hub=false \
  --wandb.enable=false --save_freq=5000 --dataset.video_backend=pyav
.venv/bin/python src/vision/eval_vla.py --ckpt runs/smolvla_strategy/checkpoints/last/pretrained_model \
  --n 50 --out results/eval/smolvla_strategy
