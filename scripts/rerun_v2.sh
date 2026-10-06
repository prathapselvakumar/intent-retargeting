#!/usr/bin/env bash
# Version 2 of all simulation results: LIBERO's default bowl physics, a planner that commands
# only the policy's 5 action dimensions (exact replay), DART-style correction demos, and
# SmolVLA trained 3 times on all demos. Steps run one after another to bound memory use.
#
# Usage: nohup scripts/rerun_v2.sh > runs/logs/rerun_v2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH=src/sim:src/policy:src/vision OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
log() { echo "[$(date +%H:%M)] $*"; }
NOISE=0.15
SETS=(strategy none strategy_dart none_dart)

# --- 1. demonstrations (CPU, 16 planner workers) -------------------------------------------
for tag in "${SETS[@]}"; do
  prior=${tag%%_dart}
  noise=0.0; [[ $tag == *_dart ]] && noise=$NOISE
  [ -f data/demos/demos_$tag.npz ] && { log "demos $tag exist, skipping"; continue; }
  log "demos $tag (prior=$prior noise=$noise)"
  $PY src/sim/gen_demos.py --prior "$prior" --noise "$noise" --tag "$tag" --n 300 --workers 16 \
    | grep -E '^\[|"kept"|"tried"' | tail -3
done

# --- 2. baselines and planner comparison under the new physics -----------------------------
log "hand-replay baseline sweep at the recorded start"
$PY src/sim/replay_hand.py episode_000_right episode_050_left | grep -E "^episode"
log "hand-replay baseline on held-out tasks"
$PY src/policy/evaluate.py --hand-replay --n 25 --video 0 --out results/eval/hand_replay | grep -E "^human"
log "planner with / without the human prior (scenario 1)"
$PY src/sim/run_planner.py episode_000_right episode_050_left --seeds 10 --workers 16 | tail -2
log "planner with / without the human prior (scenario 2)"
$PY src/sim/run_plate_planner.py --inits 0 1 2 3 4 --workers 8 | grep -c RESULT
log "scenario 2 success videos"
$PY src/sim/render_plate_success.py --inits 0 1 2 --workers 8 | grep RENDERED
ffmpeg -v error -y -i results/scenario2/robot_success_init1.mp4 \
  -vf "fps=10,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer" \
  results/figures/scenario2_success.gif

# --- 3. MLP policies: 4 demo sets x 3 seeds, evaluated 3 at a time ------------------------
for tag in "${SETS[@]}"; do
  for s in 0 1 2; do
    $PY src/policy/train.py --demos data/demos/demos_$tag.npz --out runs/policy_${tag}_s$s --seed $s > /dev/null
  done
done
log "MLP policies trained"
for tag in "${SETS[@]}"; do
  for s in 0 1 2; do
    $PY src/policy/evaluate.py --policy runs/policy_${tag}_s$s --n 50 --video $([ $s = 0 ] && echo 2 || echo 0) \
      --out results/eval/policy_${tag}_s$s > runs/logs/eval_policy_${tag}_s$s.log 2>&1 &
  done
  wait
  log "evaluated $tag: $(cat results/eval/policy_${tag}_s*/report.json | grep -o '"success": [0-9]*' | awk '{s+=$2} END {print s "/450"}')"
done

# --- 4. world model on the new physics -----------------------------------------------------
log "world-model exploration data"
mv -f data/demos/explore.npz data/demos/archive_bowl250g/ 2>/dev/null
$PY src/worldmodel/explore.py --n 3000 --workers 12 | tail -1
log "world model (training runs alongside SmolVLA below)"
$PY src/worldmodel/bowl_wm.py > runs/logs/worldmodel_v2.log 2>&1 &
WM=$!

# --- 5. SmolVLA: dataset from the better strategy set, 3 seeds, 2 chunk lengths ----------
BEST=$($PY - <<'EOF'
import glob, json
def tot(tag):
    return sum(sum(json.load(open(f))[s]["success"] for s in ("random", "human_right", "human_left"))
               for f in glob.glob(f"results/eval/policy_{tag}_s*/report.json"))
print(max(["strategy", "strategy_dart"], key=tot))
EOF
)
log "vision dataset from demos_$BEST"
rm -rf data/lerobot/bowl_vision_v2
$PY src/vision/render_dataset.py --demos data/demos/demos_$BEST.npz --root data/lerobot/bowl_vision_v2 \
  | grep -E '"(replayed|kept)"'
for s in 0 1 2; do
  log "SmolVLA seed $s"
  rm -rf runs/smolvla_v2_s$s
  .venv/bin/lerobot-train --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/bowl_vision_v2 --dataset.root=data/lerobot/bowl_vision_v2 \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
    --batch_size=32 --steps=20000 --seed=$s --output_dir=runs/smolvla_v2_s$s --policy.push_to_hub=false \
    --wandb.enable=false --save_freq=5000 --log_freq=500 --dataset.video_backend=pyav --num_workers=4 \
    > runs/logs/smolvla_v2_s$s.log 2>&1
  for k in 10 5; do
    $PY src/vision/eval_vla.py --ckpt runs/smolvla_v2_s$s/checkpoints/last/pretrained_model --n 50 \
      --n-action-steps $k --video $([ $s = 0 ] && echo 2 || echo 0) --out results/eval/smolvla_v2_s${s}_k$k \
      | grep -E "^(random|human)"
  done
done
wait $WM
log "ALL DONE"
