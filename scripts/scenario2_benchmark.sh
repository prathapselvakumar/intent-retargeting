#!/usr/bin/env bash
# Scenario 2 benchmark on LIBERO libero_goal task 8, evaluated on all 50 official initial states:
#   planners with / without my grip prior, MLP policies (3 seeds each), SmolVLA from cameras, and
#   the official SmolVLA-LIBERO checkpoint (trained on LIBERO's teleoperated demonstrations).
# Restartable: finished steps are skipped. Run detached under a sleep inhibitor:
#   setsid nohup systemd-inhibit --what=sleep:idle --why=benchmark scripts/scenario2_benchmark.sh \
#     > runs/logs/scenario2_benchmark.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH=src/sim:src/policy:src/vision OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
OUT=results/eval_scenario2
mkdir -p $OUT
log() { echo "[$(date '+%a %H:%M')] $*"; }

# 1. planners on the 50 official starts
[ -f results/scenario2/planner_50.json ] || {
  log "planners on 50 official starts"
  $PY src/sim/run_plate_planner.py --inits $(seq 0 49) --workers 8 --out results/scenario2/planner_50.json | grep -c RESULT
}

# 2. training demonstrations from jittered starts
for prior in human none; do
  [ -f data/demos/demos_plate_$prior.npz ] && continue
  log "demos plate_$prior"
  $PY src/sim/gen_plate_demos.py --prior $prior --n 200 --workers 8 | grep -E '"kept"|"tried"'
done

# 3. MLP policies, 3 seeds each, evaluated on the 50 official starts
for prior in human none; do
  for s in 0 1 2; do
    run=runs/plate_mlp_${prior}_s$s
    [ -f $run/policy.pt ] || $PY src/policy/train.py --demos data/demos/demos_plate_$prior.npz --out $run --seed $s > /dev/null
    [ -f $OUT/mlp_${prior}_s$s/report.json ] || \
      $PY src/policy/eval_plate.py --policy $run --video $([ $s = 0 ] && echo 2 || echo 0) --out $OUT/mlp_${prior}_s$s | tail -1
  done
  log "MLP plate_$prior evaluated"
done

# 4. official SmolVLA-LIBERO checkpoint on the same 50 starts (LeRobot's LIBERO evaluation)
[ -f $OUT/smolvla_libero_official/eval_info.json ] || {
  log "official SmolVLA-LIBERO checkpoint"
  .venv/bin/lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero --env.type=libero --env.task=libero_goal \
    --env.task_ids='[8]' --eval.batch_size=1 --eval.n_episodes=50 --output_dir=$OUT/smolvla_libero_official \
    2>&1 | grep -E "Success rate"
}

# 5. SmolVLA fine-tuned from camera images on my plate demonstrations
[ -f data/lerobot/bowl_plate_v2/build_info.json ] || {
  log "plate vision dataset"
  $PY src/vision/render_plate_dataset.py --demos data/demos/demos_plate_human.npz --root data/lerobot/bowl_plate_v2 | tail -1
}
run=runs/smolvla_plate_s0
if [ ! -d $run/checkpoints/020000 ]; then
  if [ -e $run/checkpoints/last/pretrained_model/train_config.json ]; then
    log "SmolVLA plate: resuming"
    .venv/bin/lerobot-train --config_path=$run/checkpoints/last/pretrained_model/train_config.json --resume=true \
      >> runs/logs/smolvla_plate_s0.log 2>&1
  else
    log "SmolVLA plate: training"
    .venv/bin/lerobot-train --policy.path=lerobot/smolvla_base \
      --dataset.repo_id=local/bowl_plate_v2 --dataset.root=data/lerobot/bowl_plate_v2 \
      --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
      --batch_size=32 --steps=20000 --seed=0 --output_dir=$run --policy.push_to_hub=false \
      --wandb.enable=false --save_freq=5000 --log_freq=500 --dataset.video_backend=pyav --num_workers=4 \
      > runs/logs/smolvla_plate_s0.log 2>&1
  fi
fi
[ -f $OUT/smolvla_plate_s0/report.json ] || \
  $PY src/policy/eval_plate.py --vla $run/checkpoints/020000/pretrained_model --out $OUT/smolvla_plate_s0 | tail -1
log "ALL DONE"
