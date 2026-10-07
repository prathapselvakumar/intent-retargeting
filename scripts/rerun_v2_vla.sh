#!/usr/bin/env bash
# SmolVLA part of the v2 rerun (step 5 of rerun_v2.sh), restartable: a seed that already has a
# step-20000 checkpoint is skipped, a seed with an earlier checkpoint is resumed from it.
# Wrapped in systemd-inhibit so the laptop cannot suspend mid-training (a suspend froze CUDA
# in the first attempt).
#
# Usage: nohup systemd-inhibit --what=sleep:idle --why="SmolVLA training" \
#          scripts/rerun_v2_vla.sh > runs/logs/rerun_v2_vla.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl PYTHONPATH=src/sim:src/policy:src/vision OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
log() { echo "[$(date '+%a %H:%M')] $*"; }

for s in 0 1 2; do
  run=runs/smolvla_v2_s$s
  if [ -d $run/checkpoints/020000 ]; then
    log "SmolVLA seed $s already trained"
  elif [ -e $run/checkpoints/last/pretrained_model/train_config.json ]; then
    log "SmolVLA seed $s: resuming from $(readlink $run/checkpoints/last)"
    .venv/bin/lerobot-train --config_path=$run/checkpoints/last/pretrained_model/train_config.json \
      --resume=true >> runs/logs/smolvla_v2_s$s.log 2>&1
  else
    log "SmolVLA seed $s: training"
    rm -rf $run
    .venv/bin/lerobot-train --policy.path=lerobot/smolvla_base \
      --dataset.repo_id=local/bowl_vision_v2 --dataset.root=data/lerobot/bowl_vision_v2 \
      --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}' \
      --batch_size=32 --steps=20000 --seed=$s --output_dir=$run --policy.push_to_hub=false \
      --wandb.enable=false --save_freq=5000 --log_freq=500 --dataset.video_backend=pyav --num_workers=4 \
      > runs/logs/smolvla_v2_s$s.log 2>&1
  fi
  [ -d $run/checkpoints/020000 ] || { log "seed $s did not reach step 20000, stopping"; exit 1; }
  for k in 10 5; do
    out=results/eval/smolvla_v2_s${s}_k$k
    [ -f $out/report.json ] && continue
    $PY src/vision/eval_vla.py --ckpt $run/checkpoints/020000/pretrained_model --n 50 --n-action-steps $k \
      --video $([ $s = 0 ] && echo 2 || echo 0) --out $out | grep -E "^(random|human)"
  done
  log "seed $s evaluated"
done
log "ALL DONE"
