#!/usr/bin/env bash
# Regenerates every figure, GIF and downscaled clip in the repo from the result files and my
# original recordings in data/raw/. Run after the pipelines (scripts/run_scenario*.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
gif() {  # gif <input video> <output gif>
  ffmpeg -v error -y -i "$1" \
    -vf "fps=10,scale=720:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer" "$2"
}
small() {  # small <input video> <output mp4>: 640 px wide, no audio
  ffmpeg -v error -y -i "$1" -an -vf "scale=640:-2" -c:v libx264 -crf 30 -preset slow -pix_fmt yuv420p "$2"
}
mkdir -p data/clips results/figures
small data/raw/episode_000_right.mov data/clips/scenario1_right_hand.mp4
small data/raw/episode_050_left.mov  data/clips/scenario1_left_hand.mp4
small data/raw/scenario2_plate.mov   data/clips/scenario2_bowl_onto_plate.mp4
small data/processed/scene/scenario2_plate_scene.mp4 results/scenario2/tracking_overlay.mp4
gif results/planner/episode_050_left_human_prior.mp4 results/figures/hero_left_clip_vs_robot.gif
gif results/scenario2/robot_success_init1.mp4        results/figures/scenario2_success.gif
.venv-extract/bin/python src/report/figures.py --scenario2
.venv/bin/python src/report/demo_video.py
