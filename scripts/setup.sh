#!/usr/bin/env bash
# Reproducible setup. Two environments, kept separate so MediaPipe's pins can't break LeRobot:
#   .venv          Python 3.12 + LeRobot (SmolVLA, LIBERO) — simulation, planning, policies
#   .venv-extract  Python 3.12 + MediaPipe/OpenCV           — video extraction
set -euo pipefail
cd "$(dirname "$0")/.."

LEROBOT_COMMIT=8c920c4270460851cedd2737657584586d3dc66f   # the commit all results were produced with
if [ ! -d third_party/lerobot ]; then
  git clone https://github.com/huggingface/lerobot.git third_party/lerobot
  git -C third_party/lerobot checkout "$LEROBOT_COMMIT"
fi
uv venv -p 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -e "third_party/lerobot[smolvla,libero]"
# LIBERO asks an interactive question on first import; answer "no" to use default paths.
echo n | .venv/bin/python -c "import libero.libero" >/dev/null

uv venv -p 3.12 .venv-extract
VIRTUAL_ENV=.venv-extract uv pip install -r requirements-extract.txt
mkdir -p models
curl -sSL -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

echo "Done. Run the pipelines with scripts/run_scenario1.sh and scripts/run_scenario2.sh"
