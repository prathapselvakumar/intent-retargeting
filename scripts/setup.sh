#!/usr/bin/env bash
# Reproducible environment setup: Python 3.12 venv + LeRobot (SmolVLA, LIBERO).
set -euo pipefail
cd "$(dirname "$0")/.."

[ -d third_party/lerobot ] || git clone --depth 1 https://github.com/huggingface/lerobot.git third_party/lerobot
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e "third_party/lerobot[smolvla,libero]"

# LIBERO asks an interactive question on first import; answer "no" to use default paths.
echo n | python -c "import libero.libero" >/dev/null

echo "Done. Activate with: source .venv/bin/activate && export MUJOCO_GL=egl"
