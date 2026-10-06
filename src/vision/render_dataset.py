"""Turn the planner demos into a camera dataset (LeRobot v3 format) for the vision policy.

Each demo is re-executed from its recorded start with its executed actions (`act_exec`)
while both cameras render (goal drawn in); the frames are labelled with the planner's
actions (`act`). The planner commands only the 5 action dimensions the policy outputs,
so replays reproduce the demonstrations; each replay is still re-scored and only
successful ones are kept.

Usage: python src/vision/render_dataset.py --demos data/demos/demos_strategy.npz \
           --root data/lerobot/bowl_vision_strategy
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from common import CAMS, IMG, KEYS, TASK, observe, state_vec          # noqa: E402
from env import CTRL_HZ, BowlEnv                                      # noqa: E402
from planner import SUCCESS_D, SUCCESS_TILT                           # noqa: E402

STATE_NAMES = ["eef_x", "eef_y", "eef_z_above_table", "sin_yaw", "cos_yaw", "gripper_opening"]
ACTION_NAMES = ["dx", "dy", "dz", "dyaw", "grip"]


def features():
    img = {"dtype": "video", "shape": (IMG, IMG, 3), "names": ["height", "width", "channels"]}
    return {KEYS[0]: img, KEYS[1]: dict(img),
            "observation.state": {"dtype": "float32", "shape": (6,), "names": STATE_NAMES},
            "action": {"dtype": "float32", "shape": (5,), "names": ACTION_NAMES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if args.root.exists():
        shutil.rmtree(args.root)
    ds = LeRobotDataset.create(repo_id=f"local/{args.root.name}", fps=CTRL_HZ, root=args.root,
                               features=features(), robot_type="panda", use_videos=True)

    d = np.load(args.demos)
    off = np.r_[0, np.cumsum(d["ep_len"])]
    n_ep = len(d["ep_len"]) if args.max_episodes is None else args.max_episodes
    env = BowlEnv(cam_size=IMG, cameras=CAMS)
    kept, errs = 0, []
    for i in range(n_ep):
        env.reset(bowl_xy=d["start"][i])
        goal = d["goal"][i]
        frames = []
        executed = d["act_exec"] if "act_exec" in d.files else d["act"]
        for a5, x5 in zip(d["act"][off[i]:off[i + 1]], executed[off[i]:off[i + 1]]):
            obs = observe(env, goal)
            frames.append({**obs, "observation.state": state_vec(env),
                           "action": a5.astype(np.float32), "task": TASK})
            a = np.zeros(7)
            a[[0, 1, 2, 5, 6]] = x5
            env.step(a)
        err = float(np.linalg.norm(env.bowl_pos()[:2] - goal) / env.bowl_diameter)
        errs.append(err)
        if err < SUCCESS_D and env.bowl_tilt_deg() < SUCCESS_TILT:
            for f in frames:
                ds.add_frame(f)
            ds.save_episode()
            kept += 1
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{n_ep} replayed, {kept} kept", flush=True)
    ds.finalize()
    meta = dict(source=str(args.demos), replayed=n_ep, kept=kept,
                replay_success_rate=round(kept / n_ep, 3),
                median_replay_err_D=round(float(np.median(errs)), 3), task=TASK)
    (args.root / "build_info.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    env.close()


if __name__ == "__main__":
    main()
