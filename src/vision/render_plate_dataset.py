"""Scenario-2 camera dataset (LeRobot v3 format) for SmolVLA: replay each planner demonstration
from its saved start state with the front and wrist cameras on, label frames with the
planner's actions, and keep only replays that LIBERO's success check accepts.

Usage: python src/vision/render_plate_dataset.py --demos data/demos/demos_plate_human.npz \
           --root data/lerobot/bowl_plate_v2
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from common import IMG, KEYS, state_vec                      # noqa: E402
from env import CTRL_HZ                                      # noqa: E402
from plate_task import PlateEnv                              # noqa: E402
from render_dataset import features                          # noqa: E402

TASK = "put the bowl on the plate"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if args.root.exists():
        shutil.rmtree(args.root)
    ds = LeRobotDataset.create(repo_id=f"local/{args.root.name}", fps=CTRL_HZ, root=args.root,
                               features=features(), robot_type="panda", use_videos=True)
    d = np.load(args.demos)
    off = np.r_[0, np.cumsum(d["ep_len"])]
    env = PlateEnv(cam_size=IMG, cameras=("agentview", "robot0_eye_in_hand"))
    kept = 0
    for i in range(len(d["ep_len"])):
        env.reset(int(d["init"][i]))
        env.sim.data.qpos[:] = d["qpos"][i]
        env.sim.data.qvel[:] = d["qvel"][i]
        env.sim.forward()
        frames = []
        for a5, x5 in zip(d["act"][off[i]:off[i + 1]], d["act_exec"][off[i]:off[i + 1]]):
            imgs = env.render()
            frames.append({KEYS[0]: np.ascontiguousarray(imgs[0]), KEYS[1]: np.ascontiguousarray(imgs[1]),
                           "observation.state": state_vec(env), "action": a5.astype(np.float32), "task": TASK})
            a = np.zeros(7)
            a[[0, 1, 2, 5, 6]] = x5
            env.step(a)
        if env.success():
            for f in frames:
                ds.add_frame(f)
            ds.save_episode()
            kept += 1
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(d['ep_len'])} replayed, {kept} kept", flush=True)
    ds.finalize()
    meta = dict(source=str(args.demos), replayed=int(len(d["ep_len"])), kept=kept, task=TASK)
    (args.root / "build_info.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    env.close()


if __name__ == "__main__":
    main()
