"""Render successful scenario-2 executions: my clip next to the Panda putting the bowl on
the plate in LIBERO's official libero_goal task 8 scene (front camera and a top camera
aimed like my phone).

The planner is re-run with the human grip prior for each initial state (same seed as
run_plate_planner.py), then the chosen primitive is executed with cameras on. LIBERO's
own success check decides the label burned into the last frames.

Run as a file (the planner's process pool uses 'spawn'):
  OMP_NUM_THREADS=1 PYTHONPATH=src/sim python src/sim/render_plate_success.py --inits 0 1 2
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio
import numpy as np

from env import CTRL_HZ, PHONE_CAM
from plate_task import PickPlanner, PlateEnv, execute, human_prior

CLIP = Path("data/raw/scenario2_plate.mov")
SIZE = 256
CLIP_WINDOW_S = (0.0, 3.3)    # my clip: reach, lift and set down on the plate (placed at 2.47 s)
HOLD_S = 1.5


def human_frames(n):
    cap = cv2.VideoCapture(str(CLIP))
    fps = cap.get(cv2.CAP_PROP_FPS)
    first, last = (int(t * fps) for t in CLIP_WINDOW_S)
    frames = []
    while len(frames) < last:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        frames.append(cv2.cvtColor(cv2.resize(f, (round(w * SIZE / h), SIZE)), cv2.COLOR_BGR2RGB))
    frames = frames[first:]
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def aim_top_camera(env):
    """Point LIBERO's birdview straight down over the bowl and plate, oriented like the phone."""
    m = env.sim.model
    cid = m.camera_name2id("birdview")
    mid = 0.5 * (env.bowl_pos()[:2] + env.plate_pos()[:2])
    m.cam_pos[cid] = [mid[0], mid[1], env.table_z + 0.9]
    m.cam_quat[cid] = PHONE_CAM["quat"]
    m.cam_fovy[cid] = 30.0
    env.sim.forward()


def label(img, text, ok):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (90, 220, 90) if ok else (230, 90, 90), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("results/scenario2"))
    args = ap.parse_args()

    prior = human_prior()
    planner = PickPlanner(args.workers)
    env = PlateEnv(cam_size=SIZE, cameras=("agentview", "birdview"))
    env.reset(0)
    D = env.bowl_diameter
    summary = []
    for init in args.inits:
        best, res, _, _ = planner.plan(init, D, prior=prior, rng=np.random.default_rng(init), pop=30, iters=3)
        env.reset(init)
        aim_top_camera(env)
        robot = []
        execute(env, best, on_step=lambda: robot.append(env.render()))
        ok = env.success()
        off = float(np.linalg.norm(env.bowl_pos()[:2] - env.plate_pos()[:2]) / D)
        hum = human_frames(len(robot))
        frames = []
        for h, (front, top) in zip(hum, robot):
            frames.append(np.hstack([label(h, "my clip", True), label(front, "LIBERO front", True),
                                     label(top, "LIBERO top", True)]))
        last = np.hstack([label(hum[-1], "my clip", True), label(robot[-1][0], f"LIBERO success: {ok}", ok),
                          label(robot[-1][1], f"off centre {off:.3f} D, grip {np.degrees(best.theta):.0f}", ok)])
        frames += [last] * int(HOLD_S * CTRL_HZ)
        path = args.out / f"robot_success_init{init}.mp4"
        imageio.mimsave(path, frames, fps=CTRL_HZ)
        summary.append(dict(init=init, success=ok, offset_D=round(off, 3), video=str(path),
                            theta_deg=round(float(np.degrees(best.theta)), 1), lift_cm=round(best.lift * 100, 1)))
        print("RENDERED", summary[-1], flush=True)
    (args.out / "robot_success_videos.json").write_text(json.dumps(summary, indent=2))
    planner.close()
    env.close()


if __name__ == "__main__":
    main()
