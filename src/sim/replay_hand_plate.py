"""Scenario 2 baseline: copy the human HAND MOTION into LIBERO's bowl-on-plate task.

The pinch point (thumb–index midpoint) is expressed relative to the plate, in bowl
diameters, from the hand-held clip. LIBERO's bowl and plate are laid out differently
from the kitchen table, so the human trajectory is fitted generously: a 2-D rotation +
scale maps the human's start (bowl relative to plate) exactly onto the sim's start. Height
follows the lift seen in the video (bowl apparent size); the gripper closes when the
thumb–index aperture closes; gripper yaw stays at its default.

Usage: python src/sim/replay_hand_plate.py --init 0 1 2 ...
"""
import argparse
import json
from pathlib import Path

import imageio
import numpy as np

from env import CTRL_HZ
from plate_task import PlateEnv
from replay_hand import grip_signal, resample, smooth

THUMB_TIP, INDEX_TIP = 4, 8
PROC_W, PROC_H = 960, 540
CAM_DIST_D = 4.6            # camera distance in bowl diameters (from the bowl's pixel size)
HOLD_ZS = (0.015, 0.03, 0.045)


def load_human(root=Path("data/processed")):
    s = np.load(root / "scene" / "scenario2_plate.npz")
    h = np.load(root / "hands" / "scenario2_plate.npz")
    fps = float(s["fps"])
    T = len(s["bowl_cur_px"])
    xy = h["img_xy"][:T, 0] * [PROC_W, PROC_H]
    pinch_px = 0.5 * (xy[:, THUMB_TIP] + xy[:, INDEX_TIP])
    bc = s["bowl_cur_px"]
    flip = np.array([1, -1])
    pinch_rel = s["bowl_rel_plate_D"] + (pinch_px - bc[:, :2]) * flip / (2 * bc[:, 2:3])
    ok = np.isfinite(pinch_rel[:, 0])
    first, last = np.flatnonzero(ok)[[0, -1]]
    for i in range(T):                          # hold the nearest visible sample when the hand is out of view
        if not ok[i]:
            pinch_rel[i] = pinch_rel[first] if i < first else pinch_rel[last if i > last else i - 1]
    wx = h["world_xyz"][:T, 0]
    ap = np.linalg.norm(wx[:, THUMB_TIP] - wx[:, INDEX_TIP], axis=-1) * 100
    ap = np.where(np.isfinite(ap), ap, 6.0)     # hand out of view → treat as open
    rel_r = np.nan_to_num(s["bowl_rel_radius"], nan=1.0)
    lift_D = np.clip(CAM_DIST_D * (1 - 1 / rel_r), 0, None)
    return dict(fps=fps, T=T, pinch=smooth(pinch_rel, 5), aperture=smooth(ap, 7),
                lift_D=smooth(lift_D, 5), start=s["bowl_rel_plate_D"][0])


def fit(human_start, sim_start):
    """Rotation + scale taking the human start vector onto the sim start vector."""
    a = np.arctan2(sim_start[1], sim_start[0]) - np.arctan2(human_start[1], human_start[0])
    s = np.linalg.norm(sim_start) / np.linalg.norm(human_start)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return lambda v: (s * (R @ np.asarray(v).T)).T


def run(init_idx, hold_z, video=None):
    Hm = load_human()
    n = int(Hm["T"] / Hm["fps"] * CTRL_HZ)
    pinch = resample(Hm["pinch"], Hm["fps"], n)
    lift = resample(Hm["lift_D"], Hm["fps"], n)
    grip = grip_signal(resample(Hm["aperture"], Hm["fps"], n))
    env = PlateEnv(cam_size=128 if video else 32, cameras=("agentview",))
    env.reset(init_idx)
    D = env.bowl_diameter
    plate = env.plate_pos()[:2]
    f = fit(Hm["start"], (env.bowl_pos()[:2] - plate) / D)
    ee = plate + f(pinch) * D
    z = env.table_z + hold_z + lift * D
    frames = []
    snap = (lambda: frames.append(env.render()[0])) if video else (lambda: None)
    for k in range(30):                          # approach from above, gripper open
        env.step_to([*ee[0], env.table_z + (0.15 if k < 15 else hold_z)], -1, yaw=0.0)
        snap()
    for i in range(n):
        env.step_to([*ee[i], z[i]], grip[i], yaw=0.0)
        snap()
    for _ in range(20):
        env.step_to([*ee[-1], env.table_z + 0.15], -1, yaw=0.0)
        snap()
    res = dict(init=init_idx, hold_cm=hold_z * 100, success=env.success(),
               offset_D=round(float(np.linalg.norm(env.bowl_pos()[:2] - plate) / D), 3),
               grip_closed_frac=round(float((grip > 0).mean()), 2))
    if video:
        imageio.mimsave(video, frames, fps=CTRL_HZ)
    env.close()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", type=Path, default=Path("results/scenario2/hand_replay.json"))
    args = ap.parse_args()
    out = []
    for z in HOLD_ZS:
        for i in args.init:
            r = run(i, z, video=f"results/scenario2/hand_replay_init{i}_z{round(z*1000)}.mp4" if i == args.init[0] else None)
            out.append(r)
            print(r, flush=True)
    args.out.write_text(json.dumps(out, indent=2))
    for z in HOLD_ZS:
        rs = [r for r in out if r["hold_cm"] == z * 100]
        print(f"hold {z*100:.1f} cm: success {sum(r['success'] for r in rs)}/{len(rs)}")
