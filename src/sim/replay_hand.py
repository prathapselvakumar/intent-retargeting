"""Baseline: copy the human HAND MOTION onto the Panda ("motion retargeting").

The pinch point (midpoint of thumb tip and index tip) is expressed relative to the bowl's
starting position in units of bowl diameters, rotated into the robot frame and scaled to
the sim bowl. The gripper closes when the human thumb–index aperture closes. A phone
camera gives no hand height, so the gripper stays at a fixed height; we sweep several
heights and report each, so the baseline isn't judged on one unlucky guess. This is the
standard "retarget the hand" recipe; it ignores what the bowl does.

The score is how well the SIM bowl reproduces the HUMAN bowl path, in bowl diameters.

Usage: python src/sim/replay_hand.py episode_000_right episode_050_left
"""
import argparse
import json
from pathlib import Path

import cv2
import imageio
import numpy as np

from env import CTRL_HZ, BowlEnv, human_to_robot

PROC_W = 960
THUMB_TIP, INDEX_TIP = 4, 8
GRASP_CLOSE_CM, GRASP_OPEN_CM = 4.6, 5.2   # hysteresis on thumb–index aperture
HOLD_ZS = (0.015, 0.03, 0.045)              # gripper heights above the table (bowl is 5.3 cm tall)
APPROACH_Z = 0.15
APPROACH_S = 2.0
SETTLE_S = 1.0


def smooth(x, k=5):
    pad = np.pad(x, [(k // 2, k // 2)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    kern = np.ones(k) / k
    return np.apply_along_axis(lambda c: np.convolve(c, kern, "valid"), 0, pad)


def load_human(name, root=Path("data/processed")):
    hd = np.load(root / "hands" / f"{name}.npz")
    bd = np.load(root / "bowl" / f"{name}.npz")
    intent = json.loads((root / "intent" / f"{name}.json").read_text())
    fps = float(bd["fps"])
    ph = round(int(hd["height"]) * PROC_W / int(hd["width"]))
    T = min(len(hd["img_xy"]), len(bd["center_px"]))
    act = intent["active_slot"]

    xy = hd["img_xy"][:T, act] * [PROC_W, ph]
    pinch = smooth(0.5 * (xy[:, THUMB_TIP] + xy[:, INDEX_TIP]))
    wx = hd["world_xyz"][:T, act]
    aperture = smooth(np.linalg.norm(wx[:, THUMB_TIP] - wx[:, INDEX_TIP], axis=-1) * 100, 7)

    c = bd["center_px"][:T]
    r = bd["radius_px"][:T]
    D = 2 * np.nanmedian(np.r_[r[:15], r[-15:]])            # rim diameter in px at rest
    flip = np.array([1, -1])                                   # image y down → y away
    pinch_rel = (pinch - c[0]) * flip / D                      # bowl diameters
    bowl_rel = (c - c[0]) * flip / D
    return dict(fps=fps, T=T, pinch_rel=pinch_rel, bowl_rel=bowl_rel, aperture=aperture)


def resample(x, fps_in, n_out):
    t_in = np.arange(len(x)) / fps_in
    t_out = np.arange(n_out) / CTRL_HZ
    return np.stack([np.interp(t_out, t_in, x[:, i]) for i in range(x.shape[1])], -1) \
        if x.ndim == 2 else np.interp(t_out, t_in, x)


def grip_signal(aperture):
    g, closed = np.empty(len(aperture)), False
    for i, a in enumerate(aperture):
        closed = a < GRASP_CLOSE_CM if not closed else a < GRASP_OPEN_CM
        g[i] = 1.0 if closed else -1.0
    return g


def human_frames(name, n_out, fps, size):
    cap = cv2.VideoCapture(f"data/raw/{name}.mov")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        frames.append(cv2.cvtColor(cv2.resize(f, (round(w * size / h), size)), cv2.COLOR_BGR2RGB))
    idx = np.clip(np.round(np.arange(n_out) / CTRL_HZ * fps).astype(int), 0, len(frames) - 1)
    return [frames[i] for i in idx]


def run(name, out_dir: Path, hold_z, bowl_xy=(0.0, 0.0)):
    H = load_human(name)
    n = int(H["T"] / H["fps"] * CTRL_HZ)
    pinch = resample(H["pinch_rel"], H["fps"], n)
    target_bowl = resample(H["bowl_rel"], H["fps"], n)
    grip = grip_signal(resample(H["aperture"], H["fps"], n))

    env = BowlEnv()
    env.reset(bowl_xy=bowl_xy)
    D = env.bowl_diameter
    b0 = env.bowl_pos()
    z_hold, z_high = env.table_z + hold_z, env.table_z + APPROACH_Z
    ee_xy = b0[:2] + human_to_robot(pinch) * D

    frames = []
    hum = human_frames(name, n, H["fps"], 256)

    def snap(i_h):
        frames.append(np.hstack([hum[min(i_h, n - 1)], *env.render()]))

    # Approach: above the first pinch point, then straight down, gripper open.
    for k in range(int(APPROACH_S * CTRL_HZ)):
        z = z_high if k < APPROACH_S * CTRL_HZ / 2 else z_hold
        env.step_to([*ee_xy[0], z], -1)
        snap(0)
    b_start = env.bowl_pos()

    sim_bowl, tilt, lift = [], [], []
    for i in range(n):
        env.step_to([*ee_xy[i], z_hold], grip[i])
        p = env.bowl_pos()
        sim_bowl.append((p[:2] - b0[:2]) / D)
        tilt.append(env.bowl_tilt_deg())
        lift.append(p[2] - env.bowl_rest_z)
        snap(i)
    for _ in range(int(SETTLE_S * CTRL_HZ)):
        env.step_to([*ee_xy[-1], z_high], -1)
        snap(n - 1)
    final = (env.bowl_pos()[:2] - b0[:2]) / D
    env.close()

    sim_bowl = np.array(sim_bowl)
    tgt = human_to_robot(target_bowl)
    err = np.linalg.norm(sim_bowl - tgt, axis=1)
    res = dict(
        method="hand_motion_replay", video=name, hold_height_cm=round(hold_z * 100, 1), sim_bowl_diameter_cm=round(D * 100, 1),
        target_displacement_D=np.round(tgt[-1], 3).tolist(),
        achieved_displacement_D=np.round(final, 3).tolist(),
        final_error_D=round(float(np.linalg.norm(final - tgt[-1])), 3),
        mean_path_error_D=round(float(err.mean()), 3),
        bowl_moved_during_approach_D=round(float(np.linalg.norm(b_start[:2] - b0[:2]) / D), 3),
        max_tilt_deg=round(float(np.max(tilt)), 1),
        max_lift_cm=round(float(np.max(lift)) * 100, 1),
        grip_closed_frac=round(float((grip > 0).mean()), 2),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_hand_replay_z{round(hold_z * 1000)}mm"
    imageio.mimsave(out_dir / f"{tag}.mp4", frames, fps=CTRL_HZ)
    (out_dir / f"{tag}.json").write_text(json.dumps(res, indent=2))
    np.savez(out_dir / f"{tag}.npz", sim_bowl=sim_bowl, target=tgt, ee_xy=ee_xy, grip=grip)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("results/hand_replay"))
    args = ap.parse_args()
    for nm in args.names:
        for z in HOLD_ZS:
            r = run(nm, args.out, z)
            print(nm, f"z={r['hold_height_cm']}cm", "final_error_D", r["final_error_D"],
                  "achieved", r["achieved_displacement_D"], "target", r["target_displacement_D"],
                  "tilt", r["max_tilt_deg"])
