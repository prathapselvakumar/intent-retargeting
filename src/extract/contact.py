"""Fuse hand landmarks and bowl track into the task 'intent': where and when the hand
touched the bowl, and what the bowl did as a result.

A frame is in contact when any fingertip of the active hand is within CONTACT_TOL of the
rim circle. The contact point is reported in the bowl frame as an angle around the rim
(0 = image right, 90° = away from the person), so it transfers to any embodiment and
any bowl position.

Output (.json per video): segments of {start_s, end_s, rim_angle_deg, bowl_delta_cm,
mode} plus the active hand slot. Contacts that don't move the bowl (a resting hand next
to it) are dropped. The mode comes from the angle between the bowl's motion and the
inward contact normal:
  push  — motion along the normal: a closed gripper or a fingertip can do this
  drag  — motion tangential to the normal: needs friction or a grasp on the rim
  pull  — motion toward the hand: needs a hook or a grasp

Usage: python src/extract/contact.py episode_000_right episode_050_left
"""
import argparse
import json
from pathlib import Path

import numpy as np

FINGERTIPS = [4, 8, 12, 16, 20]
CONTACT_TOL = 0.12     # fraction of rim radius
MIN_SEG_S = 0.2
MIN_MOVE_CM = 1.0    # contacts that move the bowl less than this are resting hands


def analyse(name, hands_dir: Path, bowl_dir: Path):
    hd = np.load(hands_dir / f"{name}.npz")
    bd = np.load(bowl_dir / f"{name}.npz")
    fps = float(bd["fps"])
    T = min(len(hd["img_xy"]), len(bd["center_px"]))
    w, h = 960, round(int(hd["height"]) * 960 / int(hd["width"]))

    xy = hd["img_xy"][:T] * [w, h]                       # [T, 2, 21, 2] px
    motion = np.nansum(np.linalg.norm(np.diff(xy[:, :, 0], axis=0), axis=-1), axis=0)
    act = int(np.argmax(motion))
    tips = xy[:, act, FINGERTIPS]                         # [T, 5, 2]

    c, r = bd["center_px"][:T], bd["radius_px"][:T]
    d = np.linalg.norm(tips - c[:, None], axis=-1)        # [T, 5]
    rim_dist = np.abs(d - r[:, None]) / r[:, None]
    contact = np.nanmin(rim_dist, axis=1) < CONTACT_TOL

    # Close 1–2 frame dropouts, then split into segments.
    k = 3
    contact = np.convolve(contact.astype(float), np.ones(k) / k, "same") > 0.34
    edges = np.flatnonzero(np.diff(np.r_[0, contact.astype(int), 0]))
    segs = []
    bowl_xy = bd["xy_m"][:T]
    for s, e in zip(edges[::2], edges[1::2]):
        if (e - s) / fps < MIN_SEG_S:
            continue
        mid_tips = np.nanmean(tips[s:e], axis=(0, 1))
        mid_c = np.nanmean(c[s:e], axis=0)
        v = mid_tips - mid_c
        angle = np.degrees(np.arctan2(-v[1], v[0])) % 360   # image y down → flip
        # Bowl displacement attributed to this contact (allow 0.3 s of follow-through).
        e2 = min(T - 1, e + int(0.3 * fps))
        delta = (bowl_xy[e2] - bowl_xy[s]) * 100
        moved = float(np.linalg.norm(delta))
        if moved < MIN_MOVE_CM:
            continue
        inward = -np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        cos = float(delta @ inward) / moved
        mode = "push" if cos > 0.5 else ("pull" if cos < -0.5 else "drag")
        segs.append(dict(start_s=round(s / fps, 2), end_s=round(e / fps, 2),
                         rim_angle_deg=round(float(angle), 1),
                         bowl_delta_cm=[round(float(x), 2) for x in delta],
                         moved_cm=round(moved, 2), mode=mode,
                         motion_vs_normal_deg=round(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))), 1)))
    return dict(video=name, active_slot=act, fps=fps, segments=segs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--hands", type=Path, default=Path("data/processed/hands"))
    ap.add_argument("--bowl", type=Path, default=Path("data/processed/bowl"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/intent"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for n in args.names:
        res = analyse(n, args.hands, args.bowl)
        (args.out / f"{n}.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
