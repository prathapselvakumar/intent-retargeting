"""Bowl + plate tracking for a hand-held top-down clip (scenario 2: bowl onto plate and off).

Unlike scenario 1 the phone was hand-held, so the image drifts. Camera motion is
estimated frame-to-frame on static background (ORB features on everything except bowl,
plate and hand; RANSAC similarity transform) and chained back to frame 0. The plate never
moves on the table, so it is detected once in frame 0 and then carried along by the
camera motion — re-detecting it fails once the bowl covers all but a thin ring. The bowl
is tracked with continuity priors and mapped into frame-0 coordinates, then expressed
relative to the plate.

Units are bowl diameters (D), as elsewhere. Lift comes for free: the bowl's apparent
radius relative to the plate's (both rigid, plate always on the table) grows when the
bowl is raised toward the camera.

Output (.npz): bowl_rel_plate_D [T,2] (bowl centre minus plate centre, in bowl diameters,
camera rotation removed, +x image right, +y away from the person), bowl_rel_radius [T]
(bowl/plate radius ratio normalised to rest → >1 means lifted), on_plate [T],
cam_rot_deg [T], reg_inliers [T].

Usage: python src/extract/scene_plate.py data/raw/scenario2_plate.mov
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from bowl import robust_circle

PROC_WIDTH = 960
BORDER = 4


def blob_circle(mask, min_area=3000):
    """Largest roughly-circular blob → (cx, cy, r, fit_quality) or None."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    best = None
    h, w = mask.shape
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        blob = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = max(cnts, key=len)[:, 0, :].astype(np.float64)
        keep = ~((pts[:, 0] < BORDER) | (pts[:, 0] > w - 1 - BORDER) |
                 (pts[:, 1] < BORDER) | (pts[:, 1] > h - 1 - BORDER))
        if keep.sum() < 50:
            continue
        cx, cy, r, q = robust_circle(pts[keep])
        # Score: fit quality × how much of the circle's area the blob fills.
        fill = stats[i, cv2.CC_STAT_AREA] / (np.pi * r * r + 1e-9)
        score = q * min(fill, 1.0 / max(fill, 1e-9))
        if r < 30 or r > 0.6 * h:
            continue
        if best is None or score > best[-1]:
            best = (cx, cy, r, q, score)
    return None if best is None else best[:4]


def tracked_circle(mask, prior, search=1.6, r_tol=0.25):
    """Circle fit on all mask contours inside a disc around the previous estimate."""
    cx, cy, r = prior[:3]
    win = np.zeros_like(mask)
    cv2.circle(win, (int(cx), int(cy)), int(r * search), 1, -1)
    m = cv2.morphologyEx(mask & win, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 200]
    if not cnts:
        return None
    pts = np.concatenate([c[:, 0, :] for c in cnts]).astype(np.float64)
    h, w = mask.shape
    keep = ~((pts[:, 0] < BORDER) | (pts[:, 0] > w - 1 - BORDER) |
             (pts[:, 1] < BORDER) | (pts[:, 1] > h - 1 - BORDER))
    # Only the outer boundary is the rim: drop points well inside the prior circle.
    keep &= np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) > 0.7 * r
    if keep.sum() < 40:
        return None
    ncx, ncy, nr, q = robust_circle(pts[keep])
    if abs(nr - r) > r_tol * r or np.hypot(ncx - cx, ncy - cy) > 0.8 * r:
        return None
    return ncx, ncy, nr, q


def segment(frame, prev_bowl=None, prev_plate=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    dark = (v < 55).astype(np.uint8)
    white = ((v > 200) & (s < 40)).astype(np.uint8)
    bowl = blob_circle(dark) if prev_bowl is None else \
        (tracked_circle(dark, prev_bowl, search=1.8, r_tol=0.3) or None)
    plate = blob_circle(white) if prev_plate is None else None
    return bowl, plate


def static_mask(frame, bowl, plate, hand_px):
    m = np.full(frame.shape[:2], 255, np.uint8)
    for c in (bowl, plate):
        if c is not None:
            cv2.circle(m, (int(c[0]), int(c[1])), int(c[2] * 1.25), 0, -1)
    if hand_px is not None and np.isfinite(hand_px).all():
        x0, y0 = hand_px.min(0) - 60
        x1, y1 = hand_px.max(0) + 60
        cv2.rectangle(m, (int(x0), int(y0)), (int(x1), int(y1)), 0, -1)
        cv2.rectangle(m, (int(x0), int(y1)), (int(x1), frame.shape[0]), 0, -1)  # forearm
    return m


def track(video: Path, hands_npz: Path, out_dir: Path):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    hd = np.load(hands_npz)
    orb = cv2.ORB_create(3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_kp = prev_des = None
    H_acc = np.eye(3)
    last_bowl = last_plate = plate0 = None
    rows = []
    writer = None
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (PROC_WIDTH, round(h * PROC_WIDTH / w)), interpolation=cv2.INTER_AREA)
        ph, pw = frame.shape[:2]
        bowl, plate = segment(frame, last_bowl, None if plate0 is None else "skip")
        last_bowl = bowl or last_bowl
        if plate0 is None:
            plate0 = plate
        # Plate in the current frame = frame-0 plate moved by the inverse camera motion.
        Hi = np.linalg.inv(H_acc)
        pc = Hi @ np.r_[plate0[0], plate0[1], 1.0]
        last_plate = (pc[0], pc[1], plate0[2] / np.sqrt(abs(np.linalg.det(H_acc[:2, :2]))))
        plate = last_plate
        hand = hd["img_xy"][t, 0] * [pw, ph] if t < len(hd["img_xy"]) else None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, static_mask(frame, last_bowl, last_plate, hand))
        inl = 0
        if prev_des is not None and des is not None:
            ms = bf.match(des, prev_des)
            if len(ms) >= 12:
                src = np.float32([kp[m.queryIdx].pt for m in ms])
                dst = np.float32([prev_kp[m.trainIdx].pt for m in ms])
                A, msk = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                     ransacReprojThreshold=2.0)
                if A is not None:
                    inl = int(msk.sum())
                    H_acc = H_acc @ np.vstack([A, [0, 0, 1]])   # current → frame 0
        prev_kp, prev_des = kp, des
        rot = np.degrees(np.arctan2(H_acc[1, 0], H_acc[0, 0]))
        scale = np.sqrt(abs(np.linalg.det(H_acc[:2, :2])))
        if bowl:
            b0 = H_acc @ np.r_[bowl[0], bowl[1], 1.0]
            rows.append((b0[0], b0[1], bowl[2] * scale, inl, rot, *bowl[:3]))   # frame-0 + current
        else:
            rows.append((np.nan, np.nan, np.nan, inl, rot, np.nan, np.nan, np.nan))
        if writer is None:
            out_dir.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_dir / f"{video.stem}_scene.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph))
        for c, col in ((bowl, (0, 255, 0)), (plate, (255, 0, 255))):
            if c is not None:
                cv2.circle(frame, (int(c[0]), int(c[1])), int(c[2]), col, 2)
        cv2.putText(frame, f"reg inliers {inl}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        writer.write(frame)
        t += 1
    cap.release()
    writer.release()

    a = np.array(rows, float)
    bowl_c, bowl_r, inl, rot = a[:, 0:2], a[:, 2], a[:, 3], a[:, 4]
    bowl_cur = a[:, 5:8]                                            # current-frame px (cx, cy, r)
    k = int(fps)
    r_rest = np.nanmedian(np.r_[bowl_r[:k], bowl_r[-k:]])         # bowl on the table
    D = 2 * r_rest                                                  # frame-0 px per bowl diameter
    rel = (bowl_c - np.array(plate0[:2])) / D
    rel[:, 1] *= -1                                                 # +y away from the person
    lift = bowl_r / r_rest
    on_plate = np.linalg.norm(rel, axis=1) < 0.25
    np.savez_compressed(out_dir / f"{video.stem}.npz", bowl_rel_plate_D=rel, bowl_rel_radius=lift,
                        on_plate=on_plate, cam_rot_deg=rot, reg_inliers=inl, fps=fps,
                        plate_to_bowl_diameter=float(plate0[2] / r_rest), bowl_cur_px=bowl_cur)
    return out_dir / f"{video.stem}.npz"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--hands", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/processed/scene"))
    args = ap.parse_args()
    hands = args.hands or Path("data/processed/hands") / f"{args.video.stem}.npz"
    print(track(args.video, hands, args.out))
