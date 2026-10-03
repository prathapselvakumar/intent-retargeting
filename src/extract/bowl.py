"""Metric bowl trajectory from a top-down phone video — no depth sensor, no learned model.

The camera looks straight down at a flat table, so the table is (to first order) a plane
parallel to the image. A black bowl on a light table segments with a brightness threshold;
a robust circle fit gives its centre and apparent radius each frame. Two physical facts
turn pixels into metres:

  * the bowl's real rim diameter D fixes the camera distance  Z = f * D / (2 r)
  * the bowl is rigid, so any growth of its apparent radius r means it moved toward the
    camera, i.e. was lifted:  height = Z_table - f * D / (2 r)

Output (.npz per video):
  center_px [T, 2]  rim centre in processing-resolution pixels (NaN if not found)
  radius_px [T]     apparent rim radius in pixels
  xy_m      [T, 2]  rim centre on the table plane in metres, origin = first-frame position,
                    +x to image right, +y to image top (away from the person)
  height_m  [T]     rim height above its resting height
  inlier    [T]     fraction of rim contour points consistent with the circle (fit quality)

Usage: python src/extract/bowl.py data/raw/episode_000_right.mov --diameter-cm 15
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

PROC_WIDTH = 960
DARK_V = 60            # HSV value threshold for the black bowl
BORDER = 4             # contour points this close to the image border are clipping, not rim
# iPhone 15 Plus main camera, 4K 16:9 video: ~26 mm equivalent → ~70° horizontal FOV.
# Only used to convert apparent-radius change into height; the in-plane trajectory
# depends on D alone.
HFOV_DEG = 70.0


def fit_circle(pts):
    """Algebraic (Kåsa) least-squares circle fit. pts: [N, 2] → (cx, cy, r)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    (cx, cy, c), *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    return cx, cy, np.sqrt(c + cx**2 + cy**2)


def robust_circle(pts, iters=5, tol=0.03):
    """Iteratively drop points off the circle (fingers notching the rim, highlights)."""
    keep = np.ones(len(pts), bool)
    for _ in range(iters):
        cx, cy, r = fit_circle(pts[keep])
        resid = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) / r
        new = resid < tol
        if new.sum() < 20 or (new == keep).all():
            break
        keep = new
    return cx, cy, r, keep.mean()


def detect(frame):
    h, w = frame.shape[:2]
    v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 2]
    mask = (v < DARK_V).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    blob = (lab == big).astype(np.uint8)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cnts, key=len)[:, 0, :].astype(np.float64)
    on_border = ((pts[:, 0] < BORDER) | (pts[:, 0] > w - 1 - BORDER) |
                 (pts[:, 1] < BORDER) | (pts[:, 1] > h - 1 - BORDER))
    pts = pts[~on_border]
    if len(pts) < 50:
        return None
    return robust_circle(pts)


def track(video_path: Path, diameter_m: float, out_dir: Path, render=True):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pw, ph = PROC_WIDTH, round(h * PROC_WIDTH / w)
    f_px = (pw / 2) / np.tan(np.radians(HFOV_DEG) / 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_dir / f"{video_path.stem}_bowl.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph)) if render else None

    centers, radii, inl = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        det = detect(frame)
        if det is None:
            centers.append((np.nan, np.nan)); radii.append(np.nan); inl.append(0.0)
        else:
            cx, cy, r, q = det
            centers.append((cx, cy)); radii.append(r); inl.append(q)
            if writer is not None:
                cv2.circle(frame, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 4, (0, 255, 0), -1)
        if writer is not None:
            writer.write(frame)
    cap.release()
    if writer is not None:
        writer.release()

    c = np.array(centers)
    r = np.array(radii)
    # Resting radius: the bowl sits on the table at the start and end of every clip.
    r_rest = np.nanmedian(np.r_[r[:15], r[-15:]])
    z_table = f_px * diameter_m / (2 * r_rest)              # camera → rim distance at rest
    z = f_px * diameter_m / (2 * r)
    height = z_table - z
    # Back-project pixel centre onto the plane at the rim's current depth.
    principal = np.array([pw / 2, ph / 2])
    xy = (c - principal) * (z / f_px)[:, None]
    xy = xy - xy[np.flatnonzero(~np.isnan(xy[:, 0]))[0]]
    xy[:, 1] *= -1  # image y points down; make +y point away from the person

    out = out_dir / f"{video_path.stem}.npz"
    np.savez_compressed(out, center_px=c, radius_px=r, xy_m=xy, height_m=height,
                        inlier=np.array(inl), fps=fps, z_table_m=z_table,
                        diameter_m=diameter_m, f_px=f_px)
    return out, z_table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--diameter-cm", type=float, default=15.0,
                    help="measured bowl rim diameter (placeholder until measured)")
    ap.add_argument("--out", type=Path, default=Path("data/processed/bowl"))
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    for v in args.videos:
        out, z = track(v, args.diameter_cm / 100, args.out, render=not args.no_render)
        print(f"{v.name} -> {out}  (camera height above rim ≈ {z * 100:.0f} cm)")
