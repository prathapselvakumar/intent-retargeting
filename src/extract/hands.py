"""Per-frame hand landmarks from a phone video using MediaPipe HandLandmarker.

Output (.npz per video):
  img_xy     [T, 2, 21, 2]  normalized image coords for up to 2 hands (NaN if missing)
  world_xyz  [T, 2, 21, 3]  metric hand-centric coords (m), origin near the hand centre
  score      [T, 2]         detection confidence
  fps, width, height

Hands are kept in a stable slot order by image x position (slot 0 = left side of image,
slot 1 = right side), because MediaPipe's handedness labels assume a mirrored selfie
camera and are unreliable for a fixed rear camera.

Usage: python src/extract/hands.py data/raw/episode_000_right.mov [...] --out data/processed/hands
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"
PROC_WIDTH = 960  # downscale 4K input; landmarks are normalized so resolution only affects accuracy/speed

# MediaPipe landmark ids
WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8
HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def make_landmarker():
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def extract(video_path: Path, out_dir: Path, render: bool = True):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pw, ph = PROC_WIDTH, round(h * PROC_WIDTH / w)

    img_xy, world_xyz, score = [], [], []
    writer = None
    if render:
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_dir / f"{video_path.stem}_overlay.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw, ph))

    with make_landmarker() as lm:
        t = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                      int(t * 1000 / fps))
            t += 1

            xy = np.full((2, 21, 2), np.nan, np.float32)
            wx = np.full((2, 21, 3), np.nan, np.float32)
            sc = np.zeros(2, np.float32)
            hands = [(np.array([[p.x, p.y] for p in l2d], np.float32),
                      np.array([[p.x, p.y, p.z] for p in l3d], np.float32),
                      hd[0].score)
                     for l2d, l3d, hd in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness)]
            hands.sort(key=lambda hnd: hnd[0][WRIST, 0])  # slot by image x
            if len(hands) == 1:  # a lone hand goes to whichever side of the image it is on
                slots = [0 if hands[0][0][WRIST, 0] < 0.5 else 1]
            else:
                slots = list(range(len(hands)))
            for s, (a, b, c) in zip(slots, hands):
                xy[s], wx[s], sc[s] = a, b, c
            img_xy.append(xy); world_xyz.append(wx); score.append(sc)

            if writer is not None:
                draw(frame, xy, wx)
                writer.write(frame)

    cap.release()
    if writer is not None:
        writer.release()

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{video_path.stem}.npz"
    np.savez_compressed(out, img_xy=np.stack(img_xy), world_xyz=np.stack(world_xyz),
                        score=np.stack(score), fps=fps, width=w, height=h)
    return out


def draw(frame, xy, wx):
    ph, pw = frame.shape[:2]
    colors = [(255, 160, 0), (0, 200, 255)]  # slot 0 blue-ish, slot 1 orange-ish (BGR)
    for s in range(2):
        if np.isnan(xy[s, 0, 0]):
            continue
        pts = (xy[s] * [pw, ph]).astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), colors[s], 2)
        ap = np.linalg.norm(wx[s, THUMB_TIP] - wx[s, INDEX_TIP]) * 100
        cv2.putText(frame, f"aperture {ap:.1f} cm", tuple(pts[WRIST] + [0, 25]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[s], 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/processed/hands"))
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    for v in args.videos:
        print(f"{v.name} -> {extract(v, args.out, render=not args.no_render)}")
