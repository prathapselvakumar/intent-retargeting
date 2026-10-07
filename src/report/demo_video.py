"""Short demo video (about 70 s) assembled from the result videos, with every number on screen
read from the result files.

Usage: .venv/bin/python src/report/demo_video.py      -> results/figures/demo.mp4
"""
import json
from pathlib import Path

import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 20
BG, INK, MUTED, ACCENT = (26, 26, 25), (255, 255, 255), (195, 194, 183), (27, 175, 122)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SETS = ("random", "human_right", "human_left")
EVAL = Path("results/eval")


def rate(runs):
    """Pooled success over several evaluation runs, as (successes, tasks)."""
    rs = [json.loads((EVAL / r / "report.json").read_text()) for r in runs]
    return sum(r[s]["success"] for r in rs for s in SETS if s in r), sum(r[s]["n"] for r in rs for s in SETS if s in r)


def text_frame(lines, base=None):
    """lines: list of (text, size, colour, bold). Drawn centred, or as a caption bar on `base`."""
    img = Image.new("RGB", (W, H), BG) if base is None else Image.fromarray(base)
    d = ImageDraw.Draw(img)
    fonts = [ImageFont.truetype(BOLD if b else FONT, sz) for _, sz, _, b in lines]
    heights = [f.getbbox(t)[3] + 14 for (t, *_), f in zip(lines, fonts)]
    y = (H - sum(heights)) // 2 if base is None else H - sum(heights) - 18
    if base is not None:
        d.rectangle([0, y - 14, W, H], fill=BG)
    for (t, _, col, _), f, h in zip(lines, fonts, heights):
        d.text(((W - d.textlength(t, font=f)) / 2, y), t, font=f, fill=col)
        y += h
    return np.array(img)


def card(lines, seconds):
    return [text_frame(lines)] * int(seconds * FPS)


def clip(path, caption, seconds=None, speed=1.0, hold=1.0):
    """Video letterboxed into the frame with a caption bar; slowed or sped up by `speed`."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    raw = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        raw.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    n_out = int(len(raw) / src_fps * FPS / speed)
    if seconds:
        n_out = min(n_out, int(seconds * FPS))
    idx = np.linspace(0, len(raw) - 1, n_out).round().astype(int)
    out = []
    for i in idx:
        f = raw[i]
        s = min(W / f.shape[1], (H - 130) / f.shape[0])
        f = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)), interpolation=cv2.INTER_AREA)
        canvas = np.full((H, W, 3), BG, np.uint8)
        y0, x0 = (H - 130 - f.shape[0]) // 2 + 10, (W - f.shape[1]) // 2
        canvas[y0:y0 + f.shape[0], x0:x0 + f.shape[1]] = f
        out.append(text_frame(caption, canvas))
    return out + [out[-1]] * int(hold * FPS)


def main():
    hr = rate(["hand_replay"])
    with_s = rate([f"policy_strategy_s{i}" for i in range(3)])
    without = rate([f"policy_none_s{i}" for i in range(3)])
    vla = rate([f"smolvla_v2_s{i}_k5" for i in range(3)])
    s2_replay = json.loads(Path("results/scenario2/hand_replay.json").read_text())
    pct = lambda kn: f"{100 * kn[0] / kn[1]:.1f}%"
    frames = []
    frames += card([("Intent retargeting", 54, INK, True),
                    ("From my phone videos to a Panda robot in LIBERO", 30, MUTED, False),
                    ("Prathap Selvakumar", 26, MUTED, False)], 3)
    frames += clip("data/clips/scenario1_right_hand.mp4",
                   [("My data: three iPhone clips of my own hands", 30, INK, True),
                    ("No depth sensor. The bowl is tracked directly; my grip on the rim is measured.", 22, MUTED, False)],
                   seconds=5)
    frames += clip("results/hand_replay/episode_000_right_hand_replay_z30mm.mp4",
                   [("Copying my hand motion fails", 30, INK, True),
                    (f"{hr[0] + sum(r['success'] for r in s2_replay)} of {hr[1] + len(s2_replay)} tasks across "
                     "both scenarios", 22, MUTED, False)], seconds=6, speed=1.5)
    frames += clip("results/planner/episode_050_left_human_prior.mp4",
                   [("Copying what happened to the bowl works", 30, INK, True),
                    ("A planner seeded with my grip strategy reproduces the bowl's motion", 22, MUTED, False)],
                   speed=0.7)
    frames += clip("results/eval/policy_strategy_s0/random_0.mp4",
                   [(f"Policy trained on those demos: {pct(with_s)} success", 30, ACCENT, True),
                    (f"vs {pct(without)} without my grip strategy (3 seeds, {with_s[1]} held-out tasks)", 22, MUTED, False)],
                   seconds=6)
    frames += clip("results/eval/smolvla_v2_s0_k5/human_left_0.mp4",
                   [(f"SmolVLA from camera images only: {pct(vla)}", 30, INK, True),
                    ("450M-parameter vision-language-action model, fine-tuned with LeRobot", 22, MUTED, False)],
                   seconds=6)
    frames += clip("results/scenario2/robot_success_init1.mp4",
                   [("Scenario 2: LIBERO's official 'put the bowl on the plate' task", 28, INK, True),
                    ("My grip angle and lift height from the video, LIBERO's own success check", 22, MUTED, False)],
                   speed=0.8)
    frames += card([("github.com/prathapselvakumar/intent-retargeting", 34, INK, True),
                    ("Every number in this video is computed by the code in the repository", 22, MUTED, False)], 3)
    out = Path("results/figures/demo.mp4")
    imageio.mimsave(out, frames, fps=FPS, codec="libx264", quality=7, macro_block_size=8)
    print(out, f"{len(frames) / FPS:.0f} s")


if __name__ == "__main__":
    main()
