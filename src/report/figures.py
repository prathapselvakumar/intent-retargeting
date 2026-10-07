"""README figures, drawn from the evaluation reports in results/eval/.

Palette: categorical slots 1-4 of the dataviz reference palette, validated for CVD
separation on the light surface. Every bar carries its value label because slots 3 and 4
are below 3:1 contrast against the surface.

Usage: python src/report/figures.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SETS = ["random", "human_right", "human_left"]
SET_LABELS = ["Random start\n+ random goal", "Random start\n+ right-clip goal",
              "Random start\n+ left-clip goal"]
EVAL = Path("results/eval")


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pooled(runs):
    """Success counts per test set, summed over several evaluation runs."""
    out = {}
    for s in SETS:
        rs = [json.loads((EVAL / r / "report.json").read_text())[s] for r in runs]
        out[s] = (sum(r["success"] for r in rs), sum(r["n"] for r in rs))
    return out


def scenario1(out=Path("results/figures/scenario1_success.png")):
    hr = json.loads((EVAL / "hand_replay" / "report.json").read_text())
    seeds = range(3)
    series = [("Replay of my hand motion", {s: (hr[s]["success"], hr[s]["n"]) for s in SETS if s in hr}, BLUE),
              ("MLP policy, demos without my grip strategy", pooled([f"policy_none_s{i}" for i in seeds]), ORANGE),
              ("MLP policy, demos with my grip strategy", pooled([f"policy_strategy_s{i}" for i in seeds]), AQUA),
              ("SmolVLA from cameras, demos with my grip strategy",
               pooled([f"smolvla_v2_s{i}_k5" for i in seeds]), YELLOW)]
    plt.rcParams.update({"font.size": 11, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor=SURF)
    ax.set_facecolor(SURF)
    x, w = np.arange(len(SETS)), 0.2
    for i, (name, data, col) in enumerate(series):
        for j, s in enumerate(SETS):
            xx = x[j] + (i - 1.5) * w
            if s not in data:
                ax.text(xx, 4, "n/a", ha="center", va="bottom", color=INK2, fontsize=10)
                continue
            k, n = data[s]
            p = 100 * k / n
            lo, hi = wilson(k, n)
            ax.bar(xx, p, w, color=col, edgecolor=SURF, linewidth=2, zorder=3,
                   label=name if j == 1 else None)
            ax.errorbar(xx, p, yerr=[[p - 100 * lo], [100 * hi - p]], fmt="none", ecolor=INK2,
                        elinewidth=1.2, capsize=3, zorder=4)
            ax.text(xx, 100 * hi + 1.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, SET_LABELS)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Success rate (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Scenario 1, slide the bowl: success on held-out tasks (95% Wilson intervals)",
                 loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2, fontsize=10)
    fig.text(0.01, 0.01, "MLP policies and SmolVLA: 3 training seeds x 50 tasks per set (n = 150 per bar); SmolVLA "
             "executes 5 actions per query. Hand replay: 25 tasks per\nclip goal; it can only replay the two "
             "recorded motions, so it has no random-goal result. LIBERO default physics.", fontsize=8.5, color=INK2)
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor=SURF)
    return out


def scenario2(scene=Path("data/processed/scene/scenario2_plate.npz"),
              overlay=Path("data/processed/scene/scenario2_plate_scene.mp4"),
              out_dir=Path("results/scenario2")):
    """Scenario-2 extraction summary: key frames, bowl path relative to the plate, place and
    remove timing, and the lift cue. Also writes summary.json with the numbers."""
    import cv2
    d = np.load(scene)
    fps = float(d["fps"])
    r = d["bowl_rel_plate_D"]
    t = np.arange(len(r)) / fps
    on = d["on_plate"]
    edges = np.flatnonzero(np.diff(on.astype(int)))
    place_t, remove_t = edges[0] / fps, edges[-1] / fps
    cap = cv2.VideoCapture(str(overlay))
    keys = [0.5, place_t - 0.4, place_t + 1.5, remove_t + 0.3, t[-1] - 0.3]
    ims = []
    for k in keys:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * fps))
        _, f = cap.read()
        ims.append(cv2.cvtColor(cv2.resize(f, (384, 216)), cv2.COLOR_BGR2RGB))
    fig = plt.figure(figsize=(16, 8.5), facecolor=SURF)
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.25])
    for i, (im, lab) in enumerate(zip(ims, ["start", "lift and carry", "on plate", "take off", "end"])):
        a = fig.add_subplot(gs[0, i])
        a.imshow(im)
        a.set_title(f"{lab} ({keys[i]:.1f} s)")
        a.axis("off")
    a = fig.add_subplot(gs[1, 0:2])
    a.plot(r[:, 0], r[:, 1], "-", c=BLUE)
    a.add_patch(plt.Circle((0, 0), float(d["plate_to_bowl_diameter"]) / 2, fill=False, ec="#d55181", lw=2, label="plate"))
    a.add_patch(plt.Circle(tuple(r[0]), 0.5, fill=False, ec=INK2, ls="--", label="bowl at start"))
    a.set_aspect("equal")
    a.set_xlabel("x (bowl diameters, image right)")
    a.set_ylabel("y (away from me)")
    a.set_title("bowl centre relative to plate centre")
    a.legend(loc="lower right")
    a = fig.add_subplot(gs[1, 2:4])
    a.plot(t, np.linalg.norm(r, axis=1), c=BLUE, label="distance to plate centre (D)")
    a.axvspan(place_t, remove_t, color=AQUA, alpha=0.15, label="bowl on plate")
    a.set_xlabel("time (s)")
    a.legend()
    a.set_title("place and remove")
    a = fig.add_subplot(gs[1, 4])
    a.plot(t, d["bowl_rel_radius"], c=BLUE, label="bowl size (rest = 1)")
    a.plot(t, d["cam_rot_deg"] / 10 + 0.9, c=ORANGE, label="camera rotation (deg/10 + 0.9)")
    a.set_xlabel("time (s)")
    a.legend(fontsize=8)
    a.set_title("lift cue and camera drift")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "summary.png", dpi=80, facecolor=SURF)
    summ = dict(frames=len(r), bowl_tracked_frames=int(np.isfinite(r[:, 0]).sum()),
                min_registration_inliers=int(d["reg_inliers"][1:].min()),
                camera_rotation_deg_max=round(float(np.nanmax(np.abs(d["cam_rot_deg"]))), 1),
                plate_to_bowl_diameter=round(float(d["plate_to_bowl_diameter"]), 2),
                start_rel_plate_D=np.round(r[0], 2).tolist(), place_time_s=round(place_t, 2),
                remove_time_s=round(remove_t, 2), time_on_plate_s=round(remove_t - place_t, 1),
                placement_error_D=round(float(np.nanmin(np.linalg.norm(r, axis=1))), 3),
                max_bowl_size_increase_pct=round(100 * (float(np.nanmax(d["bowl_rel_radius"])) - 1), 1))
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=2))
    return out_dir / "summary.png"


if __name__ == "__main__":
    import sys
    print(scenario1())
    if "--scenario2" in sys.argv:
        print(scenario2())
