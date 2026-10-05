"""README figures, drawn from the evaluation reports in results/eval/.

Palette: categorical slots 1–3 of the dataviz reference palette (validated for CVD
separation on the light surface); every bar carries its value label because slot 3 is
below 3:1 contrast against the surface.

Usage: python src/report/figures.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3dd"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
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


def pooled(prefix):
    out = {}
    for s in SETS:
        k = n = 0
        for suf in ("", "_s1", "_s2"):
            r = json.loads((EVAL / f"policy_{prefix}{suf}" / "report.json").read_text())[s]
            k, n = k + r["success"], n + r["n"]
        out[s] = (k, n)
    return out


def scenario1(out=Path("results/figures/scenario1_success.png")):
    hr = json.loads((EVAL / "hand_replay" / "report.json").read_text())
    series = [("Copy the hand motion", {s: (hr[s]["success"], hr[s]["n"]) for s in SETS if s in hr}, BLUE),
              ("Copy the effect — no human strategy", pooled("none"), ORANGE),
              ("Copy the effect — with human strategy", pooled("strategy"), AQUA)]
    plt.rcParams.update({"font.size": 11, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(figsize=(10, 5.4), facecolor=SURF)
    ax.set_facecolor(SURF)
    x, w = np.arange(len(SETS)), 0.26
    for i, (name, data, col) in enumerate(series):
        for j, s in enumerate(SETS):
            xx = x[j] + (i - 1) * w
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
            ax.text(xx, 100 * hi + 1.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, SET_LABELS)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Success rate (%)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Scenario 1 (slide the bowl): held-out tasks in LIBERO, 95% Wilson intervals",
                 loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, fontsize=10)
    fig.text(0.01, 0.01, "Policies: 3 training seeds × 50 episodes per test set (n = 150). "
             "Hand replay: n = 25 per clip goal; it can only\nreplay the two recorded motions, "
             "so it has no random-goal result.", fontsize=8.5, color=INK2)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor=SURF)
    return out


if __name__ == "__main__":
    print(scenario1())
