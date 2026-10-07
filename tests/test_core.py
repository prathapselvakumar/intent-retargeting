"""Fast tests for the geometry and bookkeeping the results depend on. No simulator needed.

Run: .venv/bin/python -m pytest tests -q
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(SRC / d) for d in ("extract", "sim", "policy")]

from bowl import fit_circle, robust_circle            # noqa: E402
from env import human_to_robot                        # noqa: E402
from evaluate import wilson                            # noqa: E402
from planner import human_strategy_prior, wrap_half    # noqa: E402
from train import chunk_targets                        # noqa: E402


def circle(cx, cy, r, n=200):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def test_fit_circle_exact():
    assert fit_circle(circle(310, 220, 75)) == pytest.approx((310, 220, 75), abs=1e-6)


def test_robust_circle_ignores_fingers_on_the_rim():
    pts = circle(310, 220, 75)
    pts[:40] = pts[:40] * 0.8 + np.array([310, 220]) * 0.2      # a finger dents 20% of the rim
    cx, cy, r, inliers = robust_circle(pts)
    assert (cx, cy, r) == pytest.approx((310, 220, 75), abs=0.5)
    assert inliers == pytest.approx(0.8, abs=0.01)


def test_human_to_robot_frame():
    # away from the person -> away from the robot (+x); person's right -> robot's right (-y)
    assert human_to_robot([0, 1]) == pytest.approx([1, 0])
    assert human_to_robot([1, 0]) == pytest.approx([0, -1])


def test_wilson_interval():
    assert wilson(0, 25) == pytest.approx((0.0, 13.3), abs=0.05)
    lo, hi = wilson(433, 450)
    assert lo < 100 * 433 / 450 < hi


def test_wrap_half_is_symmetric_for_a_parallel_gripper():
    for a in np.linspace(-3 * np.pi, 3 * np.pi, 37):
        w = wrap_half(a)
        assert -np.pi / 2 <= w <= np.pi / 2
        assert np.sin(2 * w) == pytest.approx(np.sin(2 * a), abs=1e-9)


def test_chunk_targets_pad_with_the_last_action():
    act = np.arange(5, dtype=float)[:, None]                  # two episodes: 3 and 2 steps
    t = chunk_targets(act, np.array([3, 2]), horizon=3)
    assert t.shape == (5, 3, 1)
    assert t[2, :, 0].tolist() == [2, 2, 2]                   # end of episode 1 holds its last action
    assert t[3, :, 0].tolist() == [3, 4, 4]                   # episode 2 never sees episode 1


def test_strategy_prior_is_relative_to_motion(tmp_path):
    # Two clips, opposite motions, each gripped 100 deg clockwise of the motion direction.
    files = []
    for name, rim, delta in (("a", 350.0, [0.0, 6.0]), ("b", 80.0, [-6.0, 0.0])):
        seg = dict(moved_cm=6.0, rim_angle_deg=rim, bowl_delta_cm=delta, mode="drag")
        f = tmp_path / f"{name}.json"
        f.write_text(json.dumps(dict(segments=[seg])))
        files.append(f)
    p = human_strategy_prior(files)
    assert np.degrees(p["theta_rel"]) == pytest.approx(-100, abs=1e-6)
    assert p["mode_p"]["pinch"] > p["mode_p"]["push"]
