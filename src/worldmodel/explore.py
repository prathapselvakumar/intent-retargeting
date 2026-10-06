"""Exploration data for the bowl world model: random contact primitives, not just good plans.

The planner's demos only show successful, stereotyped interactions (approach from above,
pinch, drag). A world model also needs failures and odd contacts — glancing pushes, pokes
from any side, grips at any height, hands sliding along the rim — or it extrapolates badly
on anything new (like a human hand). Each rollout samples a primitive with uniformly
random parameters from a random bowl start and records the trajectory.

Output: data/demos/explore.npz with the same per-step fields as sim_episodes() uses.

Usage: python src/worldmodel/explore.py --n 3000 --workers 8
"""
import argparse
import multiprocessing as mp
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "sim"))

_W = None


def _init():
    global _W
    os.environ.setdefault("MUJOCO_GL", "egl")
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    from env import BowlEnv
    _W = BowlEnv(cam_size=32, cameras=("birdview",))


def _collect(seed):
    from planner import MODES, Prim, execute
    rng = np.random.default_rng(seed)
    start = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.12, 0.12)])
    _W.reset(bowl_xy=start)
    prim = Prim(mode=MODES[rng.integers(3)], theta=rng.uniform(-np.pi, np.pi),
                h=rng.uniform(0.005, 0.05), phi=rng.uniform(-np.pi, np.pi), d=rng.uniform(0.0, 0.12))
    rec = dict(e=[], z=[], b=[], opening=[], yaw=[])

    def snap():
        e, b = _W.eef_pos(), _W.bowl_pos()
        rec["e"].append(e[:2]); rec["z"].append(e[2] - _W.table_z); rec["b"].append(b[:2])
        rec["opening"].append(_W.gripper_opening()); rec["yaw"].append(_W.eef_yaw())
    snap()
    execute(_W, prim, on_step=snap)
    return {k: np.array(v) for k, v in rec.items()} | dict(prim=asdict(prim))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "data/demos/explore.npz")
    args = ap.parse_args()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        eps = pool.map(_collect, range(10_000, 10_000 + args.n), chunksize=8)
    L = np.array([len(e["e"]) for e in eps])
    np.savez_compressed(args.out, ep_len=L, **{k: np.concatenate([e[k] for e in eps])
                                              for k in ("e", "z", "b", "opening", "yaw")})
    moved = np.mean([np.linalg.norm(e["b"][-1] - e["b"][0]) > 0.005 for e in eps])
    print(f"{len(eps)} episodes, {L.sum()} transitions, bowl moved in {moved:.0%} of episodes → {args.out}")


if __name__ == "__main__":
    main()
