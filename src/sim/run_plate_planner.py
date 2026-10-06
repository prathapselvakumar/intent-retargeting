"""Scenario 2: pick-and-place planner in libero_goal #8 with vs without the human grip
prior, on LIBERO's official initial states.

Must be run as a file (not piped through stdin): the planner's process pool uses the
'spawn' start method, which re-imports the main module in every worker.

Usage: OMP_NUM_THREADS=1 python src/sim/run_plate_planner.py --inits 0 1 2 3 4 --workers 10
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from plate_task import PickPlanner, PlateEnv, execute, human_prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("results/scenario2/planner.json"))
    args = ap.parse_args()

    prior = human_prior()
    env = PlateEnv(cam_size=32, cameras=("agentview",))
    env.reset(0)
    D = env.bowl_diameter
    planner = PickPlanner(args.workers)
    out = []
    for init in args.inits:
        for cond, p in (("human", prior), ("none", None)):
            t = time.time()
            best, res, first, n = planner.plan(init, D, prior=p, rng=np.random.default_rng(init),
                                               pop=30, iters=3)
            env.reset(init)
            execute(env, best)
            r = dict(init=init, condition=cond, planned_success=bool(res[1]), first_success_at=first,
                     rollouts=n, executed_success=env.success(),
                     offset_D=round(float(np.linalg.norm(env.bowl_pos()[:2] - env.plate_pos()[:2]) / D), 3),
                     theta_deg=round(float(np.degrees(best.theta)), 1), lift_cm=round(best.lift * 100, 1),
                     seconds=round(time.time() - t))
            out.append(r)
            print("RESULT", r, flush=True)
            args.out.write_text(json.dumps(out, indent=2))
    planner.close()


if __name__ == "__main__":
    main()
