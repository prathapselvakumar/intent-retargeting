"""Generate robot demonstrations with the intent planner, for policy distillation.

Each episode: random bowl start on the table, random goal displacement (direction uniform,
0.25–0.6 bowl diameters, the range of the human clips), plan with CEM, execute, keep only
successes. Observations are `BowlEnv.policy_obs`, actions the 20 Hz OSC commands
(dx, dy, dz, dyaw, grip).

Usage: python src/sim/gen_demos.py --prior strategy --n 300
       python src/sim/gen_demos.py --prior none --n 300
"""
import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from env import BowlEnv
from planner import SUCCESS_D, SUCCESS_TILT, Planner, execute, human_strategy_prior

MAX_PRIMS = 2
HOLD_STEPS = 10
ACT_IDX = [0, 1, 2, 5, 6]          # dx dy dz dyaw grip  (roll/pitch stay zero)


def sample_task(rng, D):
    start = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.12, 0.12)])
    ang = rng.uniform(-np.pi, np.pi)
    mag = rng.uniform(0.25, 0.6) * D
    return start, start + mag * np.array([np.cos(ang), np.sin(ang)])


def run(env, planner, rng, prior, pop, iters):
    env.reset()
    D = env.bowl_diameter
    start, goal = sample_task(rng, D)
    env.reset(bowl_xy=start)
    goal = goal + (env.bowl_pos()[:2] - start)          # goal relative to settled pose
    obs, acts = [env.policy_obs(goal)], []

    def rec():
        acts.append(env.last_action[ACT_IDX].copy())
        obs.append(env.policy_obs(goal))

    n_roll, prims = 0, []
    for _ in range(MAX_PRIMS):
        prim, _, _, n = planner.plan(env.get_state(), goal, env.bowl_pos()[:2], D,
                                     prior=prior, rng=rng, pop=pop, iters=iters)
        n_roll += n
        execute(env, prim, on_step=rec)
        prims.append(prim.mode)
        err = np.linalg.norm(env.bowl_pos()[:2] - goal) / D
        if err < SUCCESS_D and env.bowl_tilt_deg() < SUCCESS_TILT:
            break
    ok = bool(err < SUCCESS_D and env.bowl_tilt_deg() < SUCCESS_TILT)
    for _ in range(HOLD_STEPS):                          # teach "stop when done"
        env.step_to(env.eef_pos(), env.last_action[6], yaw=env.eef_yaw())
        rec()
    return ok, dict(obs=np.array(obs[:-1]), act=np.array(acts), start=start, goal=goal,
                    err=float(err), prims=prims, rollouts=n_roll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["strategy", "none"], required=True)
    ap.add_argument("--n", type=int, default=300, help="successful episodes to collect")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("data/demos"))
    args = ap.parse_args()

    prior = human_strategy_prior(sorted(glob.glob("data/processed/intent/*.json"))) \
        if args.prior == "strategy" else None
    rng = np.random.default_rng(args.seed)
    planner, env = Planner(args.workers), BowlEnv(cam_size=64, cameras=("birdview",))
    eps, tried, t0 = [], 0, time.time()
    while len(eps) < args.n:
        ok, ep = run(env, planner, rng, prior, args.pop, args.iters)
        tried += 1
        if ok:
            eps.append(ep)
        if tried % 10 == 0:
            print(f"[{args.prior}] {len(eps)}/{args.n} kept of {tried} tried "
                  f"({(time.time() - t0) / tried:.1f}s/ep)", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / f"demos_{args.prior}.npz",
        obs=np.concatenate([e["obs"] for e in eps]), act=np.concatenate([e["act"] for e in eps]),
        ep_len=np.array([len(e["act"]) for e in eps]),
        start=np.array([e["start"] for e in eps]), goal=np.array([e["goal"] for e in eps]))
    meta = dict(prior=args.prior, kept=len(eps), tried=tried,
                planner_success_rate=round(len(eps) / tried, 3),
                mean_rollouts=float(np.mean([e["rollouts"] for e in eps])),
                modes={m: sum(e["prims"][-1] == m for e in eps) for m in ("push", "inside", "pinch")},
                seconds=round(time.time() - t0))
    (args.out / f"demos_{args.prior}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    env.close()
    planner.close()


if __name__ == "__main__":
    main()
