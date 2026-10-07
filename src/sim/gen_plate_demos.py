"""Scenario-2 demonstrations for policy training: put the bowl on the plate in libero_goal task 8.

Each episode starts from a random official initial state with the bowl and plate shifted by up
to `--jitter` metres, so training starts differ from the 50 official ones used for evaluation.
The pick-and-place planner (with or without my grip prior) chooses a primitive, the main env
executes it while recording the policy observation (`PlateEnv.plate_obs`) and actions, and only
episodes that LIBERO's own success check accepts are kept. Start states are saved so the
episodes can be replayed with cameras for the vision dataset.

Run as a file (the planner pool uses 'spawn'):
  OMP_NUM_THREADS=1 PYTHONPATH=src/sim python src/sim/gen_plate_demos.py --prior human --n 200
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from plate_task import PickPlanner, PlateEnv, execute, human_prior

ACT_IDX = [0, 1, 2, 5, 6]
HOLD_STEPS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", choices=["human", "none"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--jitter", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("data/demos"))
    args = ap.parse_args()

    prior = human_prior() if args.prior == "human" else None
    rng = np.random.default_rng(args.seed)
    planner = PickPlanner(args.workers)
    env = PlateEnv(cam_size=32, cameras=("agentview",))
    env.reset(0)
    D = env.bowl_diameter
    eps, tried, t0 = [], 0, time.time()
    while len(eps) < args.n:
        init = int(rng.integers(50))
        env.reset(init, jitter=args.jitter, rng=rng)
        start = env.get_state()
        best, _, _, _ = planner.plan(start, D, prior=prior, rng=rng, pop=30, iters=3)
        env.set_state(start)
        obs, act, ex = [env.plate_obs()], [], []

        def rec():
            act.append(env.last_action[ACT_IDX].copy())
            ex.append(env.last_executed[ACT_IDX].copy())
            obs.append(env.plate_obs())
        execute(env, best, on_step=rec)
        ok = env.success()
        for _ in range(HOLD_STEPS):
            env.step_to(env.eef_pos(), -1.0, yaw=env.eef_yaw())
            rec()
        tried += 1
        if ok:
            eps.append(dict(obs=np.array(obs[:-1]), act=np.array(act), act_exec=np.array(ex), init=init,
                            qpos=np.array(start.qpos), qvel=np.array(start.qvel),
                            theta=best.theta, lift=best.lift))
        if tried % 10 == 0:
            print(f"[plate_{args.prior}] {len(eps)}/{args.n} kept of {tried} tried "
                  f"({(time.time() - t0) / tried:.1f}s/ep)", flush=True)
    tag = f"plate_{args.prior}"
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"demos_{tag}.npz",
                        obs=np.concatenate([e["obs"] for e in eps]), act=np.concatenate([e["act"] for e in eps]),
                        act_exec=np.concatenate([e["act_exec"] for e in eps]),
                        ep_len=np.array([len(e["act"]) for e in eps]), init=np.array([e["init"] for e in eps]),
                        qpos=np.stack([e["qpos"] for e in eps]), qvel=np.stack([e["qvel"] for e in eps]))
    meta = dict(prior=args.prior, jitter=args.jitter, kept=len(eps), tried=tried,
                planner_success_rate=round(len(eps) / tried, 3),
                grip_deg=[round(float(np.degrees(e["theta"]) % 360), 1) for e in eps],
                lift_cm=[round(float(e["lift"]) * 100, 1) for e in eps], seconds=round(time.time() - t0))
    (args.out / f"demos_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("grip_deg", "lift_cm")}, indent=2))
    planner.close()
    env.close()


if __name__ == "__main__":
    main()
