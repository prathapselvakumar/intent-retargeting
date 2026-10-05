"""Reproduce each human clip's bowl displacement with the intent-level planner, with and
without the human contact prior, over several seeds.

Usage: python src/sim/run_planner.py episode_000_right episode_050_left --seeds 5
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import imageio
import numpy as np

from env import CTRL_HZ, BowlEnv, human_to_robot
from planner import SUCCESS_D, SUCCESS_TILT, Planner, execute, human_prior
from replay_hand import human_frames, load_human

MAX_PRIMS = 3


def run_episode(env, planner, goal_D, prior, seed, render_human=None):
    rng = np.random.default_rng(seed)
    env.reset()
    D = env.bowl_diameter
    start = env.bowl_pos()[:2]
    goal = start + human_to_robot(goal_D) * D
    frames, steps, log = [], [], []
    hum_i = [0]

    def snap():
        if render_human is not None:
            h = render_human[min(hum_i[0], len(render_human) - 1)]
            hum_i[0] += 1
            frames.append(np.hstack([h, *env.render()]))

    total_rollouts, first_success = 0, None
    for _ in range(MAX_PRIMS):
        prim, (c, err_pred, tilt_pred), fs, n = planner.plan(
            env.get_state(), goal, env.bowl_pos()[:2], D, prior=prior, rng=rng,
            log=lambda s: log.append(s))
        if first_success is None and fs is not None:
            first_success = total_rollouts + fs
        total_rollouts += n
        execute(env, prim, on_step=snap)
        err = float(np.linalg.norm(env.bowl_pos()[:2] - goal) / D)
        tilt = env.bowl_tilt_deg()
        steps.append(dict(prim=asdict(prim), err_D=round(err, 3), tilt_deg=round(tilt, 1)))
        if err < SUCCESS_D and tilt < SUCCESS_TILT:
            break
    final_err = steps[-1]["err_D"]
    return dict(success=bool(final_err < SUCCESS_D and steps[-1]["tilt_deg"] < SUCCESS_TILT),
                final_err_D=final_err, n_prims=len(steps), rollouts=total_rollouts,
                rollouts_to_first_success=first_success, steps=steps), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=Path, default=Path("results/planner"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    planner = Planner(args.workers)
    env = BowlEnv()
    summary = []
    for name in args.names:
        H = load_human(name)
        goal_D = H["bowl_rel"][-1]
        prior = human_prior(Path("data/processed/intent") / f"{name}.json")
        hum = human_frames(name, 2000, H["fps"], 256)
        for cond in ("human_prior", "no_prior"):
            for seed in range(args.seeds):
                t = time.time()
                res, frames = run_episode(env, planner, goal_D,
                                          prior if cond == "human_prior" else None, seed,
                                          render_human=hum if seed == 0 else None)
                res.update(video=name, condition=cond, seed=seed, seconds=round(time.time() - t, 1))
                summary.append(res)
                last = res["steps"][-1]["prim"]
                print(f"{name} {cond:11s} seed {seed}: success={res['success']} "
                      f"err={res['final_err_D']:.3f}D prims={res['n_prims']} "
                      f"first_success@{res['rollouts_to_first_success']} "
                      f"mode={last['mode']} theta={np.degrees(last['theta']):.0f}° ({res['seconds']}s)",
                      flush=True)
                if frames:
                    imageio.mimsave(args.out / f"{name}_{cond}.mp4", frames, fps=CTRL_HZ)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    env.close()
    planner.close()


if __name__ == "__main__":
    main()
