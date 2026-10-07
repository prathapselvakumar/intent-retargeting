"""Scenario 2 evaluation on LIBERO's 50 official initial states of libero_goal task 8, with
LIBERO's own success check. As in LIBERO's evaluation, an episode succeeds as soon as the
bowl is on the plate.

The MLP policy reads PlateEnv.plate_obs (bowl- and plate-relative state); SmolVLA reads the
front and wrist cameras, the robot state and the instruction. Training demonstrations came
from jittered starts (src/sim/gen_plate_demos.py), so these 50 starts are held out.

Usage: python src/policy/eval_plate.py --policy runs/plate_mlp_human_s0
       python src/policy/eval_plate.py --vla runs/smolvla_plate/checkpoints/last/pretrained_model
"""
import argparse
import json
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "policy"), str(ROOT / "vision")]
from env import CTRL_HZ                                   # noqa: E402
from evaluate import Policy, wilson                       # noqa: E402
from plate_task import PlateEnv                           # noqa: E402

MAX_STEPS = 250
TASK = "put the bowl on the plate"


def apply(env, a5):
    a = np.zeros(7)
    a[[0, 1, 2, 5]] = np.clip(a5[:4], -1, 1)
    a[6] = 1.0 if a5[4] > 0 else -1.0
    env.step(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, help="MLP policy run dir")
    ap.add_argument("--vla", type=Path, help="SmolVLA pretrained_model dir")
    ap.add_argument("--exec-k", type=int, default=None, help="actions executed per query")
    ap.add_argument("--inits", type=int, default=50)
    ap.add_argument("--video", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    vla = args.vla is not None
    if vla:
        from common import state_vec
        from eval_vla import load
        k = args.exec_k or 5
        policy, pre, post = load(args.vla, k)
        env = PlateEnv(cam_size=256, cameras=("agentview", "robot0_eye_in_hand"))
        keys = ("observation.images.image", "observation.images.image2")
    else:
        k = args.exec_k or 4
        pol = Policy(args.policy)
        env = PlateEnv(cam_size=256 if args.video else 32, cameras=("agentview",))

    results, lat = [], []
    for init in range(args.inits):
        env.reset(init)
        if vla:
            policy.reset()
        frames = [] if init < args.video else None
        chunk, ok, steps = None, False, 0
        for t in range(MAX_STEPS):
            t0 = time.perf_counter()
            if vla:
                imgs = env.render()
                batch = {key: torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)[None].float() / 255.0
                         for key, im in zip(keys, imgs)}
                batch["observation.state"] = torch.from_numpy(state_vec(env))[None]
                batch["task"] = [TASK]
                with torch.inference_mode():
                    a5 = post(policy.select_action(pre(batch)))[0].cpu().numpy()
                if frames is not None:
                    frames.append(np.hstack(imgs))
            else:
                if t % k == 0:
                    chunk = pol(env.plate_obs())
                a5 = chunk[t % k]
                if frames is not None:
                    frames.append(env.render()[0])
            lat.append(time.perf_counter() - t0)
            apply(env, a5)
            steps = t + 1
            if env.success():
                ok = True
                break
        results.append(dict(init=init, success=ok, steps=steps))
        if frames:
            imageio.mimsave(args.out / f"init{init}.mp4", frames, fps=CTRL_HZ)
        print(f"init {init:2d}: {'SUCCESS' if ok else 'fail'} in {steps} steps", flush=True)
    n_ok = sum(r["success"] for r in results)
    report = dict(method="smolvla" if vla else "mlp", source=str(args.vla or args.policy), exec_k=k,
                  success=n_ok, n=len(results), rate=round(100 * n_ok / len(results), 1),
                  ci95=wilson(n_ok, len(results)), ms_per_step=round(1000 * float(np.mean(lat)), 2),
                  episodes=results)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({x: report[x] for x in ("method", "success", "n", "rate", "ci95", "ms_per_step")}))
    env.close()


if __name__ == "__main__":
    main()
