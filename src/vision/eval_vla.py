"""Closed-loop evaluation of a fine-tuned SmolVLA checkpoint on the same held-out task sets
as the state policy (src/policy/evaluate.py), from pixels + proprioception only.

Usage: python src/vision/eval_vla.py --ckpt runs/smolvla_strategy/checkpoints/last/pretrained_model --n 50
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
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "policy")]
from common import CAMS, IMG, TASK, observe, state_vec                 # noqa: E402
from env import CTRL_HZ, BowlEnv                                       # noqa: E402
from evaluate import MAX_STEPS, tasks, wilson                          # noqa: E402
from planner import SUCCESS_D, SUCCESS_TILT                            # noqa: E402

RENAME = {"observation.images.image": "observation.images.camera1",
          "observation.images.image2": "observation.images.camera2"}


def load(ckpt, n_action_steps):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    policy = SmolVLAPolicy.from_pretrained(ckpt).to("cuda").eval()
    policy.config.n_action_steps = n_action_steps
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": "cuda"},
                                "rename_observations_processor": {"rename_map": RENAME}})
    return policy, pre, post


def to_batch(env, goal):
    obs = observe(env, goal)
    b = {k: torch.from_numpy(v).permute(2, 0, 1)[None].float() / 255.0 for k, v in obs.items()}
    b["observation.state"] = torch.from_numpy(state_vec(env))[None]
    b["task"] = [TASK]
    return b, obs


def run(env, policy, pre, post, start, goal, frames=None):
    env.reset(bowl_xy=start)
    goal = goal + (env.bowl_pos()[:2] - start)
    policy.reset()
    lat = []
    for _ in range(MAX_STEPS):
        batch, obs = to_batch(env, goal)
        t0 = time.perf_counter()
        with torch.inference_mode():
            a5 = post(policy.select_action(pre(batch)))[0].cpu().numpy()
        lat.append(time.perf_counter() - t0)
        a = np.zeros(7)
        a[[0, 1, 2, 5]] = np.clip(a5[:4], -1, 1)
        a[6] = 1.0 if a5[4] > 0 else -1.0
        env.step(a)
        if frames is not None:
            frames.append(np.hstack(list(obs.values())))
    return goal, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--sets", nargs="+", default=["random", "human_right", "human_left"])
    ap.add_argument("--video", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    policy, pre, post = load(args.ckpt, args.n_action_steps)
    env = BowlEnv(cam_size=IMG, cameras=CAMS)
    env.reset()
    D = env.bowl_diameter
    report = dict(method="smolvla", ckpt=str(args.ckpt), n_action_steps=args.n_action_steps)
    for kind in args.sets:
        res, lat = [], []
        for i, (start, goal) in enumerate(tasks(kind, args.n, D)):
            frames = [] if i < args.video else None
            g, l = run(env, policy, pre, post, start, goal, frames)
            lat += l
            err = float(np.linalg.norm(env.bowl_pos()[:2] - g) / D)
            tilt = env.bowl_tilt_deg()
            res.append(dict(err_D=err, success=bool(err < SUCCESS_D and tilt < SUCCESS_TILT)))
            if frames:
                imageio.mimsave(args.out / f"{kind}_{i}.mp4", frames, fps=CTRL_HZ)
        k = sum(r["success"] for r in res)
        lat = np.array(lat)
        report[kind] = dict(success=k, n=len(res), rate=round(100 * k / len(res), 1),
                            ci95=wilson(k, len(res)),
                            median_err_D=round(float(np.median([r["err_D"] for r in res])), 3),
                            ms_per_step_mean=round(1000 * float(lat.mean()), 1),
                            ms_per_chunk_query=round(1000 * float(np.percentile(lat, 95)), 1))
        print(kind, report[kind], flush=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
