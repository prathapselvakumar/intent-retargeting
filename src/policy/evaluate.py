"""Closed-loop evaluation of the distilled policy (and the hand-replay baseline) on held-out
tasks.

Task sets (fixed seeds, disjoint from the demo-generation seed):
  random      — random bowl start, random goal (same distribution as training)
  human_right — random bowl start, goal = the right-hand clip's bowl displacement
  human_left  — random bowl start, goal = the left-hand clip's bowl displacement

Success: bowl within 0.1 bowl-diameters of the goal and tilted < 15° at the end.
Reports success rate with a 95% Wilson interval.

Usage: python src/policy/evaluate.py --policy runs/policy_strategy --n 50
       python src/policy/evaluate.py --hand-replay --n 25
"""
import argparse
import json
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from env import CTRL_HZ, BowlEnv, human_to_robot                       # noqa: E402
from gen_demos import sample_task                                       # noqa: E402
from planner import SUCCESS_D, SUCCESS_TILT                             # noqa: E402
from replay_hand import grip_signal, load_human, resample               # noqa: E402
from train import ChunkMLP                                              # noqa: E402

MAX_STEPS = 160
EXEC_K = 4           # actions executed per chunk before re-querying
EVAL_SEED = 1000
SET_OFFSET = dict(random=0, human_right=1, human_left=2)   # fixed, not hash(): reproducible
CLIPS = dict(human_right="episode_000_right", human_left="episode_050_left")
# Best fixed gripper height for the hand-replay baseline per clip (from the height sweep).
REPLAY_Z = dict(episode_000_right=0.045, episode_050_left=0.015)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(float(100 * (c - h)), 1), round(float(100 * (c + h)), 1))


def tasks(kind, n, D):
    rng = np.random.default_rng(EVAL_SEED + SET_OFFSET[kind])
    out = []
    for _ in range(n):
        start, goal = sample_task(rng, D)
        if kind != "random":
            g = load_human(CLIPS[kind])["bowl_rel"][-1]
            goal = start + human_to_robot(g) * D
        out.append((start, goal))
    return out


class Policy:
    def __init__(self, path, dev="cuda"):
        ck = torch.load(Path(path) / "policy.pt", weights_only=False)
        self.m = ChunkMLP(ck["obs_dim"], ck["act_dim"]).to(dev).eval()
        self.m.load_state_dict(ck["model"])
        self.ck, self.dev = ck, dev

    @torch.no_grad()
    def __call__(self, obs):
        x = torch.tensor((obs - self.ck["o_mu"]) / self.ck["o_sd"], dtype=torch.float32,
                         device=self.dev)[None]
        y = self.m(x)[0].cpu().numpy()
        return y * self.ck["a_sd"] + self.ck["a_mu"]


def run_policy(env, pol, start, goal, frames=None):
    env.reset(bowl_xy=start)
    goal = goal + (env.bowl_pos()[:2] - start)
    t_inf, chunk = [], []
    for t in range(MAX_STEPS):
        if t % EXEC_K == 0:
            t0 = time.perf_counter()
            chunk = pol(env.policy_obs(goal))
            t_inf.append(time.perf_counter() - t0)
        a5 = chunk[t % EXEC_K]
        a = np.zeros(7)
        a[[0, 1, 2, 5]] = np.clip(a5[:4], -1, 1)
        a[6] = 1.0 if a5[4] > 0 else -1.0
        env.step(a)
        if frames is not None:
            frames.append(np.hstack(env.render()))
    return goal, float(np.mean(t_inf))


def run_hand_replay(env, clip, start, goal):
    from replay_hand import APPROACH_S, APPROACH_Z
    H = load_human(clip)
    n = int(H["T"] / H["fps"] * CTRL_HZ)
    pinch = resample(H["pinch_rel"], H["fps"], n)
    grip = grip_signal(resample(H["aperture"], H["fps"], n))
    env.reset(bowl_xy=start)
    goal = goal + (env.bowl_pos()[:2] - start)
    D, b0 = env.bowl_diameter, env.bowl_pos()
    ee = b0[:2] + human_to_robot(pinch) * D
    z_hold, z_high = env.table_z + REPLAY_Z[clip], env.table_z + APPROACH_Z
    for k in range(int(APPROACH_S * CTRL_HZ)):
        env.step_to([*ee[0], z_high if k < APPROACH_S * CTRL_HZ / 2 else z_hold], -1)
    for i in range(n):
        env.step_to([*ee[i], z_hold], grip[i])
    for _ in range(20):
        env.step_to([*ee[-1], z_high], -1)
    return goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path)
    ap.add_argument("--hand-replay", action="store_true")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--sets", nargs="+", default=["random", "human_right", "human_left"])
    ap.add_argument("--video", type=int, default=3, help="episodes per set to record")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    name = "hand_replay" if args.hand_replay else args.policy.name
    out = args.out or Path("results/eval") / name
    out.mkdir(parents=True, exist_ok=True)

    env = BowlEnv(cameras=("birdview", "agentview"))
    env.reset()
    D = env.bowl_diameter
    pol = None if args.hand_replay else Policy(args.policy)
    report = dict(method=name)
    for kind in args.sets:
        if args.hand_replay and kind == "random":
            continue                       # replay only knows the two recorded motions
        res, inf = [], []
        for i, (start, goal) in enumerate(tasks(kind, args.n, D)):
            frames = [] if i < args.video else None
            if args.hand_replay:
                g = run_hand_replay(env, CLIPS[kind], start, goal)
            else:
                g, ti = run_policy(env, pol, start, goal, frames)
                inf.append(ti)
            err = float(np.linalg.norm(env.bowl_pos()[:2] - g) / D)
            tilt = env.bowl_tilt_deg()
            res.append(dict(err_D=round(err, 3), tilt=round(tilt, 1),
                            success=bool(err < SUCCESS_D and tilt < SUCCESS_TILT)))
            if frames:
                imageio.mimsave(out / f"{kind}_{i}.mp4", frames, fps=CTRL_HZ)
        k = sum(r["success"] for r in res)
        report[kind] = dict(success=k, n=len(res), rate=round(100 * k / len(res), 1),
                            ci95=wilson(k, len(res)),
                            median_err_D=round(float(np.median([r["err_D"] for r in res])), 3))
        if inf:
            report[kind]["policy_ms_per_query"] = round(1000 * float(np.mean(inf)), 2)
        print(kind, report[kind], flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
