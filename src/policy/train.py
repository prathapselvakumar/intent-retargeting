"""Distil planner demonstrations into a fast closed-loop policy.

Model: MLP that maps the bowl-centric state (BowlEnv.policy_obs) to a chunk of the next
H actions (action chunking, as in ACT / Diffusion Policy). At test time the first K
actions of each chunk are executed before re-querying, which smooths the control and
makes the phase structure (approach → descend → grip → drag → release) easier to learn
from a deterministic regressor.

Usage: python src/policy/train.py --demos data/demos/demos_strategy.npz --out runs/policy_strategy
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

H = 8


class ChunkMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=512, horizon=H):
        super().__init__()
        self.horizon, self.act_dim = horizon, act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, horizon * act_dim))

    def forward(self, x):
        return self.net(x).view(-1, self.horizon, self.act_dim)


def chunk_targets(act, ep_len, horizon=H):
    """For each step, the next `horizon` actions of the same episode, padded with the
    episode's last action (which is a 'hold' action)."""
    out, s = [], 0
    for L in ep_len:
        a = act[s:s + L]
        pad = np.concatenate([a, np.repeat(a[-1:], horizon, 0)])
        out.append(np.stack([pad[t:t + horizon] for t in range(L)]))
        s += L
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-episodes", type=int, default=None, help="use only the first N episodes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = np.load(args.demos)
    obs, act, ep_len = d["obs"], d["act"], d["ep_len"]
    if args.n_episodes:
        ep_len = ep_len[:args.n_episodes]
        n = int(ep_len.sum())
        obs, act = obs[:n], act[:n]
    tgt = chunk_targets(act, ep_len)

    o_mu, o_sd = obs.mean(0), obs.std(0) + 1e-6
    a_mu, a_sd = act.mean(0), act.std(0) + 1e-6
    X = torch.tensor((obs - o_mu) / o_sd, dtype=torch.float32, device=dev)
    Y = torch.tensor((tgt - a_mu) / a_sd, dtype=torch.float32, device=dev)

    model = ChunkMLP(obs.shape[1], act.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    for step in range(args.steps):
        idx = torch.randint(0, len(X), (args.batch,), device=dev)
        loss = nn.functional.mse_loss(model(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 5000 == 0 or step == args.steps - 1:
            print(f"step {step:6d}  loss {loss.item():.4f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=model.state_dict(), obs_dim=obs.shape[1], act_dim=act.shape[1],
                    o_mu=o_mu, o_sd=o_sd, a_mu=a_mu, a_sd=a_sd), args.out / "policy.pt")
    (args.out / "train.json").write_text(json.dumps(dict(
        demos=str(args.demos), episodes=int(len(ep_len)), samples=int(len(X)),
        steps=args.steps, final_loss=float(loss.item())), indent=2))


if __name__ == "__main__":
    main()
