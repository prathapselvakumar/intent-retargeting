"""A learned world model of the bowl, trained on robot interaction and tested on my hand.

World model: f(contact state, end-effector motion) → next bowl motion. It is trained
only on simulated Panda interactions (the planner's 600 demos), in bowl-diameter units
and an embodiment-agnostic action space: the motion of the "pinch point" (the gripper's
fingertip centre, or the midpoint of my thumb and index finger) and how open it is.

Tests
  sim     held-out robot episodes: roll the model forward open-loop from the first
          frame, given only the end-effector trajectory → where does the bowl end up?
  human   my two iPhone clips: feed my real pinch-point trajectory and finger aperture
          (no depth: pinch height fixed at the sim's typical contact height) → does the
          model predict where my real bowl went?

Baselines: "bowl doesn't move", and "bowl sticks to the hand while it's closed and
touching the rim" (a hand-coded kinematic model).

Usage: python src/worldmodel/bowl_wm.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "sim"))
from env import CTRL_HZ, human_to_robot                 # noqa: E402
from replay_hand import load_human, resample            # noqa: E402

SIM_D = 0.112                 # LIBERO bowl diameter (m)
OPEN_MAX = 0.08               # Panda finger opening range (m)
GRIP_ON_WALL = 0.013          # median opening while the bowl moves in the demos
CONTACT_Z = 0.03              # typical pinch height above the table in the demos (m)
OUT = ROOT / "results" / "worldmodel"


# ---------------------------------------------------------------------------------------
def features(rel_xy, z, de, opening, yaw, prev_db):
    """rel_xy: eef − bowl (D), z: eef height (D), de: eef step (D), prev_db: bowl step (D)."""
    z, opening, yaw = (np.asarray(v, float) for v in (z, opening, yaw))
    return np.c_[rel_xy, z, de, opening / OPEN_MAX, np.sin(yaw), np.cos(yaw), prev_db].astype(np.float32)


def sim_episodes(path):
    d = np.load(path)
    off = np.r_[0, np.cumsum(d["ep_len"])]
    eps = []
    for i in range(len(d["ep_len"])):
        o = d["obs"][off[i]:off[i + 1]]
        b = d["goal"][i] - o[:, 7:9]                       # bowl xy (m)
        e = b + o[:, 0:2]                                  # eef xy (m)
        eps.append(dict(e=e / SIM_D, z=o[:, 3] / SIM_D, b=b / SIM_D, opening=o[:, 6],
                        yaw=np.arctan2(o[:, 4], o[:, 5])))
    return eps


def transitions(eps):
    X, Y = [], []
    for ep in eps:
        e, b = ep["e"], ep["b"]
        de, db = np.diff(e, axis=0), np.diff(b, axis=0)
        prev = np.r_[np.zeros((1, 2)), db[:-1]]
        dz = np.diff(ep["z"])[:, None]
        X.append(features(e[:-1] - b[:-1], ep["z"][:-1], np.c_[de, dz], ep["opening"][:-1],
                          ep["yaw"][:-1], prev))
        Y.append(db)
    return np.concatenate(X), np.concatenate(Y).astype(np.float32)


class WM(nn.Module):
    def __init__(self, d_in, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def train(X, Y, steps=15000, seed=0, dev="cuda"):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ysd = Y.std(0) + 1e-6
    Xt = torch.tensor((X - mu) / sd, device=dev)
    Yt = torch.tensor(Y / ysd, device=dev)
    # Moving-bowl transitions are rare (~20%) but are what the model is for: upweight them.
    w = torch.where(Yt.norm(dim=1) > 1e-3 / float(ysd.mean()), 4.0, 1.0)
    m = WM(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.randint(0, len(Xt), (1024,), device=dev)
        loss = (w[i] * ((m(Xt[i]) - Yt[i]) ** 2).sum(1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    m.eval()

    @torch.no_grad()
    def step(x):
        return m(torch.tensor((x - mu) / sd, device=dev)).cpu().numpy() * ysd
    return step


def rollout(step, e, z, opening, yaw, b0=(0.0, 0.0)):
    """Open-loop: only the pinch-point trajectory is given; the bowl is predicted."""
    b = [np.array(b0, float)]
    prev = np.zeros(2)
    for t in range(len(e) - 1):
        de = np.r_[e[t + 1] - e[t], z[t + 1] - z[t]]
        x = features((e[t] - b[-1])[None], [z[t]], de[None], [opening[t]], [yaw[t]], prev[None])
        db = step(x)[0]
        b.append(b[-1] + db)
        prev = db
    return np.array(b)


def kinematic(e, z, opening, b0=(0.0, 0.0), reach=0.6, closed=0.03):
    """Hand-coded baseline: the bowl moves with the pinch point while it is closed and
    within `reach` D of the bowl centre (i.e. holding the rim)."""
    b = [np.array(b0, float)]
    for t in range(len(e) - 1):
        holding = opening[t] < closed and np.linalg.norm(e[t] - b[-1]) < reach and z[t] < 0.5
        b.append(b[-1] + (e[t + 1] - e[t] if holding else 0))
    return np.array(b)


def human_episode(name):
    """My clip → the world model's input, in the robot frame and bowl-diameter units."""
    H = load_human(name)
    n = int(H["T"] / H["fps"] * CTRL_HZ)
    pinch = human_to_robot(resample(H["pinch_rel"], H["fps"], n))
    bowl = human_to_robot(resample(H["bowl_rel"], H["fps"], n))
    ap = resample(H["aperture"], H["fps"], n)
    # Aperture (cm) → Panda opening: ≤4.0 cm gripping the wall, ≥5.5 cm fully open.
    opening = GRIP_ON_WALL + np.clip((ap - 4.0) / 1.5, 0, 1) * (OPEN_MAX - GRIP_ON_WALL)
    rel = pinch - bowl
    yaw = np.arctan2(rel[:, 1], rel[:, 0]) - np.pi / 2       # fingers across the rim, as in sim pinches
    z = np.full(n, CONTACT_Z / SIM_D)
    return dict(e=pinch, z=z, opening=opening, yaw=yaw, b=bowl)


def path_err(pred, true):
    k = min(len(pred), len(true))
    return float(np.linalg.norm(pred[:k] - true[:k], axis=1).mean()), float(np.linalg.norm(pred[k - 1] - true[k - 1]))


def explore_episodes(path):
    d = np.load(path)
    off = np.r_[0, np.cumsum(d["ep_len"])]
    return [dict(e=d["e"][a:b] / SIM_D, z=d["z"][a:b] / SIM_D, b=d["b"][a:b] / SIM_D,
                 opening=d["opening"][a:b], yaw=d["yaw"][a:b]) for a, b in zip(off[:-1], off[1:])]


def split(eps, frac=0.1, seed=0):
    idx = np.random.default_rng(seed).permutation(len(eps))
    k = int(len(eps) * frac)
    return [eps[i] for i in idx[k:]], [eps[i] for i in idx[:k]]


def evaluate(steps, test, humans):
    """Open-loop rollouts per ensemble member + ensemble disagreement on contact steps."""
    r = dict(sim={}, human={}, disagreement={})
    fin = {"wm": [], "static": [], "kinematic": []}
    for ep in test:
        true, e = ep["b"] - ep["b"][0], ep["e"] - ep["b"][0]
        fin["wm"] += [path_err(rollout(st, e, ep["z"], ep["opening"], ep["yaw"]), true)[1] for st in steps]
        fin["static"].append(float(np.linalg.norm(true[-1])))
        fin["kinematic"].append(path_err(kinematic(e, ep["z"], ep["opening"]), true)[1])
    r["sim"] = {k: dict(median_final_err_D=round(float(np.median(v)), 3),
                        within_0p1D=round(float(np.mean(np.array(v) < 0.1)), 3)) for k, v in fin.items()}
    traces = {}
    for h, H in humans.items():
        preds = [rollout(st, H["e"], H["z"], H["opening"], H["yaw"]) for st in steps]
        errs = [path_err(p, H["b"])[1] for p in preds]
        mean_pred = np.mean(preds, axis=0)
        r["human"][h] = dict(true_final_D=np.round(H["b"][-1], 3).tolist(),
                             ensemble_final_D=np.round(mean_pred[-1], 3).tolist(),
                             ensemble_final_err_D=round(path_err(mean_pred, H["b"])[1], 3),
                             member_final_err_D=[round(e, 3) for e in errs],
                             kin_final_err_D=round(path_err(kinematic(H["e"], H["z"], H["opening"]), H["b"])[1], 3),
                             static_final_err_D=round(float(np.linalg.norm(H["b"][-1])), 3))
        traces[h] = dict(true=H["b"], hand=H["e"], wm=np.array(preds))

    def spread(eps):
        X, _ = transitions(eps)
        near = np.linalg.norm(X[:, :2], axis=1) < 0.8          # pinch point within 0.8 D of bowl centre
        P = np.stack([st(X[near]) for st in steps])             # [members, N, 2]
        return float(np.linalg.norm(P.std(0), axis=1).mean())
    r["disagreement"] = dict(sim_contact_steps=round(spread(test), 5),
                             human_contact_steps=round(spread(list(humans.values())), 5))
    r["disagreement"]["ratio_human_over_sim"] = round(r["disagreement"]["human_contact_steps"] /
                                                      r["disagreement"]["sim_contact_steps"], 1)
    return r, traces


def main():
    demos = sim_episodes(ROOT / "data/demos/demos_strategy.npz") + sim_episodes(ROOT / "data/demos/demos_none.npz")
    d_tr, d_te = split(demos)
    sets = {"demos_only": (d_tr, d_te)}
    exp_path = ROOT / "data/demos/explore.npz"
    if exp_path.exists():
        e_tr, e_te = split(explore_episodes(exp_path))
        sets["demos_plus_exploration"] = (d_tr + e_tr, d_te + e_te)
    humans = {h: human_episode(h) for h in ["episode_000_right", "episode_050_left"]}
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, (tr, te) in sets.items():
        X, Y = transitions(tr)
        steps = [train(X, Y, seed=s) for s in range(3)]          # 3-member ensemble
        r, traces = evaluate(steps, te, humans)
        r.update(train_transitions=int(len(X)), test_episodes=len(te))
        report[name] = r
        print(name, json.dumps(r, indent=1), flush=True)
        np.savez(OUT / f"human_traces_{name}.npz",
                 **{f"{h}_{k}": v for h, t in traces.items() for k, v in t.items()})
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
