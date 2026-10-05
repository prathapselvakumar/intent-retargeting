"""Intent-level planner: make the SIM bowl do what the HUMAN bowl did, letting the robot
choose its own contact strategy.

The search space is a short manipulation primitive, not a joint trajectory:

  mode   push   — closed gripper outside the rim, moves through the bowl
         inside — closed gripper lowered inside the bowl, drags it by the inner wall
         pinch  — open gripper straddles the wall, closes on it, drags, releases
  theta  contact angle around the bowl (robot frame)
  h      contact height above the table
  phi    motion direction
  d      motion distance

Cross-entropy method (CEM) over these parameters; every candidate is a full physics
rollout from the current state, evaluated by how close the bowl lands to the goal.
Closed loop: after executing the best primitive, re-plan from the resulting state until
the bowl is within tolerance or the primitive budget is spent.

The human video enters twice:
  * the GOAL — bowl displacement measured from the video (always used)
  * an optional CONTACT PRIOR — where and how the human touched the rim (rim angle and
    push/drag/pull mode from src/extract/contact.py). `--prior none` drops it, which is
    the ablation that shows what the human contact information is worth.
"""
import json
import multiprocessing as mp
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

MODES = ("push", "inside", "pinch")
SUCCESS_D = 0.10            # bowl within 0.1 bowl-diameters of the goal …
SUCCESS_TILT = 15.0         # … and upright
Z_TRAVEL = 0.12             # travel height above the table
MOVE_STEP_M = 0.01          # 1 cm per 20 Hz step while in contact (20 cm/s)
FINGER_CLEAR = 0.015        # gripper centre offset outside the rim for pushing
INSIDE_DEPTH = 0.025        # … and inside the rim for inside-drags
H_RANGE = (0.008, 0.045)


@dataclass
class Prim:
    mode: str
    theta: float
    h: float
    phi: float
    d: float


# ---------------------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------------------
def wrap_half(a):
    """Wrap to (-pi/2, pi/2]: the parallel gripper is symmetric under 180° yaw."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def execute(env, p: Prim, on_step=None):
    """Run one primitive. Returns per-step bowl tilt (for the cost)."""
    c = env.bowl_pos()[:2]
    R = env.bowl_diameter / 2
    radial = np.array([np.cos(p.theta), np.sin(p.theta)])
    r = {"push": R + FINGER_CLEAR, "inside": R - INSIDE_DEPTH, "pinch": R - 0.003}[p.mode]
    contact = c + r * radial
    yaw = wrap_half(p.theta - np.pi / 2) if p.mode == "pinch" else 0.0
    pre_grip = -1.0 if p.mode == "pinch" else 1.0
    z_top, z_low = env.table_z + Z_TRAVEL, env.table_z + p.h
    move = np.array([np.cos(p.phi), np.sin(p.phi)]) * p.d
    tilts = []

    def go(target, grip, n, interp_from=None):
        for k in range(n):
            tgt = target if interp_from is None else interp_from + (target - interp_from) * (k + 1) / n
            env.step_to(tgt, grip, yaw=yaw)
            tilts.append(env.bowl_tilt_deg())
            if on_step is not None:
                on_step()

    go(np.r_[contact, z_top], pre_grip, 14)                               # travel above
    go(np.r_[contact, z_low], pre_grip, 10, np.r_[contact, z_top])          # descend
    if p.mode == "pinch":
        go(np.r_[contact, z_low], 1.0, 6)                                   # close on wall
    grip = 1.0
    n_move = max(3, int(np.ceil(p.d / MOVE_STEP_M)))
    go(np.r_[contact + move, z_low], grip, n_move, np.r_[contact, z_low])  # drag / push
    if p.mode == "pinch":
        go(np.r_[contact + move, z_low], -1.0, 5)                           # release
    go(np.r_[contact + move, z_top], -1.0 if p.mode == "pinch" else 1.0, 8,
       np.r_[contact + move, z_low])                                        # retreat
    go(np.r_[contact + move, z_top], -1.0 if p.mode == "pinch" else 1.0, 4)  # let it settle
    return np.array(tilts)


def cost(env, start_xy, goal_xy, tilts):
    D = env.bowl_diameter
    err = np.linalg.norm(env.bowl_pos()[:2] - goal_xy) / D
    final_tilt = env.bowl_tilt_deg()
    c = err + max(0.0, final_tilt - 10) / 45 + 0.3 * float(tilts.max() > 35)
    return c, err, final_tilt


# ---------------------------------------------------------------------------------------
# Parallel rollouts: each worker owns one headless sim
# ---------------------------------------------------------------------------------------
_W = None


def _init_worker():
    global _W
    os.environ.setdefault("MUJOCO_GL", "egl")
    import logging, warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    from env import BowlEnv
    _W = BowlEnv(cam_size=64, cameras=("birdview",))
    _W.reset()


def _rollout(args):
    state, goal_xy, prim = args
    _W.set_state(state)
    start = _W.bowl_pos()[:2]
    tilts = execute(_W, Prim(**prim))
    c, err, tilt = cost(_W, start, goal_xy, tilts)
    return c, err, tilt


class Planner:
    def __init__(self, workers=16):
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(workers, initializer=_init_worker)

    def close(self):
        self.pool.close()
        self.pool.join()

    # ---- CEM ---------------------------------------------------------------------------
    def plan(self, state, goal_xy, start_xy, D, prior=None, rng=None, pop=48, elites=8,
             iters=4, log=None):
        """Returns (best Prim, best stats, rollouts until the first successful candidate)."""
        rng = rng or np.random.default_rng()
        to_goal = goal_xy - start_xy
        phi0, d0 = np.arctan2(to_goal[1], to_goal[0]), np.linalg.norm(to_goal)

        # Search distribution. Without a prior the contact angle and mode are uniform;
        # direction and distance always start from the goal (any planner would).
        mode_p = np.ones(3) / 3
        th_mu, th_sd = 0.0, None                           # None → uniform angle
        if prior is not None:
            mode_p = np.array([prior["mode_p"][m] for m in MODES])
            if "theta_rel" in prior:          # strategy prior: angle relative to motion
                th_mu, th_sd = phi0 + prior["theta_rel"], prior["theta_rel_sd"]
            else:                             # clip prior: absolute rim angle
                th_mu, th_sd = prior["theta"], np.radians(30)
        h_mu, h_sd = np.mean(H_RANGE), 0.012
        phi_mu, phi_sd = phi0, np.radians(25)
        d_mu, d_sd = d0, 0.4 * d0 + 0.01

        best, n_done, first_success = None, 0, None
        for it in range(iters):
            modes = rng.choice(3, size=pop, p=mode_p)
            th = rng.uniform(-np.pi, np.pi, pop) if th_sd is None else rng.normal(th_mu, th_sd, pop)
            prims = [Prim(MODES[m], float(t), float(np.clip(rng.normal(h_mu, h_sd), *H_RANGE)),
                          float(rng.normal(phi_mu, phi_sd)),
                          float(np.clip(rng.normal(d_mu, d_sd), 0.0, 0.25)))
                     for m, t in zip(modes, th)]
            res = self.pool.map(_rollout, [(state, goal_xy, asdict(p)) for p in prims])
            costs = np.array([r[0] for r in res])
            for k, (_, err, tilt) in enumerate(res):
                if first_success is None and err < SUCCESS_D and tilt < SUCCESS_TILT:
                    first_success = n_done + k + 1
            n_done += pop
            order = np.argsort(costs)
            if best is None or costs[order[0]] < best[1][0]:
                best = (prims[order[0]], res[order[0]])
            if log:
                log(f"    iter {it}: best cost {costs[order[0]]:.3f} err {res[order[0]][1]:.3f}D "
                    f"mode {prims[order[0]].mode}")
            # Refit on elites.
            el = [prims[i] for i in order[:elites]]
            counts = np.array([sum(e.mode == m for e in el) for m in MODES], float)
            mode_p = 0.7 * (counts / counts.sum()) + 0.3 * mode_p
            mode_p = np.maximum(mode_p, 0.02); mode_p /= mode_p.sum()
            ang = np.array([e.theta for e in el])
            th_mu = float(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()))
            th_sd = max(np.radians(8), float(np.sqrt(-2 * np.log(np.hypot(np.sin(ang).mean(), np.cos(ang).mean()) + 1e-9))))
            hs = np.array([e.h for e in el]); h_mu, h_sd = hs.mean(), max(0.003, hs.std())
            ps = np.array([e.phi for e in el])
            phi_mu = float(np.arctan2(np.sin(ps).mean(), np.cos(ps).mean())); phi_sd = max(np.radians(5), ps.std())
            ds = np.array([e.d for e in el]); d_mu, d_sd = ds.mean(), max(0.004, ds.std())
        return best[0], best[1], first_success, n_done


def human_prior(intent_json: Path):
    """Contact prior from the human video: rim angle (rotated into the robot frame) and a
    mode distribution from the push/drag/pull label."""
    intent = json.loads(Path(intent_json).read_text())
    seg = max(intent["segments"], key=lambda s: s["moved_cm"])
    theta = np.radians(seg["rim_angle_deg"] - 90.0)       # human frame → robot frame
    mode_p = {"push": dict(push=0.6, inside=0.2, pinch=0.2),
              "drag": dict(push=0.15, inside=0.25, pinch=0.6),
              "pull": dict(push=0.1, inside=0.3, pinch=0.6)}[seg["mode"]]
    return dict(theta=float(np.arctan2(np.sin(theta), np.cos(theta))), mode_p=mode_p,
                source=dict(rim_angle_deg=seg["rim_angle_deg"], mode=seg["mode"]))


def human_strategy_prior(intent_jsons):
    """Contact prior that generalises to any goal: the rim angle RELATIVE to the bowl's
    motion direction, pooled over all human clips. (Both recorded clips — right hand
    pushing away, left hand pulling back — grip the rim ~100° clockwise of the motion.)"""
    rels, modes = [], []
    for f in intent_jsons:
        seg = max(json.loads(Path(f).read_text())["segments"], key=lambda s: s["moved_cm"])
        theta = np.radians(seg["rim_angle_deg"] - 90.0)
        dx, dy = seg["bowl_delta_cm"]
        phi = np.arctan2(-dx, dy)                         # human → robot frame
        rels.append(np.arctan2(np.sin(theta - phi), np.cos(theta - phi)))
        modes.append(seg["mode"])
    rels = np.array(rels)
    mu = float(np.arctan2(np.sin(rels).mean(), np.cos(rels).mean()))
    sd = max(np.radians(15), float(np.std(rels)))
    drag_like = sum(m in ("drag", "pull") for m in modes) / len(modes)
    mode_p = dict(push=0.15 + 0.45 * (1 - drag_like), inside=0.25,
                  pinch=0.15 + 0.45 * drag_like)
    z = sum(mode_p.values())
    return dict(theta_rel=mu, theta_rel_sd=sd, mode_p={k: v / z for k, v in mode_p.items()},
                source=dict(rel_deg=np.degrees(rels).round(1).tolist(), modes=modes))
