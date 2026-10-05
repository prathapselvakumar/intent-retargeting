"""Scenario 2 in the real LIBERO task: libero_goal #8 "put the bowl on the plate".

Same idea as scenario 1 — copy the effect, not the motion — but the effect is now a
pick-and-place, scored by LIBERO's own success check (bowl On plate). The scene, physics
and the 50 official initial states are LIBERO's, untouched.

Pieces
  PlateEnv          LIBERO task env with the BowlEnv control/state helpers
  Pick              pick-and-place primitive: pinch the rim at angle `theta`, lift `lift`,
                    carry, set down at `offset` from the plate centre, release, retreat
  human_prior()     grip angle + lift height measured from the plate video
  PickPlanner       CEM over Pick parameters, parallel physics rollouts
"""
import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from env import BowlEnv, SETTLE_STEPS

TASK_SUITE, TASK_ID = "libero_goal", 8
PLATE = "plate_1"
Z_TRAVEL = 0.15
MOVE_STEP_M = 0.01
H_RANGE = (0.015, 0.045)        # pinch height above the table (bowl is 5.3 cm tall)
LIFT_RANGE = (0.02, 0.12)


def task_files():
    from libero.libero import benchmark, get_libero_path
    bm = benchmark.get_benchmark_dict()[TASK_SUITE]()
    t = bm.get_task(TASK_ID)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    return bddl, bm.get_task_init_states(TASK_ID), t.language


class PlateEnv(BowlEnv):
    def __init__(self, cam_size=128, cameras=("agentview", "robot0_eye_in_hand")):
        from libero.libero.envs import OffScreenRenderEnv
        bddl, self.init_states, self.language = task_files()
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=list(cameras),
                                      camera_heights=cam_size, camera_widths=cam_size)
        self.cameras = cameras

    def reset(self, init_idx=0, seed=0):
        """Official LIBERO initial state `init_idx` (0–49); LIBERO's default physics."""
        self.env.seed(seed)
        self.env.reset()
        self.env.set_init_state(self.init_states[init_idx % len(self.init_states)])
        self._bind()
        m = self.sim.model
        self.plate_bodies = [i for i in range(m.nbody) if PLATE in m.body_id2name(i)]
        for _ in range(SETTLE_STEPS):
            self.step(np.zeros(7))
        lo, hi = self._bowl_extent()
        self.table_z = float(lo[2])
        self.bowl_diameter = float(np.mean((hi - lo)[:2]))
        self.bowl_height = float(hi[2] - lo[2])
        self.bowl_rest_z = self.bowl_pos()[2]

    def plate_pos(self):
        return self.sim.data.body_xpos[self.plate_bodies[0]].copy()

    def success(self):
        return bool(self.inner._check_success())

    def plate_obs(self):
        """Low-dim state for a distilled policy (bowl- and plate-centric)."""
        b, p, e = self.bowl_pos(), self.plate_pos(), self.eef_pos()
        yaw = self.eef_yaw()
        return np.r_[e - b, p[:2] - b[:2], e[2] - self.table_z, b[2] - self.bowl_rest_z,
                     np.sin(yaw), np.cos(yaw), self.gripper_opening(),
                     self.bowl_tilt_deg() / 45.0].astype(np.float32)


# ---------------------------------------------------------------------------------------
@dataclass
class Pick:
    theta: float        # rim angle of the pinch (robot frame)
    h: float            # pinch height above the table
    lift: float         # carry height of the pinch point above its pick height
    ox: float           # set-down offset from the plate centre (m)
    oy: float


def wrap_half(a):
    return (a + np.pi / 2) % np.pi - np.pi / 2


def execute(env, p: Pick, on_step=None):
    c = env.bowl_pos()[:2]
    R = env.bowl_diameter / 2
    radial = np.array([np.cos(p.theta), np.sin(p.theta)])
    grasp = c + (R - 0.004) * radial
    yaw = wrap_half(p.theta - np.pi / 2)
    goal = env.plate_pos()[:2] + [p.ox, p.oy]
    place = grasp + (goal - c)                       # pinch point when the bowl is on the goal
    z_top = env.table_z + Z_TRAVEL
    z_pick = env.table_z + p.h
    z_carry = z_pick + p.lift
    z_place = z_pick + 0.02                           # plate rim is ~1.7 cm tall
    tilts = []

    def go(tgt, grip, n, frm=None):
        for k in range(n):
            x = tgt if frm is None else frm + (tgt - frm) * (k + 1) / n
            env.step_to(x, grip, yaw=yaw)
            tilts.append(env.bowl_tilt_deg())
            if on_step is not None:
                on_step()

    go(np.r_[grasp, z_top], -1, 16)
    go(np.r_[grasp, z_pick], -1, 12, np.r_[grasp, z_top])
    go(np.r_[grasp, z_pick], 1, 8)                                      # pinch the rim
    go(np.r_[grasp, z_carry], 1, 8, np.r_[grasp, z_pick])               # lift
    n = max(4, int(np.ceil(np.linalg.norm(place - grasp) / MOVE_STEP_M)))
    go(np.r_[place, z_carry], 1, n, np.r_[grasp, z_carry])              # carry
    go(np.r_[place, z_place], 1, 8, np.r_[place, z_carry])              # set down
    go(np.r_[place, z_place], -1, 6)                                    # release
    go(np.r_[place, z_top], -1, 10, np.r_[place, z_place])              # retreat
    go(np.r_[place, z_top], -1, 6)
    return np.array(tilts)


def cost(env, tilts):
    off = np.linalg.norm(env.bowl_pos()[:2] - env.plate_pos()[:2]) / env.bowl_diameter
    ok = env.success()
    return off + (0 if ok else 1.0) + max(0.0, env.bowl_tilt_deg() - 10) / 45, ok, off


# ---------------------------------------------------------------------------------------
def human_prior(intent=Path("data/processed/intent/scenario2_plate.json"), lift_D=0.36):
    """Grip angle (both clip phases grip the same near-left rim) and lift from the video.
    Human image angle → robot frame: robot = human − 90°. Lift: the bowl's apparent size
    grew ~8.5%, which at the camera distance implied by the bowl's size (~4.6 D) is ~0.36 D."""
    d = json.loads(Path(intent).read_text())
    angs = np.radians([d[k]["grip_angle_deg"] for k in ("place", "remove")])
    th_h = np.arctan2(np.sin(angs).mean(), np.cos(angs).mean())
    th = th_h - np.pi / 2
    return dict(theta=float(np.arctan2(np.sin(th), np.cos(th))), theta_sd=np.radians(30),
                lift_D=lift_D, source=d)


_W = None


def _init_worker():
    global _W
    os.environ.setdefault("MUJOCO_GL", "egl")
    import logging, warnings
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)
    _W = PlateEnv(cam_size=32, cameras=("agentview",))
    _W.reset(0)
    _W.cur, _W.start = 0, _W.get_state()


def _rollout(args):
    init_idx, prim = args
    if _W.cur != init_idx:
        _W.reset(init_idx)
        _W.cur = init_idx
        _W.start = _W.get_state()
    _W.set_state(_W.start)
    tilts = execute(_W, Pick(**prim))
    return cost(_W, tilts)


class PickPlanner:
    def __init__(self, workers=16):
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker)

    def close(self):
        self.pool.close()
        self.pool.join()

    def plan(self, init_idx, D, prior=None, rng=None, pop=48, elites=8, iters=4):
        rng = rng or np.random.default_rng()
        th_mu, th_sd = (prior["theta"], prior["theta_sd"]) if prior else (0.0, None)
        lift_mu = prior["lift_D"] * D if prior else np.mean(LIFT_RANGE)
        lift_sd = 0.015 if prior else 0.035
        h_mu, h_sd = np.mean(H_RANGE), 0.01
        o_mu, o_sd = np.zeros(2), 0.015
        best, first_ok, n = None, None, 0
        for _ in range(iters):
            th = rng.uniform(-np.pi, np.pi, pop) if th_sd is None else rng.normal(th_mu, th_sd, pop)
            prims = [Pick(float(t), float(np.clip(rng.normal(h_mu, h_sd), *H_RANGE)),
                          float(np.clip(rng.normal(lift_mu, lift_sd), *LIFT_RANGE)),
                          *map(float, rng.normal(o_mu, o_sd)))
                     for t in th]
            res = self.pool.map(_rollout, [(init_idx, asdict(p)) for p in prims])
            costs = np.array([r[0] for r in res])
            for k, r in enumerate(res):
                if first_ok is None and r[1]:
                    first_ok = n + k + 1
            n += pop
            order = np.argsort(costs)
            if best is None or costs[order[0]] < best[1][0]:
                best = (prims[order[0]], res[order[0]])
            el = [prims[i] for i in order[:elites]]
            a = np.array([e.theta for e in el])
            th_mu = float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))
            th_sd = max(np.radians(6), float(np.sqrt(-2 * np.log(np.hypot(np.sin(a).mean(), np.cos(a).mean()) + 1e-9))))
            hs = np.array([e.h for e in el]); h_mu, h_sd = hs.mean(), max(0.002, hs.std())
            ls = np.array([e.lift for e in el]); lift_mu, lift_sd = ls.mean(), max(0.004, ls.std())
            os_ = np.array([[e.ox, e.oy] for e in el]); o_mu, o_sd = os_.mean(0), max(0.003, os_.std())
        return best[0], best[1], first_ok, n

