"""Watch the distilled policy live in an interactive MuJoCo window.

Each episode: random bowl start and goal (or one of the human-clip goals), the policy
runs closed-loop at real-time 20 Hz. Green disc = goal, grey ring = bowl start.
Mouse: rotate/zoom the camera. Close the window to stop.

Usage: python src/sim/live.py                                  # human-strategy policy
       python src/sim/live.py --policy runs/policy_none        # policy trained without it
       python src/sim/live.py --goals human_right --speed 0.5  # slow motion, right-clip goal
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "policy"))
from env import CTRL_HZ, SETTLE_STEPS, BowlEnv, human_to_robot   # noqa: E402
from evaluate import EXEC_K, MAX_STEPS, Policy                   # noqa: E402
from gen_demos import sample_task                                # noqa: E402
from planner import SUCCESS_D, SUCCESS_TILT                      # noqa: E402
from replay_hand import load_human                               # noqa: E402

CLIPS = dict(human_right="episode_000_right", human_left="episode_050_left")


def soft_reset(env, home_state, home_eef, start):
    """Reset without rebuilding the sim (a LIBERO hard reset would invalidate the viewer).

    Restoring MjData alone is not enough: the OSC controller keeps its previous goal, so
    the arm would drift away from home before the policy starts. Refresh the controller,
    then settle exactly as BowlEnv.reset does (zero actions), so the policy sees the same
    start distribution as in training and evaluation."""
    env.set_state(home_state)
    ctrl = env.inner.robots[0].controller
    ctrl.update(force=True)
    ctrl.reset_goal()
    m, d = env.sim.model, env.sim.data
    jid = m.body_jntadr[env.bowl_bodies[0]]
    qadr = m.jnt_qposadr[jid]
    d.qpos[qadr:qadr + 2] = start
    env.sim.forward()
    for _ in range(SETTLE_STEPS):
        env.step(np.zeros(7))


def draw_markers(viewer, env, start, goal):
    scn = viewer.user_scn
    scn.ngeom = 0
    R = env.bowl_diameter / 2
    z = env.table_z + 0.001
    for pos, rgba, size in [(goal, (0.1, 0.9, 0.2, 0.45), [R, 0.0008, 0]),
                            (start, (0.5, 0.5, 0.5, 0.35), [R, 0.0005, 0])]:
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, size,
                            np.r_[pos, z], np.eye(3).flatten(), np.array(rgba, np.float32))
        scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, default=Path("runs/policy_strategy"))
    ap.add_argument("--goals", choices=["random", "human_right", "human_left", "mixed"], default="mixed")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    env = BowlEnv(cam_size=64, cameras=("birdview",))
    env.reset()
    home, home_eef = env.get_state(), env.eef_pos()
    D = env.bowl_diameter
    pol = Policy(args.policy)
    rng = np.random.default_rng(args.seed)
    kinds = ["random", "human_right", "human_left"]
    wins = tries = 0

    with mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data) as viewer:
        viewer.cam.lookat[:] = [-0.05, 0.0, env.table_z]
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.1, -40, 200
        ep = 0
        while viewer.is_running():
            kind = kinds[ep % 3] if args.goals == "mixed" else args.goals
            start, goal = sample_task(rng, D)
            if kind != "random":
                goal = start + human_to_robot(load_human(CLIPS[kind])["bowl_rel"][-1]) * D
            soft_reset(env, home, home_eef, start)
            goal = goal + (env.bowl_pos()[:2] - start)
            start_xy = env.bowl_pos()[:2]
            draw_markers(viewer, env, start_xy, goal)
            viewer.sync()
            time.sleep(0.8 / args.speed)

            chunk = None
            for t in range(MAX_STEPS):
                if not viewer.is_running():
                    break
                t0 = time.perf_counter()
                if t % EXEC_K == 0:
                    chunk = pol(env.policy_obs(goal))
                a5 = chunk[t % EXEC_K]
                a = np.zeros(7)
                a[[0, 1, 2, 5]] = np.clip(a5[:4], -1, 1)
                a[6] = 1.0 if a5[4] > 0 else -1.0
                env.step(a)
                viewer.sync()
                time.sleep(max(0.0, 1 / (CTRL_HZ * args.speed) - (time.perf_counter() - t0)))

            err = np.linalg.norm(env.bowl_pos()[:2] - goal) / D
            ok = err < SUCCESS_D and env.bowl_tilt_deg() < SUCCESS_TILT
            wins += ok
            tries += 1
            print(f"episode {ep:3d} [{kind:11s}] {'SUCCESS' if ok else 'fail   '} "
                  f"err={err:.3f}D   running: {wins}/{tries}", flush=True)
            time.sleep(1.0 / args.speed)
            ep += 1
    env.close()


if __name__ == "__main__":
    main()
