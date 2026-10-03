"""Bowl-on-table LIBERO scene (Franka Panda, OSC end-effector control) with the helpers
the replay and planning code need.

Frames
  human:  x = image right, y = away from the person (top-down phone view)
  robot:  LIBERO world; the Panda base sits at x = -0.66 facing +x, so
          "away from the robot" = +x and "robot's right" = -y
  The person sat at the image's bottom edge facing up the image, i.e. in the same pose
  relative to the table as the robot, so the mapping is a fixed rotation:
          robot_x = human_y,  robot_y = -human_x
  Rim angles rotate the same way: robot_angle = human_angle - 90°.

Stepping
  `step()` runs the controller and physics only (≈4 ms per 20 Hz step). It skips
  LIBERO's observation and camera pipeline, which the planner doesn't need and which
  would cost ~10× more; call `render()` explicitly when frames are wanted.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

BDDL = Path(__file__).parent / "bddl" / "bowl_slide.bddl"
BOWL = "akita_black_bowl_1"
BOWL_MASS_KG = 0.25        # LIBERO's default is 6 g; a ceramic bowl is ~250 g
CTRL_HZ = 20
OSC_STEP_M = 0.05          # OSC output_max: action 1.0 → 5 cm per control step
OSC_STEP_RAD = 0.5         # … and 0.5 rad per control step
SETTLE_STEPS = 20          # LIBERO spawns objects slightly above the table
GRIP_SITE = "gripper0_grip_site"
# "birdview" is re-aimed to mimic the phone: straight down, far side of the table at the
# top of the image (camera +y = world +x, camera +x = world -y).
PHONE_CAM = dict(name="birdview", height=1.0, fovy=28.0, quat=(0.7071068, 0, 0, -0.7071068))


def human_to_robot(v):
    """Rotate planar vectors [..., 2] from the human (camera) frame into the robot frame."""
    v = np.asarray(v, float)
    return np.stack([v[..., 1], -v[..., 0]], axis=-1)


class BowlEnv:
    def __init__(self, cam_size=256, cameras=("agentview", "birdview")):
        from libero.libero.envs import OffScreenRenderEnv
        self.env = OffScreenRenderEnv(bddl_file_name=str(BDDL), camera_names=list(cameras),
                                      camera_heights=cam_size, camera_widths=cam_size)
        self.cameras = cameras

    def _bind(self):
        # LIBERO hard-resets rebuild the MjSim, so handles must be refreshed after reset.
        self.inner = self.env.env
        self.sim = self.inner.sim
        m = self.sim.model
        self.bowl_bodies = [i for i in range(m.nbody) if BOWL in m.body_id2name(i)]
        self.site = m.site_name2id(GRIP_SITE)
        self.n_sub = int(round(self.inner.control_timestep / m.opt.timestep))

    # ---- setup -------------------------------------------------------------------------
    def reset(self, bowl_xy=(0.0, 0.0), seed=0):
        self.env.seed(seed)
        self.env.reset()
        self._bind()
        m, d = self.sim.model, self.sim.data
        # Realistic bowl mass (scale inertia with it).
        main = self.bowl_bodies[0]
        k = BOWL_MASS_KG / m.body_mass[main]
        m.body_mass[main] *= k
        m.body_inertia[main] *= k
        # Place the bowl through its free joint.
        jid = m.body_jntadr[main]
        qadr = m.jnt_qposadr[jid]
        d.qpos[qadr:qadr + 2] = bowl_xy
        d.qvel[m.jnt_dofadr[jid]:m.jnt_dofadr[jid] + 6] = 0
        self.sim.forward()
        for _ in range(SETTLE_STEPS):
            self.step(np.zeros(7))
        lo, hi = self._bowl_extent()
        self.table_z = float(lo[2])                 # bowl bottom rests on the table top
        self.bowl_diameter = float(np.mean((hi - lo)[:2]))
        self.bowl_height = float(hi[2] - lo[2])
        self.bowl_rest_z = self.bowl_pos()[2]
        cid = m.camera_name2id(PHONE_CAM["name"])
        m.cam_pos[cid] = [0.0, 0.0, self.table_z + PHONE_CAM["height"]]
        m.cam_quat[cid] = PHONE_CAM["quat"]
        m.cam_fovy[cid] = PHONE_CAM["fovy"]
        self.sim.forward()

    def _bowl_extent(self):
        import mujoco
        m, d = self.sim.model, self.sim.data
        lo, hi = np.full(3, 1e9), np.full(3, -1e9)
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in self.bowl_bodies and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                vs = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                w = vs @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
        return lo, hi

    # ---- state -------------------------------------------------------------------------
    def bowl_pos(self):
        return self.sim.data.body_xpos[self.bowl_bodies[0]].copy()

    def bowl_tilt_deg(self):
        z_axis = self.sim.data.body_xmat[self.bowl_bodies[0]].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(z_axis[2], -1, 1))))

    def eef_pos(self):
        return self.sim.data.site_xpos[self.site].copy()

    def eef_rot(self):
        return self.sim.data.site_xmat[self.site].reshape(3, 3).copy()

    def get_state(self):
        return self.sim.get_state()

    def set_state(self, state):
        self.sim.set_state(state)
        self.sim.forward()

    def render(self):
        obs = self.inner._get_observations(force_update=True)
        return [obs[f"{c}_image"][::-1] for c in self.cameras]

    # ---- control -----------------------------------------------------------------------
    def step(self, action):
        policy_step = True
        for _ in range(self.n_sub):
            self.sim.forward()
            self.inner._pre_action(action, policy_step)
            self.sim.step()
            policy_step = False

    def step_to(self, target, grip, yaw=None, gain=1.0):
        """One 20 Hz control step toward a Cartesian target.

        grip: -1 open … +1 closed. yaw: gripper rotation about world z relative to the
        home orientation, where fingers close along world y; None keeps the current one.
        """
        a = np.zeros(7)
        a[:3] = np.clip((np.asarray(target) - self.eef_pos()) / OSC_STEP_M * gain, -1, 1)
        if yaw is not None:
            err = Rotation.from_matrix(self.home_rot(yaw) @ self.eef_rot().T).as_rotvec()
            a[3:6] = np.clip(err / OSC_STEP_RAD, -1, 1)
        a[6] = grip
        self.step(a)

    @staticmethod
    def home_rot(yaw):
        r0 = np.array([[0., 1, 0], [1, 0, 0], [0, 0, -1]])   # gripper pointing down
        return Rotation.from_euler("z", yaw).as_matrix() @ r0

    def close(self):
        self.env.close()
