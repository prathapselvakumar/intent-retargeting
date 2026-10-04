"""Shared pieces for the vision policy: what the policy sees and how the goal is shown.

The vision policy gets NO privileged bowl state. Its inputs are
  * two RGB cameras (front 'agentview', wrist 'eye_in_hand'), 256×256
  * robot proprioception: end-effector position above the table, gripper yaw, opening
  * a language instruction
The goal is shown visually: a translucent green disc of the bowl's size is projected
into both camera images at the goal location (an AR-style marker). The policy therefore
has to find the bowl and the target in pixels.
"""
import cv2
import numpy as np
from robosuite.utils.camera_utils import get_camera_transform_matrix

IMG = 256
CAMS = ("agentview", "robot0_eye_in_hand")
KEYS = ("observation.images.image", "observation.images.image2")   # LeRobot LIBERO naming
TASK = "slide the black bowl onto the green target"
GOAL_RGB = (40, 220, 60)
GOAL_ALPHA = 0.45


def state_vec(env):
    e = env.eef_pos()
    yaw = env.eef_yaw()
    return np.r_[e[0], e[1], e[2] - env.table_z, np.sin(yaw), np.cos(yaw),
                 env.gripper_opening()].astype(np.float32)


def project(env, cam, pts):
    """World points [N,3] → float pixel (col, row) in the upright image that
    BowlEnv.render returns (robosuite's matrix already matches it)."""
    P = get_camera_transform_matrix(env.sim, cam, IMG, IMG)
    h = np.c_[pts, np.ones(len(pts))] @ P.T
    uv = h[:, :2] / h[:, 2:3]
    return uv, h[:, 2]


def draw_goal(env, cam, img, goal_xy):
    R = env.bowl_diameter / 2
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ring = np.c_[goal_xy[0] + R * np.cos(ang), goal_xy[1] + R * np.sin(ang),
                 np.full_like(ang, env.table_z + 0.001)]
    px, depth = project(env, cam, ring)
    if (depth <= 0).any():                       # behind the camera
        return img
    over = img.copy()
    cv2.fillPoly(over, [np.round(px).astype(np.int32)], GOAL_RGB, lineType=cv2.LINE_AA)
    return cv2.addWeighted(over, GOAL_ALPHA, img, 1 - GOAL_ALPHA, 0)


def observe(env, goal_xy):
    """Camera images with the goal drawn in, keyed by LeRobot feature name."""
    imgs = env.render()
    return {k: draw_goal(env, c, np.ascontiguousarray(im), goal_xy)
            for k, c, im in zip(KEYS, CAMS, imgs)}
