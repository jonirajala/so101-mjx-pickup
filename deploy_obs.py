"""Deploy-side observation builder + action accumulator.

Reproduces the env's proprio obs and ctrl accumulation exactly, using plain CPU MuJoCo
(mj_forward) — no MJX needed on the deploy machine. LeRobot I/O and the real<->MJX joint
calibration live in deploy.py.
"""
import numpy as np
import mujoco

from model_loader import load_mj_model
from ik import ARM_JOINTS, solve_ik

_ACTION_SCALE = 0.04   # MUST match SO101PickCube default_config action_scale
# Env reset pose (so101_pick_cube: _HOVER_TARGET, jaw=0.6). Duplicated here (not imported)
# to keep the deploy path MJX-free.
_HOVER_TARGET = np.array([0.0, -0.20, 0.14])
_JAW_OPEN = 0.6


def home_pose(model: mujoco.MjModel):
    """(init_q_full, init_ctrl) for the IK hover pose used by the calibration helper."""
    arm_qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS])
    q_home, err = solve_ik(model, _HOVER_TARGET, jaw=_JAW_OPEN)
    assert err < 5e-3, f"home IK did not converge (err={err:.4f})"
    init_ctrl = np.zeros(model.nu)
    init_ctrl[:5] = q_home[arm_qadr]
    init_ctrl[5] = _JAW_OPEN
    return np.array(q_home), init_ctrl


class ObsBuilder:
    """Forward-kinematics gripper pose + the deploy proprio, from joint angles."""

    def __init__(self, model: mujoco.MjModel | None = None):
        m = model if model is not None else load_mj_model()
        self.m = m
        self.d = mujoco.MjData(m)
        arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        jaw_qadr = m.jnt_qposadr[m.joint("gripper").id]
        self.robot_qadr = np.append(arm_qadr, jaw_qadr)          # 6: arm(5)+jaw
        self.gripper_site = m.site("gripper").id

    def gripper_pose(self, arm_q):
        """(grip_xpos[3], grip_xmat[9]) for the gripper site at joint config arm_q."""
        self.d.qpos[self.robot_qadr] = np.asarray(arm_q, dtype=np.float64)
        mujoco.mj_forward(self.m, self.d)
        return (self.d.site_xpos[self.gripper_site].copy(),
                self.d.site_xmat[self.gripper_site].copy())

    def build_state_minimal(self, arm_q, target_qpos):
        """12-d proprio [qpos(6), target_qpos(6)], mirroring the training obs. target_qpos is
        the controller's accumulated position target — it carries the policy's commitment
        (stays closed once driven closed) and the load/stall gap (target - qpos)."""
        arm_q = np.asarray(arm_q, dtype=np.float32)
        target_qpos = np.asarray(target_qpos, dtype=np.float32)
        return np.concatenate([arm_q, target_qpos]).astype(np.float32)


class DeltaTargetController:
    """Accumulating position-target controller — mirrors the env step:
        target = clip(target + action * action_scale, ctrl_lo, ctrl_hi)
    `target` is the per-joint position target in MJX radians."""

    def __init__(self, model: mujoco.MjModel | None = None, action_scale=_ACTION_SCALE,
                 init_target=None):
        m = model if model is not None else load_mj_model()
        self.lo, self.hi = m.actuator_ctrlrange.T.copy()
        self.action_scale = action_scale
        init = home_pose(m)[1] if init_target is None else np.asarray(init_target)
        self.target = np.array(init, dtype=np.float64)
        self._home = self.target.copy()

    def reset(self):
        self.target = self._home.copy()
        return self.target.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()
        self.target = np.clip(self.target + action * self.action_scale, self.lo, self.hi)
        return self.target.copy()
