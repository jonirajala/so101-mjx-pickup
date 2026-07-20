"""Squint-style SO-101 cube-lift env (Squint: arXiv 2602.21203).

Subclasses SO101PickCubeVision. Specs:
  - Spawn: wide radial fan (dual-cam) or a 14x20 cm box (wrist-only) -- wide enough that the
    policy must localize the cube visually, which is what makes RL actually use vision.
  - Six-term dense reward: reach(1-tanh(5d)) + grasp + return-to-rest(exp(-2*restdist)*grasped)
    - 3*table_touch - 1*(~lifted) + dense_lift((lift/clearance)*grasped) + hold(lifted&grasped).
  - Horizon 90 steps @ 10 Hz control (9 s); per-step joint obs-noise sigma = 5 deg.
"""
import os, sys
import numpy as np
import jax, jax.numpy as jp
from ml_collections import config_dict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "envs"))
from so101_pick_cube import SO101PickCube, default_config, _TABLE_Z
from so101_vision import SO101PickCubeVision

# Lift clearance (m): the cube must rise this far above the table to count as lifted
# (forces a real lift, not a drag).
_LIFT_CLEARANCE = 0.04
# Wide fan spawn (dual-cam): r in [0.18, 0.42] m, theta +/-50 deg, opening forward along +X.
# The SO-101 max reach is 0.464 m, so the whole fan is reachable. The OVERHEAD cam localizes the
# full fan; the wrist alone cannot see the far edges.
_FAN_R = (0.18, 0.42)
_FAN_DEG = (-50.0, 50.0)

# Per-step photometric jitter on the rendered image (ColorJitter-style, sampled per step):
# brightness/contrast/saturation each U(0.7, 1.3); hue is a rotation about the grey axis of
# +/-0.05 of the FULL hue circle = 0.05*2*pi ~= 0.314 rad (+/-18 deg) -- note the bare 0.05 rad
# would be only 2.86 deg, 6.3x too weak. chan and gamma are identity: warm/cool tint and exposure
# DR are done IN-RENDER by the patched additive Madrona shader
# (bvh_raycast.cpp: albedo*(ambient + sum light_color*Lambert)) fed by vision_dr's per-episode
# light_diffuse/light_ambient draws; the linear->sRGB tone match lives in
# so101_vision._linear_to_srgb.
_CAM_JITTER = dict(bright=(0.7, 1.3), contrast=(0.7, 1.3), sat=(0.7, 1.3), hue=2.0 * np.pi * 0.05,
                      chan=(1.0, 1.0), gamma=(1.0, 1.0), pixel_noise=0.0)

# Wrist-only spawn box: 14x20 cm, x in [0.19, 0.33], y in [-0.10, 0.10]. Wide enough to force
# large-range visual servoing and cover real cube-placement variation. With the RAISED start pose
# below (cam at z~0.15 m) every corner is in the wrist frame at 6-9 px, and all 4 corners are
# IK-reachable (err < 0.1 mm). The deploy spawn must stay inside this box.
_SQUINT_BOX_LO = jp.array([0.19, -0.10])
_SQUINT_BOX_HI = jp.array([0.33, 0.10])

# RAISED start pose: shoulder_lift -20 deg puts the wrist cam at z~0.15 m pointing straight DOWN
# (camera down-ness -0.99), so its footprint (radius = height * tan(0.5*fov)) covers the WHOLE
# spawn box at t=0. The arm descends from here to grasp and RETURNS here (= rest target;
# success = lifted & grasped & back-at-raised). MUST match deploy_vision._SQUINT_START_ARM_DEG.
_SQUINT_START_ARM_DEG = (0.0, -20.0, 0.0, 90.0, -90.0)
_SQUINT_START_JAW_DEG = 60.0     # start gripper slightly open


def squint_config(obs_noise_qpos=0.087):
    cfg = default_config()
    # 10 Hz control: gamma 0.9 then spans ~1 s of real time, long enough to credit the lift.
    # n_substeps = ctrl_dt/sim_dt = 0.1/0.004 = 25.
    cfg.ctrl_dt = 0.1                             # 10 Hz policy control
    cfg.action_scale = 0.15                       # per-decision joint delta (rad)
    cfg.episode_length = 90                       # 90 steps @ 10 Hz = 9 s
    cfg.obs_noise.qpos = obs_noise_qpos           # sigma=5deg per step
    cfg.start_pose_noise = 0.02                   # initial qpos noise scale (rad)
    cfg.action_latency = 0
    cfg.joint_backlash = 0.0
    cfg.descent_gate_min = 1.0                    # 1.0 = no descent gating
    # Reward scales: each dense term / 3 -> max ~1.67/step, return ~16, inside the C51 support
    # [-20, 20]. table_touch penalizes the GRIPPER touching the table (scale -3*_R = -1.0).
    _R = 1.0 / 3.0
    # No action-rate penalty and no grasp gates: shaping that a 16x16-image policy cannot satisfy
    # suppresses the grasp commit.
    cfg.reward_config = config_dict.create(scales=config_dict.create(
        reach=_R, grasp=_R, rest=_R, table_touch=-3.0 * _R, not_lifted=-_R, lift=_R, hold=_R))
    return cfg


class SquintEnv(SO101PickCubeVision):
    # Proprio = [qpos(6), target_qpos(6)] ONLY (12-d). The controller TARGET is the
    # grasp-commitment memory: without it the policy has no anchor that the jaw was already driven
    # closed and reopens mid-grasp. qvel / gripper pose / prev_action are excluded (sim2real OOD
    # surface).
    _OBS_MINIMAL = True

    def __init__(self, num_envs, render_width=128, render_height=128, obs_noise_qpos=0.087,
                 spawn=None, dual_cam=False, jitter_kwargs=None,
                 overhead_pan=0.0, overhead_dropout=0.0, **kw):
        # dual_cam = wrist + OVERHEAD cameras. The overhead localizes the whole fan, so spawn
        # defaults to "fan" with it and to the small box wrist-only (from the wrist the cube is
        # ~1 px / out-of-frame at the wide-fan edges, so wrist-only + fan is unlearnable).
        self._dual_cam = dual_cam
        self._spawn = spawn if spawn is not None else ("fan" if dual_cam else "box")
        # Renders at 128, area-downsampled to the 16x16 obs; obs noise + reward via squint_config.
        super().__init__(num_envs=num_envs, render_width=render_width, render_height=render_height,
                         vision=True, rgb=True, compute_reward=True, dual_cam=dual_cam,
                         overhead_pan=overhead_pan, overhead_dropout=overhead_dropout,
                         config=squint_config(obs_noise_qpos),
                         jitter_kwargs=jitter_kwargs if jitter_kwargs is not None else _CAM_JITTER,
                         **kw)
        # Replace the base IK-hover start with the fixed extended keyframe above -- set the joints
        # DIRECTLY (no IK), since this is a fixed keyframe, not a reach target.
        _arm_start = np.deg2rad(_SQUINT_START_ARM_DEG)
        _jaw_start = float(np.deg2rad(_SQUINT_START_JAW_DEG))
        q_start = np.array(self._init_q)
        q_start[self._arm_qadr] = _arm_start
        q_start[self._jaw_qadr] = _jaw_start
        self._init_q = jp.array(q_start)
        self._home_arm_q = jp.array(_arm_start)                # rest target = raised start pose
        ic = np.array(self._init_ctrl)
        ic[:5] = _arm_start
        ic[5] = _jaw_start
        self._init_ctrl = jp.array(ic)
        # Per-joint action deltas: action in [-1,1] * scale -> arm +/-0.1, gripper +/-0.2 rad/step
        # (gripper faster so the close is decisive). 6 actuators: 5 arm + gripper.
        self._action_scale = jp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2])
        # Grasp detection is force-only: no geometric alignment gate (0.0 = off). A pre-contact
        # alignment test lets a visually-uncertain policy postpone the close forever; grasping a
        # rotated cuboid already physically requires rotating to align, so alignment never needs
        # to block the close.
        self._align_thresh = 0.0

    def _sample_box_xy(self, rng):
        if self._spawn == "box":
            return jax.random.uniform(rng, (2,), minval=_SQUINT_BOX_LO, maxval=_SQUINT_BOX_HI)
        kr, kt = jax.random.split(rng)
        r = jax.random.uniform(kr, (), minval=_FAN_R[0], maxval=_FAN_R[1])
        th = jp.deg2rad(jax.random.uniform(kt, (), minval=_FAN_DEG[0], maxval=_FAN_DEG[1]))
        # Fan opens FORWARD along +X, the arm's front (gripper FK at the start pose is
        # [+0.235, -0.01, 0.146]): x = r*cos(th), y = r*sin(th).
        return jp.array([r * jp.cos(th), r * jp.sin(th)])

    def _get_reward(self, data, info, action, grasped):
        box = data.xpos[self._box_body]
        grip = self._gripper_pos(data)
        reach = 1.0 - jp.tanh(5.0 * jp.linalg.norm(box - grip))
        # rest uses the controller TARGET qpos (arm only), NOT the measured qpos.
        rest_dist = jp.linalg.norm(data.ctrl[:5] - self._home_arm_q)
        lift_amt = jp.clip(box[2] - _TABLE_Z, 0.0, _LIFT_CLEARANCE)
        lifted = (box[2] - _TABLE_Z > _LIFT_CLEARANCE).astype(jp.float32)
        # Dense terms, no gates; `hold` gates on the INSTANTANEOUS grasp.
        return {
            "reach": reach,                                                # 1 - tanh(5d)
            "grasp": grasped,
            "rest": jp.exp(-2.0 * rest_dist) * grasped,                    # return-to-rest (target_qpos), gated
            "table_touch": self._touching_table(data),                    # gripper-table contact, scaled -3
            "not_lifted": (1.0 - lifted),                                  # *-1 penalty
            "lift": (lift_amt / _LIFT_CLEARANCE) * grasped,             # dense lift, grasp-gated
            "hold": lifted * grasped,                                      # lifted & grasped
        }
