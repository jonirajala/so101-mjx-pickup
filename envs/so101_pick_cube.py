"""SO101PickCube — state-based cube-lift env on MJX.

A MuJoCo Playground MjxEnv. RL on proprio + privileged cube pose (renders nothing).
Reward: reach -> grasp -> grasp-gated lift -> hold. Obs is split into a
pixel-inferable `state` (actor) and a `privileged_state` (asymmetric critic).
"""
from typing import Any, Dict, Optional, Union
import sys
import os

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.mjx_env import State

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_loader import load_mj_model, SCENE_XML  # noqa: E402
from ik import solve_ik, ARM_JOINTS  # noqa: E402

_GRASP_SENSORS = ["fixed_pad1_box", "fixed_pad2_box", "fixed_pad3_box", "fixed_pad4_box",
                  "moving_pad1_box", "moving_pad2_box", "moving_pad3_box", "moving_pad4_box"]
_TABLE_SENSORS = [f"{p}p{i}_table" for p in ("f", "m") for i in (1, 2, 3, 4)]  # pad<->table force sensors (table conaff=0 -> inert)
_TABLE_TOUCH_FORCE = 0.01   # N: min normal force counting as "finger pressing the table"
# Min NORMAL pad contact force (N) to count as gripping: enforces a force-closed grip,
# not a graze that cannot lift (a firm grip reads 15-27 N, a graze <1 N).
_GRASP_FORCE = 0.5
_TABLE_Z = 0.0105         # cube rest height = nominal cuboid half-z (m)
_LIFT_H = 0.12            # lift saturates at +12 cm
_Z_SUCCESS = 0.05         # cube center counted as lifted: rest 0.0105 + ~0.04 m clearance
_Z_FLOOR = 0.02           # gripper below this = "too close to floor"
# Cube spawn box relative to base, kept inside the IK-verified SO-101 reachable -Y region
# (x in [-0.06,0.14], y in [-0.24,-0.12]).
_SPAWN_LO = jp.array([-0.06, -0.23])
_SPAWN_HI = jp.array([0.13, -0.13])
_HOVER_TARGET = np.array([0.03, -0.18, 0.13])  # ready pose: gripper hovers over workspace
# Descent-gate thresholds: the arm is gated only when the gripper z is in the terminal
# descent band [_GATE_Z_LOW, _GATE_Z_HIGH] AND the lateral grip->cube error exceeds ~_GATE_E_XY.
_GATE_Z_HIGH = 0.11   # m: above this the gripper is still approaching -> no gating
_GATE_Z_LOW = 0.05    # m: at/below this the gripper is in the terminal descent -> full gating weight
_GATE_E_XY = 0.04     # m: lateral grip->cube error at which "mis-aligned" saturates


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,        # 50 Hz control
        sim_dt=0.004,        # stable with elliptic cone / impratio=10
        episode_length=200,
        action_repeat=1,
        action_scale=0.04,
        # DR knobs, OFF by default so eval/deploy run the clean nominal env; the
        # trainer enables them alongside the model-level DR in envs/randomize.py.
        obs_noise=config_dict.create(
            qpos=0.0,   # rad std added to proprio joint angles (actor obs only)
            qvel=0.0,   # rad/s std added to proprio joint velocities
        ),
        action_latency=0,   # max action delay in control steps (0 = no latency)
        # STS3215 gear backlash / lost-motion DR (HardwareX 2026: ~0.62deg unloaded ->
        # ~1.30deg loaded, ~1-2mm at the gripper): per-joint deadband on the realized
        # position target, half-width ~U(0.5,1)*joint_backlash rad per episode. 0 = OFF.
        joint_backlash=0.0,
        start_pose_noise=0.03,   # rad std on the initial robot qpos (all joints)
        # <1 enables descent gating: the arm action is scaled toward this floor when the
        # gripper is low AND laterally off the cube (training-only, privileged xy). 1.0 = OFF.
        descent_gate_min=1.0,
        reward_config=config_dict.create(
            scales=config_dict.create(
                reach=1.0,
                grasp=1.0,
                lift=4.0,
                success=2.0,
                no_floor_collision=0.25,
                robot_target_qpos=0.3,
                action_rate=0.01,   # applied to a negative term -> penalty
            )
        ),
        impl="jax",
        # nconmax is PER-WORLD in the jax backend (the buffer is (num_envs, nconmax),
        # ~968 B/contact for elliptic-cone/condim=4 contacts). This scene has only
        # pad<->box + box<->floor contacts (tens, capped at 16/pair), so 256 is ample.
        nconmax=256,
        njmax=128,
    )


class SO101PickCube(mjx_env.MjxEnv):
    """Reach -> grasp -> lift a cube with the SO-101, state observations."""

    # Append an observable jaw-load dim to the actor `state` (see _get_obs). Default OFF (30-d state).
    _OBS_JAW_LOAD = False
    # Minimal proprio obs = [qpos(6), target_qpos(6)]: measured joints + the controller's accumulated
    # position target — the target is the grasp-commitment memory, and its gap with qpos is a signed,
    # persistent load signal. Overrides _OBS_JAW_LOAD when True.
    _OBS_MINIMAL = False

    def __init__(
        self,
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Union[str, int, list]]] = None,
    ):
        super().__init__(config or default_config(), config_overrides)
        mj_model = load_mj_model()
        mj_model.opt.timestep = self.sim_dt
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)
        self._xml_path = SCENE_XML.as_posix()
        self._action_scale = self._config.action_scale
        self._post_init()

    def _post_init(self):
        m = self._mj_model
        self._arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self._arm_dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self._jaw_qadr = m.jnt_qposadr[m.joint("gripper").id]
        self._jaw_dadr = m.jnt_dofadr[m.joint("gripper").id]
        self._robot_qadr = np.append(self._arm_qadr, self._jaw_qadr)
        self._robot_dadr = np.append(self._arm_dadr, self._jaw_dadr)
        self._compute_reward = True   # vision env sets False (obs+done only)
        self._gripper_site = m.site("gripper").id
        self._box_body = m.body("box").id
        self._box_geom = m.geom("box").id
        self._box_qadr = m.jnt_qposadr[m.joint("box_free").id]
        self._box_dadr = m.jnt_dofadr[m.joint("box_free").id]
        self._floor_geom = m.geom("floor").id
        self._sensor_adr = np.array([m.sensor_adr[m.sensor(s).id] for s in _GRASP_SENSORS])
        self._table_sensor_adr = np.array([m.sensor_adr[m.sensor(s).id] for s in _TABLE_SENSORS])
        self._lowers, self._uppers = m.actuator_ctrlrange.T
        # Alignment-aware grasp gate: 0.0 = OFF (force-only); >0 also requires jaw/cube-axis
        # alignment (see _grasp_align), thresh ~0.75 allows ~30 deg slop.
        self._align_thresh = 0.0
        # Gripper position actuator; its commanded-vs-actual angle gap is the observable jaw load.
        self._gripper_act = m.actuator("gripper").id

        # Ready pose via IK: gripper hovers above the workspace, jaw open.
        q_home, err = solve_ik(m, _HOVER_TARGET, jaw=0.6)
        assert err < 5e-3, f"home IK did not converge (err={err:.4f})"
        self._init_q = jp.array(q_home)
        self._home_arm_q = jp.array(q_home[self._arm_qadr])
        init_ctrl = np.zeros(m.nu)
        init_ctrl[:5] = q_home[self._arm_qadr]
        init_ctrl[5] = 0.6
        self._init_ctrl = jp.array(init_ctrl)

        # Obs-noise std matches the actor `state` layout (first 12 = arm_q[6]+arm_qd[6]);
        # everything downstream (gripper pose, rel cube, prev_action) stays clean.
        self._latency = int(self._config.action_latency)
        self._descent_gate_min = float(self._config.descent_gate_min)  # <1 enables descent gating
        self._backlash = float(self._config.joint_backlash)            # 0 = OFF (lost-motion DR)
        # Actor `state` length: minimal 12 (qpos+target_qpos), else 30 (+1 jaw-load).
        nq, nv = len(self._robot_qadr), len(self._robot_dadr)  # 6, 6
        if self._OBS_MINIMAL:
            self._state_dim = 2 * nq                            # 12: qpos(6) + target_qpos(6)
        elif self._OBS_JAW_LOAD:
            self._state_dim = 31
        else:
            self._state_dim = 30
        std = np.zeros(self._state_dim, dtype=np.float32)
        if self._OBS_MINIMAL:
            # Noise the MEASURED joint positions only; the controller TARGET is the policy's
            # own clean internal state, so dims 6:12 stay clean.
            std[:nq] = float(self._config.obs_noise.qpos)
        else:
            std[:nq] = float(self._config.obs_noise.qpos)
            std[nq:nq + nv] = float(self._config.obs_noise.qvel)
            if self._OBS_JAW_LOAD:
                # jaw-load dim: same jitter as the jaw angle it is built from.
                std[30] = float(self._config.obs_noise.qpos)
        self._obs_noise_std = jp.array(std)
        self._obs_noisy = bool(std.any())

    # ---- helpers ----
    def _gripper_pos(self, data):
        return data.site_xpos[self._gripper_site]

    def _grasp_align(self, data):
        # Alignment of the jaw-closing axis (gripper site local X, per so101_mjx.xml) with the
        # cube's nearest horizontal principal axis: a good grasp closes ACROSS a cube axis, not
        # diagonally. Returns max(cos^2) over the two cube axes: 1.0 = axis-aligned, 0.5 = 45°
        # diagonal (cos^2 folds the 180° jaw symmetry).
        jaw = data.site_xmat[self._gripper_site].reshape(3, 3)[:, 0]
        jaw_h = jaw[:2] / (jp.linalg.norm(jaw[:2]) + 1e-6)
        Rb = data.xmat[self._box_body].reshape(3, 3)
        ex = Rb[:2, 0] / (jp.linalg.norm(Rb[:2, 0]) + 1e-6)
        ey = Rb[:2, 1] / (jp.linalg.norm(Rb[:2, 1]) + 1e-6)
        return jp.maximum((jaw_h @ ex) ** 2, (jaw_h @ ey) ** 2)

    def _is_grasped(self, data):
        # data="force" contact sensors -> sensordata[adr] is the NORMAL force per pad. Require a
        # real grip force, not a touch, so the policy must SQUEEZE hard enough to lift.
        f = data.sensordata[self._sensor_adr]   # 8 pads: fixed 1-4, moving 1-4 (normal force, N)
        found = f > _GRASP_FORCE
        fixed = found[0] | found[1] | found[2] | found[3]    # any fixed-jaw pad grips the box
        moving = found[4] | found[5] | found[6] | found[7]   # any moving-jaw pad grips the box
        force_ok = fixed & moving       # BOTH jaws must be force-closed
        # Opt-in alignment gate: a force-closed grip must also have the jaw aligned to a cube face.
        if self._align_thresh > 0.0:
            force_ok = force_ok & (self._grasp_align(data) > self._align_thresh)
        return force_ok.astype(jp.float32)

    def _touching_table(self, data):
        # True when EITHER prong tip presses the table >= _TABLE_TOUCH_FORCE; used to
        # penalize over-descending the tips into the table.
        f = data.sensordata[self._table_sensor_adr]      # 3 finger meshes <-> table (normal force, N); max = any finger pressing
        return (jp.max(f) > _TABLE_TOUCH_FORCE).astype(jp.float32)

    # ---- API ----
    def _sample_box_xy(self, rng):
        """Cube spawn (x,y). Default = the reachable box; SquintEnv overrides with a
        wider fan spawn to force vision use."""
        return jax.random.uniform(rng, (2,), minval=_SPAWN_LO, maxval=_SPAWN_HI)

    def reset(self, rng: jax.Array) -> State:
        rng, rng_box, rng_arm, rng_delay, rng_yaw, rng_bl = jax.random.split(rng, 6)
        box_xy = self._sample_box_xy(rng_box)
        init_q = self._init_q
        init_q = init_q.at[self._box_qadr:self._box_qadr + 2].set(box_xy)
        # Rest the cube on the floor at ITS (possibly DR'd) half-height, read from the
        # per-world model — a fixed _TABLE_Z would leave smaller DR'd cubes floating
        # 1-4 mm up and free-falling at episode start.
        cube_half_z = self._mjx_model.geom_size[self._box_geom, 2]
        init_q = init_q.at[self._box_qadr + 2].set(cube_half_z)
        # Cube yaw DR: the cuboid is anisotropic (DR'd ~4-6 x 3-5 x 2 cm), so its footprint
        # vs the jaws depends on yaw; without it a real cube placed at an angle is OOD.
        yaw = jax.random.uniform(rng_yaw, (), minval=-jp.pi, maxval=jp.pi)
        init_q = init_q.at[self._box_qadr + 3:self._box_qadr + 7].set(
            jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)]))
        # Start-pose noise on ALL joints (5 arm + jaw): a FIXED start lets the policy
        # memorize a ballistic descent instead of closed-loop servoing, and on the real arm
        # small lateral error then compounds until the cube drifts out of the wrist frame.
        q_noise = self._config.start_pose_noise * jax.random.normal(rng_arm, (len(self._robot_qadr),))
        init_q = init_q.at[self._robot_qadr].add(q_noise)

        data = mjx_env.make_data(
            self._mj_model, qpos=init_q, qvel=jp.zeros(self._mjx_model.nv),
            ctrl=self._init_ctrl, impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax, njmax=self._config.njmax,
        )
        info = {"rng": rng, "prev_action": jp.zeros(self.action_size), "reached": 0.0,
                # latch: 1 once the episode has EVER hit the success condition (see step()).
                "succ_latch": jp.array(0.0),
                # prev_grasped: last step's grasp flag, so a reward can detect a DROP
                # (was grasped, now not). Updated at the END of step().
                "prev_grasped": jp.array(0.0),
                # grasp_steps: CONSECUTIVE steps the grip has been held (0 when lost);
                # updated BEFORE the reward in step().
                "grasp_steps": jp.array(0.0)}
        if self._backlash > 0:
            # Backlash state: cmd_target = the accumulated COMMANDED position target;
            # bl_center = the REALIZED output lagging it by the per-joint deadband bl_b.
            info["cmd_target"] = self._init_ctrl
            info["bl_center"] = self._init_ctrl
            info["bl_b"] = self._backlash * jax.random.uniform(
                rng_bl, (self.action_size,), minval=0.5, maxval=1.0)
        if self._latency > 0:
            # Ring buffer of recent actions (newest at index 0) + a per-episode integer
            # delay in [0, latency]: the applied target lags the policy by that many steps.
            info["act_buf"] = jp.zeros((self._latency + 1, self.action_size))
            info["act_delay"] = jax.random.randint(
                rng_delay, (), 0, self._latency + 1)
        metrics = {
            "success": jp.array(0.0),         # per-step held-up flag -> sums to a dwell-count
            "success_once": jp.array(0.0),    # latch delta -> sums to {0,1} = episode success RATE
            "lifted": jp.array(0.0),          # ENV-NATIVE lift (grasped & up, NO rest gate); best gates on this
            "is_grasped": jp.array(0.0),
            "reached": jp.array(0.0),
            "out_of_bounds": jp.array(0.0),
            **{k: jp.array(0.0) for k in self._config.reward_config.scales.keys()},
        }
        obs = self._get_obs(data, info)
        return State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        # Action-latency DR: the applied target lags the policy by `act_delay` steps;
        # `prev_action` stays the commanded action (what the real policy knows about itself).
        applied = action
        if self._latency > 0:
            buf = jp.roll(state.info["act_buf"], 1, axis=0).at[0].set(action)
            applied = buf[state.info["act_delay"]]
            state.info["act_buf"] = buf
        # Descent gating: scale the ARM action toward `descent_gate_min` when the gripper is
        # LOW and laterally OFF the cube (privileged pre-step xy, training only; action[5] free).
        if self._descent_gate_min < 1.0:
            grip = self._gripper_pos(state.data)
            box_xy = state.data.xpos[self._box_body][:2]
            e_xy = jp.linalg.norm(grip[:2] - box_xy)
            low = jp.clip((_GATE_Z_HIGH - grip[2]) / (_GATE_Z_HIGH - _GATE_Z_LOW), 0.0, 1.0)
            misalign = jp.clip(e_xy / _GATE_E_XY, 0.0, 1.0)
            gate = 1.0 - low * misalign * (1.0 - self._descent_gate_min)
            applied = applied.at[:5].multiply(gate)
        if self._backlash > 0:
            # Lost-motion backlash: the REALIZED target fed to physics lags the COMMANDED
            # accumulator by the per-joint deadband: center <- clip(center, cmd - b, cmd + b).
            cmd = jp.clip(state.info["cmd_target"] + applied * self._action_scale,
                          self._lowers, self._uppers)
            b = state.info["bl_b"]
            ctrl = jp.clip(state.info["bl_center"], cmd - b, cmd + b)
            state.info["cmd_target"] = cmd
            state.info["bl_center"] = ctrl
        else:
            ctrl = state.data.ctrl + applied * self._action_scale
            ctrl = jp.clip(ctrl, self._lowers, self._uppers)
        data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)

        grasped = self._is_grasped(data)  # needed for obs; reused into reward + _get_obs
        # Consecutive-grasp counter (reset to 0 the moment the grip is lost). Set BEFORE the
        # reward so a reward can gate lift/hold on a SUSTAINED grasp (commit-then-hold).
        state.info["grasp_steps"] = (state.info["grasp_steps"] + 1.0) * grasped
        box_pos = data.xpos[self._box_body]
        out_of_bounds = jp.any(jp.abs(box_pos[:2]) > 0.5) | (box_pos[2] < -0.05)
        done = (out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()).astype(float)
        state.info["prev_action"] = action

        # Per-step `success` sums to a dwell-count under brax's EvalWrapper; the latch DELTA
        # sums to {0,1}, so eval/episode_success_once is a true success RATE.
        # Full success = lifted AND the arm back near its rest pose (target within 0.2 rad).
        reached_rest = (jp.linalg.norm(data.ctrl[:5] - self._home_arm_q) < 0.2).astype(jp.float32)
        # `lifted` = grasped AND cube above the line, NO rest gate — the deploy-relevant
        # "picked the cube up" signal. Gate best-checkpoint selection on `lifted`, NEVER on
        # `success`: a policy that lifts but does not return home reads success ~0 all run.
        lifted = grasped * (box_pos[2] > _Z_SUCCESS).astype(jp.float32)
        success = lifted * reached_rest
        prev_latch = state.info["succ_latch"]
        cur_latch = jp.maximum(prev_latch, success)
        state.info["succ_latch"] = cur_latch
        success_once = cur_latch - prev_latch

        # Reward + metrics are RL-only; a consumer needing only obs + done can set
        # _compute_reward=False to skip this whole block in the render hot loop.
        if self._compute_reward:
            raw = self._get_reward(data, state.info, action, grasped)
            scales = self._config.reward_config.scales
            # A blown-up contact can NaN the reward; clip(nan)=nan, so sanitize. The done
            # flag (isnan qpos/qvel) resets the env on the same step.
            reward = jp.nan_to_num(jp.clip(sum(raw[k] * scales[k] for k in raw), -1e4, 1e4), nan=0.0)
            state.info["reached"] = jp.maximum(
                state.info["reached"],
                (jp.linalg.norm(box_pos - self._gripper_pos(data)) < 0.03).astype(float),
            )
            # Subclasses override _get_reward and may DROP "success" from raw, which would
            # leave metrics["success"] stuck at 0 — set it EXPLICITLY from the step-level
            # value so it is always live (the trailing key wins over any "success" in raw).
            state.metrics.update(
                {**raw, "is_grasped": grasped, "reached": state.info["reached"],
                 "out_of_bounds": out_of_bounds.astype(float),
                 "success_once": success_once, "success": success, "lifted": lifted},
            )
        else:
            # Reward-free path: reward = sparse success, so eval/episode_reward ≈ lift rate
            # while the expensive reach/lift/rest-pose shaping stays skipped.
            reward = success
            state.metrics.update(success=success, success_once=success_once, is_grasped=grasped, lifted=lifted)
        # Refresh prev_grasped AFTER _get_reward has read the previous value (drop detection).
        state.info["prev_grasped"] = grasped
        obs = self._get_obs(data, state.info, grasped)
        if self._obs_noisy:
            # Encoder/observation noise on the actor proprio only; the privileged
            # critic obs (built from clean state inside _get_obs) is left untouched.
            rng, k = jax.random.split(state.info["rng"])
            state.info["rng"] = rng
            obs = {**obs, "state": obs["state"]
                   + self._obs_noise_std * jax.random.normal(k, (self._state_dim,))}
        return State(data, obs, reward, done, state.metrics, state.info)

    def _get_reward(self, data, info, action, grasped) -> Dict[str, jax.Array]:
        box_pos = data.xpos[self._box_body]
        grip = self._gripper_pos(data)
        dist = jp.linalg.norm(box_pos - grip)

        reach = 1 - jp.tanh(10.0 * dist)
        lift = grasped * jp.clip((box_pos[2] - _TABLE_Z) / _LIFT_H, 0.0, 1.0)
        success = grasped * (box_pos[2] > _Z_SUCCESS).astype(jp.float32)
        # NOTE: mild tension with grasping — the gripper site must dip below _Z_FLOOR to
        # reach the cube, so this term (scale 0.25) slightly opposes the grasp at the
        # moment it matters; the grasp-gated lift (scale 4.0) dominates once grasped.
        no_floor_collision = (grip[2] > _Z_FLOOR).astype(jp.float32)
        robot_target_qpos = 1 - jp.tanh(
            jp.linalg.norm(data.qpos[self._arm_qadr] - self._home_arm_q)
        )
        action_rate = -jp.sum((action - info["prev_action"]) ** 2)
        return {
            "reach": reach,
            "grasp": grasped,
            "lift": lift,
            "success": success,
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
            "action_rate": action_rate,
        }

    def _get_obs(self, data, info, grasped=None) -> Dict[str, jax.Array]:
        # `grasped` is reused from step() (computed once for the reward) to avoid a second
        # sensordata gather/reduce per step; reset has no precomputed value -> compute here.
        if grasped is None:
            grasped = self._is_grasped(data)
        clean = lambda x: jp.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        if self._OBS_MINIMAL:
            # Minimal 12-D proprio: measured joint pos + the controller's accumulated position
            # TARGET (data.ctrl, arm5+jaw) — the policy's own commitment, available identically
            # on the real arm; the (target - qpos) gap is a signed, persistent load signal.
            # NO qvel/grip-pose/prev_action (extra OOD sim2real surface).
            qpos = data.qpos[self._robot_qadr]                 # 6 measured (noised in step)
            # 6 controller target (clean). With backlash on, data.ctrl is the BACKLASHED realized
            # target; the policy/deploy only knows the COMMANDED accumulator -> read it from info
            # (the real arm has no view of its gear-side lost-motion).
            target_qpos = info["cmd_target"] if self._backlash > 0 else data.ctrl
            state = jp.concatenate([qpos, target_qpos])
            # Privileged (critic only; SAC is symmetric so unused) = + abs cube pose/vel + grasp flag.
            privileged = jp.concatenate([
                state, data.xpos[self._box_body],
                data.qvel[self._box_dadr:self._box_dadr + 6], grasped[None],
            ])
            return {"state": clean(state), "privileged_state": clean(privileged)}
        grip = self._gripper_pos(data)
        grip_mat = data.site_xmat[self._gripper_site].ravel()
        box_pos = data.xpos[self._box_body]
        arm_q = data.qpos[self._robot_qadr]
        arm_qd = data.qvel[self._robot_dadr]

        # Actor state: proprio + gripper pose + gripper-relative cube pos + prev action.
        parts = [
            arm_q,                       # 6
            arm_qd,                      # 6
            grip,                        # 3
            grip_mat[3:],                # 6 (last two rows of rotation matrix)
            box_pos - grip,              # 3  (gripper-relative cube position)
            info["prev_action"],         # 6
        ]
        if self._OBS_JAW_LOAD:
            # Jaw load = relu(jaw_q - jaw_target): the gripper servo's commanded-vs-actual gap,
            # ~0 unless the jaw is loaded against something. A kp-invariant, unit-matched (MJX
            # radians) analogue of the STS3215 Present_Load; deploy computes the same quantity.
            jaw_q = data.qpos[self._jaw_qadr]
            jaw_target = data.ctrl[self._gripper_act]
            parts.append(jp.maximum(jaw_q - jaw_target, 0.0)[None])  # 1
        state = jp.concatenate(parts)
        # Privileged (critic only): absolute cube pose/vel + grasp flag.
        privileged = jp.concatenate([
            state,
            box_pos,                                          # 3  abs cube pos
            data.qvel[self._box_dadr:self._box_dadr + 6],     # 6  cube lin+ang vel
            grasped[None],                                    # 1  grasp flag (reused)
        ])
        # Sanitize so a single NaN/inf from a blown-up env never poisons the running
        # observation normalizer.
        clean = lambda x: jp.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return {"state": clean(state), "privileged_state": clean(privileged)}

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
