"""Real-SO-101 hardware library + calibration: the real<->MJX joint map, LeRobot I/O,
and the limp-arm calibration helper.

The vision deploy (deploy_vision.py) imports from here: RealArmMJX (calibrated servo I/O),
make_robot (LeRobot so101 follower), reset_to (slow safe reset), resolve_ckpt (local/HF
checkpoint paths). Runs with lerobot + numpy + mujoco only — NO mjx, NO renderer.

The one thing you MUST derive for your own arm is the CALIBRATION — the real-servo <->
MJX-joint map (sign/offset per joint + the gripper endpoint map). Use the limp-arm
procedure: `python deploy.py --calibrate`.
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np

from model_loader import load_mj_model
from deploy_obs import ObsBuilder, DeltaTargetController, home_pose

# ======================================================================================
# CALIBRATION — real SO-101 servos <-> MJX joints. These values are for THIS arm:
# you MUST re-derive them for yours (`python deploy.py --calibrate`).
# ======================================================================================
# MJX joint order (policy convention): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
# LeRobot motor keys for the so101 follower, in the SAME physical order (VERIFY against
# `bus.sync_read("Present_Position").keys()` on your arm — order/names must line up):
ARM_REAL_KEYS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Per arm joint: mjx_rad = SIGN * deg2rad(real_deg - OFFSET_DEG). SIGN flips when the servo's
# positive direction opposes the MJX joint axis; OFFSET_DEG aligns the zeros.
# To derive: two-pose method with the arm limp (torque off) — hand-pose to known MJX configs
# and read the servos; SIGN = sign(delta_mjx * delta_real), OFFSET = real_home - SIGN*mjx_home
# (the --calibrate helper prints these). Anchor the PAN offset functionally: place a cube at a
# known model position and set the offset so the arm actually reaches it. Pan is independent
# of arm shape, so re-aiming the base does not disturb the other four offsets.
ARM_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
ARM_OFFSET_DEG = np.array([-89.18, 13.9, 0.0, 0.0, 90.0])
# Gripper (MJX "gripper" joint: + = open): linear map from MJX-rad to the real servo
# Present_Position. RAD endpoints = the model jaw jnt_range (closed -0.174, open 1.75).
JAW_RAD_CLOSED, JAW_RAD_OPEN = -0.174, 1.75
# SERVO endpoints: MEASURE on your arm by hand-moving the jaw (torque off) to its true
# mechanical stops and reading Present_Position at each. Do NOT assume a nominal range —
# a wrong closed-end stretches and shifts the whole map, the commanded "close" stops
# short, and the jaw never actually shuts (the policy grasps air).
JAW_SERVO_CLOSED, JAW_SERVO_OPEN = -65.89, 62.64



def real_to_mjx_q(real_deg: dict, arm_offset_deg: np.ndarray = ARM_OFFSET_DEG) -> np.ndarray:
    """LeRobot Present_Position dict (deg) -> 6-d MJX joint vector (rad). The vision deploy
    passes its own arm_offset_deg (deploy_vision.VISION_ARM_OFFSET_DEG)."""
    arm = np.array([real_deg[k] for k in ARM_REAL_KEYS], dtype=np.float64)
    arm_rad = ARM_SIGN * np.deg2rad(arm - arm_offset_deg)
    servo = real_deg["gripper"]
    frac = (servo - JAW_SERVO_CLOSED) / (JAW_SERVO_OPEN - JAW_SERVO_CLOSED)
    jaw_rad = JAW_RAD_CLOSED + frac * (JAW_RAD_OPEN - JAW_RAD_CLOSED)
    return np.append(arm_rad, jaw_rad)


def mjx_target_to_real(target_rad: np.ndarray, arm_offset_deg: np.ndarray = ARM_OFFSET_DEG) -> dict:
    """6-d MJX position target (rad) -> LeRobot send_action dict (deg)."""
    arm_deg = np.rad2deg(np.asarray(target_rad[:5]) / ARM_SIGN) + arm_offset_deg
    out = {f"{ARM_REAL_KEYS[i]}.pos": float(arm_deg[i]) for i in range(5)}
    jaw_rad = float(target_rad[5])
    frac = (jaw_rad - JAW_RAD_CLOSED) / (JAW_RAD_OPEN - JAW_RAD_CLOSED)
    out["gripper.pos"] = JAW_SERVO_CLOSED + frac * (JAW_SERVO_OPEN - JAW_SERVO_CLOSED)
    return out


def resolve_ckpt(spec: str) -> str:
    """Local path, or 'hf:<repo_id>/<file>' -> downloaded (cached) path via huggingface_hub."""
    if spec.startswith("hf:"):
        from huggingface_hub import hf_hub_download
        repo_id, _, fn = spec[3:].rpartition("/")
        return hf_hub_download(repo_id=repo_id, filename=fn, repo_type="model")
    return spec


# ======================================================================================
# Real arm (LeRobot). Thin wrapper applying the MJX calibration above.
# ======================================================================================
class RealArmMJX:
    def __init__(self, robot, max_rel_deg: float = 6.0, arm_offset_deg: np.ndarray | None = None):
        self.robot = robot
        self.max_rel_deg = max_rel_deg     # per-step safety clip on commanded servo move
        # Per-joint real<->MJX zero alignment; None = ARM_OFFSET_DEG. The vision deploy
        # passes VISION_ARM_OFFSET_DEG (pan re-aimed to the front workspace).
        self.arm_offset_deg = ARM_OFFSET_DEG if arm_offset_deg is None else np.asarray(arm_offset_deg)
        self._last_cmd = None
        from lerobot.motors.motors_bus import MotorNormMode
        self.robot.bus.motors["gripper"].norm_mode = MotorNormMode.DEGREES

    def start(self):
        self.robot.connect()

    def stop(self):
        try:
            self.robot.disconnect()
        except Exception as e:                # a stalled motor can drop the serial bus
            print(f"warning: clean disconnect failed ({e}). Power-cycle the arm to clear it.")

    def read_q(self) -> np.ndarray:
        return real_to_mjx_q(self.robot.bus.sync_read("Present_Position"), self.arm_offset_deg)

    def send_target(self, target_rad: np.ndarray):
        cmd = mjx_target_to_real(target_rad, self.arm_offset_deg)
        if self._last_cmd is not None:               # per-step safety clip (degrees)
            for k, v in cmd.items():
                prev = self._last_cmd[k]
                cmd[k] = float(np.clip(v, prev - self.max_rel_deg, prev + self.max_rel_deg))
        self._last_cmd = dict(cmd)
        self.robot.send_action(cmd)


def make_robot(port: str, calib_dir: str | None):
    """Construct the LeRobot so101 follower. lerobot 0.4.x ships it under `so_follower`
    (NOT `so101_follower`). Change `id` below to your own lerobot calibration id
    (resolves to .../calibration/robots/so_follower/<id>.json). use_degrees=True so
    Present_Position is in degrees, which is what real_to_mjx_q assumes."""
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config
    cfg = SO101FollowerConfig(port=port, id="robobobo_follower", use_degrees=True,
                              cameras={}, calibration_dir=calib_dir)
    return make_robot_from_config(cfg)


def reset_to(arm: RealArmMJX, target_q, secs=8.0, hz=30, max_rad_per_step=0.02):
    """Slowly interpolate ALL 6 joints (arm + jaw) to an absolute config (rad) — safe start.
    Includes the jaw on purpose: it must OPEN to the home value, else a gripper left closed
    from a prior run (e.g. --post_grasp_jaw) reads jaw<thresh at t=0, trips grasped_latch
    immediately, and the policy lifts the (empty) hand instead of descending."""
    from lerobot.utils.robot_utils import precise_sleep
    tgt = np.asarray(target_q, dtype=float).copy()
    pos = arm.read_q()
    for _ in range(int(secs * hz)):
        t0 = time.perf_counter()
        delta = np.clip(tgt - pos, -max_rad_per_step, max_rad_per_step)
        if np.linalg.norm(delta) < 1e-4:
            break
        pos = pos + delta
        arm.send_target(pos)
        precise_sleep(max(0.0, 1 / hz - (time.perf_counter() - t0)))


def calibrate(args):
    """Limp-arm calibration helper (torque off): hand-pose the arm to the MJX home/hover
    config, read the servos, and print the per-joint OFFSET that aligns real->MJX. Fill the
    printed values into ARM_OFFSET_DEG / ARM_SIGN above."""
    model = load_mj_model()
    _, init_ctrl = home_pose(model)
    expect_deg = np.rad2deg(init_ctrl[:5])
    robot = make_robot(args.port, args.calib_dir)
    robot.connect()
    # connect()->configure() leaves the servos torqued (stiff). Disable so the arm is limp
    # and can be hand-posed. Read-only, never energized.
    robot.bus.disable_torque()
    try:
        print("Torque is now OFF — arm is limp. Hand-pose it to the hover/home config, then Enter.")
        input()
        real = robot.bus.sync_read("Present_Position")
        got = np.array([real[k] for k in ARM_REAL_KEYS])
        print("joint            real_deg   mjx_home_deg   suggested_offset(real-mjx)")
        for i, k in enumerate(ARM_REAL_KEYS):
            print(f"{k:14s}  {got[i]:+8.2f}    {expect_deg[i]:+8.2f}      {got[i]-expect_deg[i]:+8.2f}")
        print("gripper servo:", real["gripper"], " (note closed/open extremes separately)")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/tty.usbmodem5B415308971")
    ap.add_argument("--calib_dir", default=None)
    args = ap.parse_args()
    calibrate(args)
