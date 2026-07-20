"""Real-SO-101 hardware library + calibration: the real<->MJX joint map, LeRobot I/O,
and the limp-arm calibration helper.

The vision deploy imports from here: calibrated servo I/O, LeRobot construction, and the
slow safe reset. Calibration values live in a user-owned JSON file, not this source tree.

The one thing you MUST derive for your own arm is the CALIBRATION — the real-servo <->
MJX-joint map (sign/offset per joint + the gripper endpoint map). Use the limp-arm
procedure documented in `docs/DEPLOY.md`.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from model_loader import load_mj_model
from deploy_obs import ObsBuilder, DeltaTargetController, home_pose

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
# Gripper (MJX "gripper" joint: + = open): linear map from MJX-rad to the real servo
# Present_Position. RAD endpoints = the model jaw jnt_range (closed -0.174, open 1.75).
JAW_RAD_CLOSED, JAW_RAD_OPEN = -0.174, 1.75
# SERVO endpoints: MEASURE on your arm by hand-moving the jaw (torque off) to its true
# mechanical stops and reading Present_Position at each. Do NOT assume a nominal range —
# a wrong closed-end stretches and shifts the whole map, the commanded "close" stops
# short, and the jaw never actually shuts (the policy grasps air).
def load_calibration(path: str) -> dict:
    """Load and validate a user-owned deploy calibration JSON file."""
    with open(path) as f:
        cfg = json.load(f)
    required = ("robot_id", "port", "arm_sign", "arm_offset_deg",
                "jaw_servo_closed", "jaw_servo_open")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing calibration fields: {', '.join(missing)}")
    if len(cfg["arm_sign"]) != 5 or len(cfg["arm_offset_deg"]) != 5:
        raise ValueError(f"{path}: arm_sign and arm_offset_deg must each contain five values")
    numeric = list(cfg["arm_sign"]) + list(cfg["arm_offset_deg"]) + [
        cfg["jaw_servo_closed"], cfg["jaw_servo_open"]]
    if any(v is None or not isinstance(v, (int, float)) for v in numeric):
        raise ValueError(f"{path}: replace every null calibration placeholder with a measurement")
    if any(v not in (-1, 1) for v in cfg["arm_sign"]):
        raise ValueError(f"{path}: every arm_sign value must be -1 or 1")
    if cfg["jaw_servo_closed"] == cfg["jaw_servo_open"]:
        raise ValueError(f"{path}: jaw closed and open endpoints must differ")
    return cfg


def real_to_mjx_q(real_deg: dict, calibration: dict) -> np.ndarray:
    """LeRobot Present_Position dict (deg) -> 6-d MJX joint vector (rad)."""
    arm = np.array([real_deg[k] for k in ARM_REAL_KEYS], dtype=np.float64)
    sign = np.asarray(calibration["arm_sign"], dtype=np.float64)
    offsets = np.asarray(calibration["arm_offset_deg"], dtype=np.float64)
    arm_rad = sign * np.deg2rad(arm - offsets)
    servo = real_deg["gripper"]
    closed, opened = calibration["jaw_servo_closed"], calibration["jaw_servo_open"]
    frac = (servo - closed) / (opened - closed)
    jaw_rad = JAW_RAD_CLOSED + frac * (JAW_RAD_OPEN - JAW_RAD_CLOSED)
    return np.append(arm_rad, jaw_rad)


def mjx_target_to_real(target_rad: np.ndarray, calibration: dict) -> dict:
    """6-d MJX position target (rad) -> LeRobot send_action dict (deg)."""
    sign = np.asarray(calibration["arm_sign"], dtype=np.float64)
    offsets = np.asarray(calibration["arm_offset_deg"], dtype=np.float64)
    arm_deg = np.rad2deg(np.asarray(target_rad[:5]) / sign) + offsets
    out = {f"{ARM_REAL_KEYS[i]}.pos": float(arm_deg[i]) for i in range(5)}
    jaw_rad = float(target_rad[5])
    frac = (jaw_rad - JAW_RAD_CLOSED) / (JAW_RAD_OPEN - JAW_RAD_CLOSED)
    closed, opened = calibration["jaw_servo_closed"], calibration["jaw_servo_open"]
    out["gripper.pos"] = closed + frac * (opened - closed)
    return out


# ======================================================================================
# Real arm (LeRobot). Thin wrapper applying the MJX calibration above.
# ======================================================================================
class RealArmMJX:
    def __init__(self, robot, calibration: dict, max_rel_deg: float = 6.0):
        self.robot = robot
        self.max_rel_deg = max_rel_deg     # per-step safety clip on commanded servo move
        self.calibration = calibration
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
        return real_to_mjx_q(self.robot.bus.sync_read("Present_Position"), self.calibration)

    def send_target(self, target_rad: np.ndarray):
        cmd = mjx_target_to_real(target_rad, self.calibration)
        if self._last_cmd is not None:               # per-step safety clip (degrees)
            for k, v in cmd.items():
                prev = self._last_cmd[k]
                cmd[k] = float(np.clip(v, prev - self.max_rel_deg, prev + self.max_rel_deg))
        self._last_cmd = dict(cmd)
        self.robot.send_action(cmd)


def make_robot(calibration: dict):
    """Construct the LeRobot so101 follower. lerobot 0.4.x ships it under `so_follower`
    (NOT `so101_follower`). `robot_id` resolves to the user's LeRobot calibration file
    under calibration/robots/so_follower. use_degrees=True so
    Present_Position is in degrees, which is what real_to_mjx_q assumes."""
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.utils import make_robot_from_config
    cfg = SO101FollowerConfig(port=calibration["port"], id=calibration["robot_id"],
                              use_degrees=True, cameras={},
                              calibration_dir=calibration.get("lerobot_calibration_dir"))
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
    config, read the servos, and print a first offset estimate. See docs/DEPLOY.md for the
    required two-pose sign check, jaw endpoint measurement, and verification procedure."""
    model = load_mj_model()
    _, init_ctrl = home_pose(model)
    expect_deg = np.rad2deg(init_ctrl[:5])
    # Calibration discovery only needs the LeRobot connection identity. The measured
    # values printed here should then be copied into deploy_config.json.
    robot = make_robot({"port": args.port, "robot_id": args.robot_id,
                        "lerobot_calibration_dir": args.lerobot_calibration_dir})
    robot.connect()
    # connect()->configure() leaves the servos torqued (stiff). Disable so the arm is limp
    # and can be hand-posed. Read-only, never energized.
    robot.bus.disable_torque()
    try:
        print("Torque is now OFF — arm is limp.")
        print("Target MJX home angles (deg), in joint order:")
        for name, angle in zip(ARM_REAL_KEYS, expect_deg):
            print(f"  {name:14s} {angle:+8.2f}")
        print("Hand-pose each joint to those angles using the model/viewer as reference, then Enter.")
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
    ap.add_argument("--port", required=True)
    ap.add_argument("--robot_id", required=True, help="your LeRobot calibration ID")
    ap.add_argument("--lerobot_calibration_dir", default=None)
    args = ap.parse_args()
    calibrate(args)
