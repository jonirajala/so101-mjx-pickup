"""Deploy the Squint-SAC vision policy (squint/train_sac.py) to the real SO-101.

Same hardware path as deploy.py — identical real<->MJX calibration, RealArmMJX I/O and
DeltaTargetController accumulator — but the policy is the dual-camera CNN trained in
MJX+Madrona instead of the privileged state policy. There is NO fixed cube constant and
NO FK grasp handoff: the policy perceives the cube live from the camera images.

The policy consumes (mirror squint/squint_env.SquintEnv):
  obs["state"]           = 12-d proprio [qpos(6), target_qpos(6)] — the controller's
                           accumulated target is the grasp-commitment memory.
  obs["pixels"]          = (16,16,6) [overhead, wrist] RGB, each captured high-res then
                           area-downsampled to 16x16 ("squinting"), rgb_u8 / 255.
  action                 = 6-d joint-delta (deterministic = tanh(mean) at deploy).

Runs on the deploy machine with: jax (CPU) + flax + mujoco + lerobot + opencv, PLUS the
wrist/overhead cams framed to match the sim cameras (wrist_camera.py). Madrona is NOT
needed — inference is a plain CNN forward pass.

  # dry-run the control loop with a blank frame (no camera, safe bring-up):
  python deploy_vision.py --pixel_source zeros --step
  # real run (dual cam; probe indices with `python wrist_camera.py --probe`):
  python deploy_vision.py --pixel_source webcam --dual_cam --camera 0 --overhead_camera 1
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

import deploy
from deploy import RealArmMJX, make_robot, reset_to, resolve_ckpt
from deploy_obs import ObsBuilder, DeltaTargetController
from model_loader import load_mj_model

# --- policy obs spec (constants, mirror envs/so101_vision; no MJX/Madrona import) ---
_OBS_RES = 16                           # policy sees 16x16 RGB (sim renders high -> area->16)
_PIX_CH = 3                             # RGB channels per camera
_ACTION_SIZE = 6

# --- Training start pose (MUST match squint/squint_env._SQUINT_START_ARM_DEG/_JAW_DEG) ---
# Arm reaching forward, gripper slightly open, shoulder_lift raised so the wrist cam sees the
# whole 14x20 cm spawn at t=0. This MUST be the deploy reset pose — a wrong start is OOD from step 0.
_SQUINT_START_ARM_DEG = np.array([0.0, -20.0, 0.0, 90.0, -90.0])
_SQUINT_START_JAW_DEG = 60.0   # start gripper (slightly open)

# Front-frame real<->MJX zero alignment for the vision deploy. Height (lift/elbow/wrist_flex) and
# roll come from deploy.ARM_OFFSET_DEG; only PAN differs. PAN is ANCHORED functionally: with a
# cube at a known front position, the offset is set so the arm actually reaches it (here real
# servo pan 0 <-> MJX pan -3.53deg, so offset = +3.53). Roll is left as-is: position-only IK
# cannot robustly anchor wrist_roll (underdetermined). Re-derive for YOUR arm.
VISION_ARM_OFFSET_DEG = deploy.ARM_OFFSET_DEG.copy()
VISION_ARM_OFFSET_DEG[0] = 3.53   # shoulder_pan: front-anchored

# Per-joint per-step action delta the policy trained with (squint_env._action_scale):
# arm +/-0.1, gripper +/-0.2 rad/step. Deploy MUST use this exact per-joint vector — a uniform
# scalar under-drives the gripper (2x the arm in training), so it can't close decisively and
# the closed-loop dynamics diverge from training.
_SQUINT_ACTION_SCALE = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.2])


def load_squint_sac_policy(ckpt_path: str):
    """Rebuild the SAC policy (squint/train_sac.py + squint_sac_net) and return
    act(proprio27, pixels) -> 6-d action. The pickle is (RunningStatisticsState, encoder_params,
    actor_params) saved by train_sac.py. Deploy action = tanh(mean) (the squashed-Gaussian MODE,
    no exploration noise). Pure flax/jax; running_statistics only reproduces the state norm."""
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "squint"))
    import jax.numpy as jp
    from brax.training.acme import running_statistics
    from brax.training import networks as bnet
    import squint_sac_net as SN

    with open(resolve_ckpt(ckpt_path), "rb") as f:
        norm, enc_p, act_p = pickle.load(f)

    enc, actor = SN.CNNEncoder(), SN.Actor(action_size=_ACTION_SIZE)

    def act(proprio27: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        s = jp.asarray(proprio27[None], jp.float32)
        s = running_statistics.normalize(s, bnet.normalizer_select(norm, "state"))
        feat = enc.apply(enc_p, jp.asarray(pixels[None], jp.float32))   # CNN gets [0,1], subtracts 0.5
        mean, _ = actor.apply(act_p, feat, s)
        return np.array(jp.tanh(mean))[0]              # deploy = squashed-Gaussian mode = tanh(mean)

    return act


def squint_start_pose(model):
    """(init_q_full, init_ctrl) for the trained start keyframe -- mirrors
    squint/squint_env.SquintEnv.__init__ EXACTLY: joints set DIRECTLY to [0,0,0,90,-90,60]deg
    (no IK, it's a fixed keyframe). run() uses only init_ctrl."""
    from ik import ARM_JOINTS
    arm_qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS])
    jaw_qadr = model.jnt_qposadr[model.joint("gripper").id]
    arm = np.deg2rad(_SQUINT_START_ARM_DEG); jaw = float(np.deg2rad(_SQUINT_START_JAW_DEG))
    q = np.array(model.qpos0)
    q[arm_qadr] = arm
    q[jaw_qadr] = jaw
    init_ctrl = np.zeros(model.nu)
    init_ctrl[:5] = arm
    init_ctrl[5] = jaw
    return q, init_ctrl


def normalize_rgb(rgb_u8: np.ndarray) -> np.ndarray:
    """(16,16,3) uint8 RGB -> float32 [0,1], matching so101_vision._pixels. The wrist cam must be
    cropped + AREA-downsampled to 16x16 (wrist_camera, INTER_AREA) — the deploy analog of the
    sim render->area-downsample. (Train-only blur/photometric DR is NOT applied at deploy.)"""
    x = np.asarray(rgb_u8, dtype=np.float32)
    assert x.shape[:2] == (_OBS_RES, _OBS_RES) and x.shape[2] in (3, 6), \
        f"rgb must be (16,16,3) wrist or (16,16,6) dual, got {x.shape}"
    return x / 255.0


# ======================================================================================
# Wrist RGB camera (pluggable).
# ======================================================================================
def make_pixel_source(kind: str, camera_index: int = 0, fov_zoom: float | None = None):
    """Return a callable() -> (16,16,3) uint8 RGB, framed + AREA-downsampled like the sim wrist cam.
    fov_zoom center-crops the real feed to emulate the policy's narrower sim fovy (see wrist_camera)."""
    if kind == "zeros":
        return lambda: np.zeros((_OBS_RES, _OBS_RES, _PIX_CH), np.uint8)
    if kind == "webcam":
        from wrist_camera import WristCamera
        # Mirror the SIM downsample EXACTLY: training renders the wrist cam at 128 then area-pools
        # 128->16 ("squinting", so101_vision area-mean). So frame the real feed to 128 (square-crop,
        # no zoom), then apply the SAME 8x8 area-mean to 16 — NOT a direct cv2 resize to 16.
        _SIM_RENDER = 128
        cam = WristCamera(index=camera_index, size=_SIM_RENDER, fov_zoom=fov_zoom)
        f = _SIM_RENDER // _OBS_RES                                  # 8
        def grab():
            x = cam.grab_rgb().astype(np.float32)                    # (128,128,3) sim-framed (size=_SIM_RENDER)
            x = x.reshape(_OBS_RES, f, _OBS_RES, f, 3).mean((1, 3))  # 128->16 area-mean == so101_vision._pixels
            return np.clip(x, 0, 255).astype(np.uint8)
        return grab
    raise ValueError(f"unknown --pixel_source {kind!r} (use 'zeros' or 'webcam')")


def make_dual_pixel_source(wrist_index: int, overhead_index: int, overhead_rot: int = 0,
                           fov_zoom: float | None = None):
    """Dual-cam source: callable() -> (16,16,6) uint8, channel-concat [overhead(0:3), wrist(3:6)]
    EXACTLY like so101_vision._pixels. Each cam is framed to 128 (square-crop) then 8x8 area-meaned
    to 16, mirroring the sim render->squint. The sim render is linear->sRGB-encoded in training and
    the real webcams are ALREADY sRGB, so NO gamma is applied here. Wrist keeps its 180deg
    inverted-mount rotation; the overhead gets its own --overhead_rot (typically 0)."""
    from wrist_camera import WristCamera
    _SIM_RENDER = 128
    f = _SIM_RENDER // _OBS_RES                                      # 8
    wrist = WristCamera(index=wrist_index, size=_SIM_RENDER, fov_zoom=fov_zoom)         # rot180 default
    overhead = WristCamera(index=overhead_index, size=_SIM_RENDER, rot_deg=overhead_rot, fov_zoom=None)

    def grab():
        o = overhead.grab_rgb().astype(np.float32).reshape(_OBS_RES, f, _OBS_RES, f, 3).mean((1, 3))
        w = wrist.grab_rgb().astype(np.float32).reshape(_OBS_RES, f, _OBS_RES, f, 3).mean((1, 3))
        x = np.concatenate([o, w], axis=-1)                         # (16,16,6) [overhead, wrist]
        return np.clip(x, 0, 255).astype(np.uint8)
    return grab


# ======================================================================================
# Control loop (mirrors deploy.run; cube pose replaced by live depth perception)
# ======================================================================================
def run(args):
    model = load_mj_model()
    ob = ObsBuilder(model)
    # Reset pose MUST match the policy's training t=0 distribution (the elevated squint
    # standoff, gripper open). Wrong start = OOD wrist view -> random motion.
    _, init_ctrl = squint_start_pose(model)
    # Match the policy's TRAINING action scale exactly; --deploy_action_scale is a unitless
    # multiplier (1.0 = exact match; lower for slower bring-up).
    base_scale = _SQUINT_ACTION_SCALE
    scale_vec = base_scale * args.deploy_action_scale
    # A low uniform deploy_action_scale also slows the JAW far below its trained close rate,
    # so it stalls half-open before the policy moves on. --gripper_action_scale lets the jaw
    # close at its trained rate while the arm stays gentle (pass 1.0 to shut decisively).
    # None = same as deploy_action_scale.
    if args.gripper_action_scale is not None:
        scale_vec = scale_vec.copy()
        scale_vec[5] = base_scale[5] * args.gripper_action_scale
    ctrl = DeltaTargetController(model, init_target=init_ctrl, action_scale=scale_vec)
    print(f"action scale: arm x{args.deploy_action_scale} jaw x"
          f"{args.gripper_action_scale if args.gripper_action_scale is not None else args.deploy_action_scale}",
          flush=True)
    # 'sac' = the SAC vision policy (squint/train_sac.py).
    act = load_squint_sac_policy(args.ckpt)
    print("reset/start pose: squint standoff (z=0.20, jaw open)",
          flush=True)
    print(f"loaded {args.policy} policy from {args.ckpt}", flush=True)
    # NO external zoom by default: only square-crop (min(h,w)) + resize the real feed, with the
    # SIM camera calibrated to match it. Over-cropping a normal ~70deg webcam pushes the cube out
    # of frame; sim-camera calibration is the correct knob. --fov_zoom only forces an override.
    fov_zoom = args.fov_zoom
    print(f"wrist fov_zoom: {fov_zoom if fov_zoom is not None else 'none (square-crop only)'}", flush=True)
    if args.dual_cam:
        # wrist+overhead: 6-ch [overhead, wrist]. Needs --overhead_camera; the policy ckpt's CNN
        # already has 6 input channels so act() consumes (16,16,6) unchanged.
        if args.pixel_source != "webcam":
            get_pixels = lambda: np.zeros((_OBS_RES, _OBS_RES, 6), np.uint8)   # dry-run blank dual frame
        else:
            get_pixels = make_dual_pixel_source(args.camera, args.overhead_camera,
                                                overhead_rot=args.overhead_rot, fov_zoom=fov_zoom)
        print(f"DUAL-CAM: wrist=cam{args.camera} (rot180) + overhead=cam{args.overhead_camera} "
              f"(rot{args.overhead_rot}) -> 6-ch [overhead, wrist]", flush=True)
    else:
        get_pixels = make_pixel_source(args.pixel_source, args.camera, fov_zoom=fov_zoom)

    robot = make_robot(args.port, args.calib_dir)
    # The policy trains in the FRONT-facing workspace -> use the front-frame pan calibration.
    arm_offset = VISION_ARM_OFFSET_DEG
    arm = RealArmMJX(robot, max_rel_deg=args.max_rel_deg, arm_offset_deg=arm_offset)
    print(f"pan calibration: {'front-frame anchored (+3.53)' if arm_offset is not None else 'default'}",
          flush=True)
    arm.start()
    from lerobot.utils.robot_utils import precise_sleep
    dt = 1.0 / args.hz

    try:
        print(f"resetting to hover pose over ~{args.reset_secs}s ...")
        reset_to(arm, init_ctrl, secs=args.reset_secs, hz=args.hz)
        ctrl.reset()

        # --- start-pose calibration diagnostic (read-only) ---
        # Compare what we COMMANDED (init_ctrl, the model start pose) against what the servos
        # actually READ back, plus the gripper height the model infers from the read angles.
        # A per-joint mismatch = the arm can't reach that commanded angle (limit / offset error);
        # a gripper height != ~12 cm above table with matching joints = a model/real kinematic gap.
        from ik import ARM_JOINTS
        q_chk = arm.read_q()
        print("\n--- START POSE CHECK (commanded vs actual) ---")
        print(f"{'joint':14s} {'cmd(deg)':>9s} {'read(deg)':>9s} {'delta':>7s}")
        for i, j in enumerate(ARM_JOINTS):
            cmd, rd = np.degrees(init_ctrl[i]), np.degrees(q_chk[i])
            print(f"{j:14s} {cmd:9.1f} {rd:9.1f} {rd-cmd:7.1f}")
        print(f"{'gripper(jaw)':14s} {np.degrees(init_ctrl[5]):9.1f} {np.degrees(q_chk[5]):9.1f} "
              f"{np.degrees(q_chk[5]-init_ctrl[5]):7.1f}")
        g_cmd, _ = ob.gripper_pose(init_ctrl)
        g_act, _ = ob.gripper_pose(q_chk)
        print(f"gripper FK height above table:  cmd={g_cmd[2]*100-1.05:5.1f}cm  "
              f"read={g_act[2]*100-1.05:5.1f}cm   (table at 1.05cm; model start = 12cm)")
        print(f"gripper FK forward(x):          cmd={g_cmd[0]*100:5.1f}cm  read={g_act[0]*100:5.1f}cm\n", flush=True)

        prev_q = arm.read_q()
        prev_action = np.zeros(_ACTION_SIZE)
        grasped_latch = False
        # The policy consumes a 12-d proprio = [qpos(6), target_qpos(6)] (measured joints + the
        # controller's accumulated TARGET) — the grasp-commitment memory + load/stall gap,
        # with no FK/velocity OOD surface.
        for t in range(args.max_steps):
            t0 = time.perf_counter()
            q = arm.read_q()
            qd = (q - prev_q) / dt          # finite-difference velocity (qvel DR covers noise)
            prev_q = q

            grip, _ = ob.gripper_pose(q)
            # proprio = [q(6), ctrl.target(6)] (the controller accumulator is the
            # commitment + load signal); the cube is perceived from the camera images.
            proprio27 = ob.build_state_minimal(q, ctrl.target)            # 12-d
            px_u8 = get_pixels()
            pixels = normalize_rgb(px_u8)

            action = np.clip(act(proprio27, pixels), -1.0, 1.0)
            prev_action = action          # policy sees its own intended action next step

            # HEIGHT-GATED GRASP (optional deploy band-aid): while the gripper-site is above
            # --grasp_below_z and we haven't latched, suppress JAW CLOSING (MJX gripper: + = open)
            # so the arm must descend to the cube before the jaw can close on air.
            action_exec = action
            if args.grasp_below_z is not None and not grasped_latch and grip[2] > args.grasp_below_z:
                action_exec = action.copy()
                action_exec[5] = max(float(action[5]), 0.0)

            if args.dump:
                import cv2, os, glob, csv
                d_dir = "out/rollout"
                os.makedirs(d_dir, exist_ok=True)
                if t == 0:                       # fresh run: clear stale frames + (re)start the trace
                    for _old in glob.glob(f"{d_dir}/step_*.png"):
                        os.remove(_old)
                    with open(f"{d_dir}/trace.csv", "w", newline="") as _f:
                        csv.writer(_f).writerow(
                            ["t", "a0_pan", "a1_lift", "a2_elbow", "a3_wflex", "a4_wroll", "a5_jaw",
                             "grip_x", "grip_y", "grip_z",          # REALIZED (from measured q)
                             "cmd_x", "cmd_y", "cmd_z"])            # COMMANDED (from ctrl.target accumulator)
                _cmd_grip, _ = ob.gripper_pose(ctrl.target)        # commanded gripper pose this step
                with open(f"{d_dir}/trace.csv", "a", newline="") as _f:
                    csv.writer(_f).writerow(
                        [t, *[round(float(x), 4) for x in action],
                         *[round(float(v), 4) for v in grip],
                         *[round(float(v), 4) for v in _cmd_grip]])
                def _panel(rgb16, tag):
                    p = cv2.cvtColor(cv2.resize(rgb16, (256, 256), interpolation=cv2.INTER_NEAREST),
                                     cv2.COLOR_RGB2BGR)
                    cv2.putText(p, tag, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    return p
                if px_u8.shape[-1] == 6:        # dual-cam obs = [overhead 0:3, wrist 3:6]
                    big = np.hstack([_panel(px_u8[..., :3], "OVERHEAD"),
                                     _panel(px_u8[..., 3:6], "WRIST")])
                else:                           # wrist-only (3-ch)
                    big = _panel(px_u8, "WRIST")
                cv2.putText(big, f"t={t} a={np.round(action, 2)}", (4, big.shape[0] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.imwrite(f"out/rollout/step_{t:03d}.png", big)
            prev_target = ctrl.target.copy()
            target = ctrl.step(action_exec)

            # z-floor descent veto (sim had a table, the real arm does not) — identical to
            # deploy.py: freeze the arm joints below z_floor but keep the jaw closing.
            grip_t, _ = ob.gripper_pose(target)
            floored = grip_t[2] < args.z_floor
            if floored:
                ctrl.target[:5] = prev_target[:5]
                target = ctrl.target.copy()

            jaw = q[5]
            grasped_latch = grasped_latch or (jaw < args.jaw_grasp_thresh)
            if grasped_latch and args.post_grasp_jaw is not None:
                ctrl.target[5] = float(np.clip(args.post_grasp_jaw, ctrl.lo[5], ctrl.hi[5]))
                target = ctrl.target.copy()

            if args.step:
                # diagnostic jaw load = the (target - qpos) gap carried implicitly in the obs
                load_str = f" load={max(0.0, float(q[5]-ctrl.target[5])):+.3f}"
                input(f"[{t}] a={np.round(action,2)} grasped={grasped_latch} jaw={jaw:+.3f}{load_str} "
                      f"z={grip_t[2]*1000:.0f}mm"
                      f"{'  <-FLOOR (descent vetoed)' if floored else ''}  Enter to send...")
            elif floored:
                print(f"[{t}] z-floor engaged: gripper at {grip_t[2]*1000:.0f}mm, descent vetoed")
            arm.send_target(target)
            precise_sleep(max(0.0, dt - (time.perf_counter() - t0)))
        print("done.")
    except ConnectionError as e:
        print(f"\nBUS DROPPED — a motor likely stalled (over-torque): {e}\n"
              "Power-cycle the arm to clear it (see deploy.py for the same guidance).")
    finally:
        arm.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", default="sac", choices=["sac"],
                    help="the SAC vision policy (squint/train_sac.py)")
    ap.add_argument("--ckpt", default="hf:jonirajala/so101-mjx-pickup/sac_dual_cam.pkl",
                    help="SAC policy checkpoint (norm,encoder,actor). Default = the released dual-cam "
                         "policy (use with --dual_cam --pixel_source webcam). Accepts a local path "
                         "(runs/<name>/policy_best.pkl) or hf:<repo>/<file>. Obs = 6-ch 16x16 "
                         "[overhead,wrist] pixels + 12-d proprio [qpos(6), target_qpos(6)] -- the "
                         "controller target is the grasp-commitment memory; deploy builds the identical "
                         "[q, ctrl.target] (build_state_minimal).")
    ap.add_argument("--pixel_source", default="zeros", choices=["zeros", "webcam"],
                    help="'zeros' = dry-run blank frame (no camera); 'webcam' = the wrist RGB cam "
                         "fed directly to the policy (see wrist_camera.py)")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV wrist-cam index (probe with "
                    "`python wrist_camera.py --probe`)")
    ap.add_argument("--dual_cam", action="store_true",
                    help="wrist+OVERHEAD dual-cam policy: feed 6-ch [overhead, wrist]. "
                         "Needs --overhead_camera and a 6-ch-CNN ckpt.")
    ap.add_argument("--overhead_camera", type=int, default=1,
                    help="OpenCV index of the OVERHEAD cam (dual-cam only; the across-table front view)")
    ap.add_argument("--overhead_rot", type=int, default=0,
                    help="overhead feed rotation deg (0/90/180/270); calibrate to your overhead mount "
                         "(wrist is rot180, overhead typically 0)")
    ap.add_argument("--fov_zoom", type=float, default=None,
                    help="OPTIONAL center-zoom on the real wrist crop. DEFAULT None = no zoom "
                         "(rot180 + square-crop + resize only). The right fix for a sim/real fovy "
                         "gap is sim-camera calibration, not cropping the real feed.")
    ap.add_argument("--port", default="/dev/tty.usbmodem5B415308971")
    ap.add_argument("--calib_dir", default=None)
    ap.add_argument("--hz", type=float, default=30.0,
                    help="control rate. The policy trains at 10 Hz but deploys well at 30 Hz with "
                         "deploy_action_scale 0.15: per-step delta 0.015/0.03 rad (smooth, 0.45 rad/s), "
                         "6.7x smaller than training's 0.1/0.2 -- the reactive policy tolerates the "
                         "slower, smoother regime.")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--deploy_action_scale", type=float, default=0.15,
                    help="unitless MULTIPLIER on the policy's per-joint training action scale "
                         "(arm +/-0.1, gripper +/-0.2). DEFAULT 0.15 with --hz 30 = the smooth deploy "
                         "regime (per-step delta 0.015/0.03 rad). (1.0 + --hz 10 = the raw training "
                         "regime: 6.7x coarser, janky on the real servos.)")
    ap.add_argument("--gripper_action_scale", type=float, default=None,
                    help="separate MULTIPLIER for the JAW only (default None = same as "
                         "--deploy_action_scale). Raise it (e.g. 1.0) so the jaw closes at its trained "
                         "rate while the arm stays gentle -- fixes the grasp stalling half-open because "
                         "the 0.15-scaled jaw is too slow to shut before the policy lifts.")
    ap.add_argument("--max_rel_deg", type=float, default=6.0, help="per-step servo safety clip")
    ap.add_argument("--jaw_grasp_thresh", type=float, default=0.2,
                    help="MJX gripper rad below this = grasped (for the optional post-grasp clamp)")
    ap.add_argument("--z_floor", type=float, default=0.012,
                    help="min gripper-site height (m); descent below it is vetoed")
    ap.add_argument("--grasp_below_z", type=float, default=None,
                    help="hold the jaw OPEN until the gripper-site drops below this height (m), so "
                         "the policy can't close on air too high. Try 0.035. Optional deploy band-aid.")
    ap.add_argument("--post_grasp_jaw", type=float, default=None,
                    help="once grasped, drive the jaw to this MJX rad (e.g. -0.174 = closed) for "
                         "cubes smaller than the 3cm trained one; torque-limited (stalls, no burnout)")
    ap.add_argument("--reset_secs", type=float, default=8.0)
    ap.add_argument("--step", action="store_true", help="gate every action on Enter (safe bring-up)")
    ap.add_argument("--dump", action="store_true", help="save each step's 16x16 policy input + action "
                    "to out/rollout/step_NNN.png (diagnose closed-loop divergence). "
                    "Dual-cam saves OVERHEAD|WRIST side by side.")
    args = ap.parse_args()
    run(args)
