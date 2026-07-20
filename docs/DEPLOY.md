# Deploying to the real SO-101

Inference is a plain CNN forward pass — **no Madrona/GPU needed** on the deploy machine
(we run it on a MacBook). The policy consumes two webcam feeds downsampled to 16×16 and
12-d proprio, and emits 6-d joint deltas at deploy rate.

## Hardware

- **Arm:** SO-101 follower (STS3215 servos), driven via [lerobot](https://github.com/huggingface/lerobot).
- **Wrist camera:** USB RGB cam on the wrist-roll mount (ours: OV2710), mounted upside-down
  → frames are rotated 180°, then centre-cropped square and resized (identical to the sim
  framing; see `wrist_camera.py`).
- **Overhead camera:** any webcam on a fixed mount looking down the workspace, matched to
  the sim overhead pose (`models/so101_pick_cube.xml` camera `overhead`: eye ≈ [0.60, 0, 0.42],
  look-at ≈ [0.28, 0, 0], fovy 60). The DR (±6 cm pose jitter per step in training) covers
  small mounting errors — get it roughly right, not perfect.

## Software (deploy machine)

Create a separate environment and run `pip install -r requirements-deploy.txt`. Madrona is
not needed for inference.

## Calibration

Calibration is deliberately not shipped as a working default: offsets and gripper endpoints are
specific to each assembled arm. Copy `deploy_config.example.json` to the ignored
`deploy_config.json`, then fill it using this procedure:

1. Find the serial port and your existing LeRobot follower calibration ID. Put them in `port`
   and `robot_id`; set `lerobot_calibration_dir` only if you use a non-default directory.
2. Run `python deploy.py --port PORT --robot_id ID`. It disables torque. Hand-pose the arm to
   the training start/home pose shown by the model and press Enter; copy the printed first offset
   estimates into `arm_offset_deg`.
3. Determine each `arm_sign` with two limp-arm readings: move one joint alone in the positive
   MJX direction, read `Present_Position` before and after, and use `+1` if both deltas have the
   same sign or `-1` if they oppose. Recompute its offset as
   `real_home_deg - sign * mjx_home_deg`.
4. With torque still off, move the jaw to its physical closed stop and record
   `Present_Position` as `jaw_servo_closed`; repeat at the open stop for `jaw_servo_open`.
5. Anchor shoulder pan to the actual front workspace: place the gripper over a known point on the
   model centreline and adjust only the pan offset until real and model headings agree.
6. Start with the blank-pixel, step-gated command below. Before sending actions, inspect the
   printed commanded-versus-read joint check. Stop if errors are large or the inferred gripper
   height is wrong. Then verify camera orientation with `python wrist_camera.py --probe`.

The loader rejects the example's null placeholders and equal jaw endpoints. A wrong closed
endpoint can make the commanded close stop short, so do not guess these values.

## Run

```bash
# 1. dry-run the control loop with blank frames (no cameras, safe bring-up):
python deploy_vision.py --config deploy_config.json \
  --ckpt squint/runs/myrun/policy_best.pkl --pixel_source zeros --step

# 2. real run — dual-cam policy trained in this repository:
python deploy_vision.py --pixel_source webcam --dual_cam \
    --config deploy_config.json --ckpt squint/runs/myrun/policy_best.pkl \
    --camera 0 --overhead_camera 1 --hz 30

# 3. diagnose a run: --dump writes per-step obs montages + trace.csv to out/rollout/
python deploy_vision.py ... --dump
```

The `--dump` trace (commanded vs measured joints, per-step 16×16 obs exactly as the policy
sees them) is the single most useful debugging artifact: it separates "the policy commands
the wrong thing" (sim2real gap) from "the arm doesn't reach what's commanded" (hardware).

## What to expect

Grasps across the 14×20 cm spawn area, lifts, holds, and returns toward the raised rest pose.
A common residual failure mode is an off-centre approach where a prong nudges the cube before
closure ("prong-push"), a fidelity limit of the simplified box-pad grasp collision.
