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

Python env with: `lerobot` (arm I/O), `jax` (CPU), `flax`, `brax` (only
`running_statistics`/network utils for unpickling), `mujoco` (model constants), `opencv-python`,
`huggingface_hub` (checkpoint download). No renderer.

## Calibration

Real-servo ↔ MJX-model joint mapping lives in `deploy.py` (`DEG_OFFSETS`, gripper endpoint
mapping). The gripper endpoints were measured by hand-moving the jaw to its mechanical stops
and reading `Present_Position` — re-measure for your arm; a wrong closed-endpoint means the
policy's "close" never actually shuts the jaw. Probe camera indices with
`python wrist_camera.py --probe`.

## Run

```bash
# 1. dry-run the control loop with blank frames (no cameras, safe bring-up):
python deploy_vision.py --pixel_source zeros --step

# 2. real run — dual-cam, released checkpoint (downloads from HF):
python deploy_vision.py --pixel_source webcam --dual_cam \
    --camera 0 --overhead_camera 1 --hz 30 \
    --ckpt hf:jonirajala/so101-mjx-pickup/sac_dual_cam.pkl

# 3. diagnose a run: --dump writes per-step obs montages + trace.csv to out/rollout/
python deploy_vision.py ... --dump
```

The `--dump` trace (commanded vs measured joints, per-step 16×16 obs exactly as the policy
sees them) is the single most useful debugging artifact: it separates "the policy commands
the wrong thing" (sim2real gap) from "the arm doesn't reach what's commanded" (hardware).

## What to expect

Grasps across the 14×20 cm spawn area, lifts, holds, and returns toward the raised rest pose.
Weakest behaviours of the released checkpoint: re-approach after a missed/slipped grasp, and
off-centre approaches where the prongs can nudge the cube ("prong-push" — a known sim-fidelity
limit of the box-pad grasp collision). Training longer than the released 3M steps improves the
recovery behaviours; the success curve had not plateaued at cutoff.
