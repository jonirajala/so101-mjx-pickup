# so101-mjx-pickup

**Sim-to-real cube pickup on the SO-101 arm — dual-camera RGB reinforcement learning, trained
entirely in MuJoCo MJX + Madrona on a single 8 GB GPU, deployed zero-shot on real hardware.**

This is a replication of **[Squint](https://github.com/aalmuzairee/squint)** (Almuzairee &
Christensen, [arXiv 2602.21203](https://arxiv.org/abs/2602.21203)), ported from its original
ManiSkill/SAPIEN setting to the MuJoCo MJX + Madrona stack — including the renderer patches
that porting turned out to require.

No real-world training data, no teleop demos, no motion capture. A SAC policy learns
grasp-and-lift from two 16×16 RGB views plus joint angles in a GPU-batched MJX simulation,
and the same weights drive the real arm through two webcams.

<p align="center"><i>overhead cam + wrist cam &nbsp;→&nbsp; 6×16×16 pixels + 12-d proprio &nbsp;→&nbsp; 6-d joint deltas @ 10 Hz</i></p>

## Why this needed engine patches

Stock Madrona-MJX raytraces a scene that *looks* different from what a webcam sees, and
several things a sim-to-real DR pipeline needs are not randomizable per world. The patches in
`engine-patches/` (also available as ready-built forks, see [docs/BUILD.md](docs/BUILD.md)) add:

- **additive Lambert shading** — `albedo·(ambient + Σ colour·N·L)`, the shading model a real
  webcam image actually resembles, replacing the stock path (plus shadows off, sRGB encode in
  the env),
- **per-world light colour/ambient DR** (`mjx.Model.light_diffuse` / `light_ambient`),
- **per-world colour override** (`matid = -2`) so cube-colour DR works on textured geoms,
- **per-world camera FOV** (`CameraFovScale`) so `cam_fovy` can be domain-randomized,
- CUDA 12.6 build fixes.

## What makes the transfer work

1. **Render appearance matched to the real camera** — additive shading, shadows off, linear→sRGB,
   exposure/ColorJitter DR. If the pixels are systematically off, no amount of policy DR saves you.
2. **"Squinting"** (from [Squint](https://github.com/aalmuzairee/squint),
   [arXiv 2602.21203](https://arxiv.org/abs/2602.21203)) — render at 128², area-downsample to 16².
   The policy never sees renderer-sharp pixels it could overfit.
3. **Minimal proprio** `[qpos(6), target_qpos(6)]` — the controller's accumulated target is the
   policy's grasp-commitment memory (it stays closed once driven closed) and carries the
   load/stall signal `(target − qpos)`. No FK poses, no velocities: less sim-to-real surface.
4. **DR with the right timing** — camera extrinsics jittered **per control step** (forces true
   visual servoing), appearance (cube colour/size, lights, floor) **per env**, physics
   (STS3215-like kp/damping/friction) per episode.
5. **Scale on a budget** — 1024 envs at UTD 0.25 for 3M steps fits an RTX 3060 Ti (7.3 GB peak)
   because the 1M-transition replay buffer lives in host RAM as fp16 pixels.

## Repo layout

```
envs/               MJX cube-pickup env + Madrona RGB rendering + domain randomization
squint/             SAC+C51 trainer, environment wrapper, policy networks and evaluator
models/             MJCF: SO-101 arm (grasp-pad collision), table scene, cameras
deploy.py           real-arm hardware library: real<->MJX calibration, LeRobot I/O
deploy_vision.py    the deploy entry point (webcams -> policy -> servo targets)
engine-patches/     Madrona / Madrona-MJX patches + build verification scripts
docs/               BUILD.md (renderer build, 8 GB, no sudo), DEPLOY.md (rig setup)
```

## Quickstart

**1. Build the patched renderer** — [docs/BUILD.md](docs/BUILD.md) (micromamba env, no sudo, ~30 min).

**2. Train** (8 GB GPU is enough):

```bash
micromamba run -n madmjx python squint/train_sac.py --dual_cam --name myrun
# ~smoke test first: python squint/train_sac.py --smoke
```

**3. Evaluate** with the failure-mode breakdown:

```bash
micromamba run -n madmjx python squint/rollout_diag.py --ckpt squint/runs/myrun/policy_best.pkl --dual_cam
```

**4. Deploy** on a real SO-101 — [docs/DEPLOY.md](docs/DEPLOY.md). Inference is CPU-only:

```bash
cp deploy_config.example.json deploy_config.json  # fill this with your calibration
python deploy_vision.py --config deploy_config.json \
  --ckpt squint/runs/myrun/policy_best.pkl \
  --pixel_source webcam --dual_cam --camera 0 --overhead_camera 1
```

## Method notes

RL: Squint-style SAC with a C51 distributional critic (101 atoms, support ±20), twin critics,
γ 0.9, τ 0.01, alpha autotune, shared CNN encoder trained by the critic (actor gets detached
features), DrQ random-shift + random-scale augmentation on the replay batch. Control is 10 Hz
position-target deltas (arm ±0.1 rad/step, gripper ±0.2), 9 s episodes. Reward is a seven-term
dense shaping (reach, grasp, not-lifted penalty, lift, hold, table-touch penalty,
return-to-rest), normalized to
~1/step. See `squint/train_sac.py` — every hyperparameter is an argparse flag with its meaning.

## License & attribution

MIT (see [LICENSE](LICENSE) for third-party notices). The training method follows
**Squint** — Almuzairee & Christensen, [arXiv 2602.21203](https://arxiv.org/abs/2602.21203),
reference implementation [aalmuzairee/squint](https://github.com/aalmuzairee/squint) (this repo
ports the method from its original ManiSkill/SAPIEN setting to MuJoCo MJX + Madrona).
SO-101 robot model from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(Apache-2.0). Renderer: [Madrona](https://github.com/shacklettbp/madrona) /
[Madrona-MJX](https://github.com/shacklettbp/madrona_mjx) (MIT). Arm I/O:
[lerobot](https://github.com/huggingface/lerobot).
