# Building the renderer: Madrona-MJX with the sim2real patches (8 GB GPU, no sudo)

Training renders through [Madrona-MJX](https://github.com/shacklettbp/madrona_mjx) (batch
GPU raytracer for MuJoCo MJX). This project needs a **patched** build — the stock renderer's
shading does not match what a real webcam sees, and several DR hooks are
missing (see `engine-patches/`). Two ways to get it:

- **Fork (recommended):** clone `jonirajala/madrona_mjx`, branch `so101-lighting`, with
  `--recursive` — its `external/madrona` submodule already points at the patched engine.
- **Patches:** clone upstream `shacklettbp/madrona_mjx` (recursive), then
  `git apply engine-patches/madrona_mjx_so101.patch` in the repo root and
  `git apply ../../engine-patches/madrona_engine_so101.patch` in `external/madrona`.

What the patches add:
1. **Additive Lambert lighting** in the raytracer — `albedo * (ambient + Σ light_colour · max(0, N·L))`,
   the shading model a real webcam image resembles far more than the stock path.
   Per-light colour + ambient are plumbed as `LightDesc` fields.
2. **Per-world light DR** — `mjx.Model.light_diffuse` / `light_ambient` consumed per world.
3. **Per-world colour override** (`matid = -2`) — lets `geom_rgba` DR actually change a textured
   geom's colour per world (stock Madrona ignores `geom_rgba` once a material is bound).
4. **Per-world camera FOV** — a `CameraFovScale` component so `cam_fovy` can be
   domain-randomized per world (the stock GPU path freezes FOV at construction).
5. **Build fixes** — drop `-G`/`-lineinfo` from NVRTC flags (CUDA 12.6 + cccl 3.5 rejects the
   megakernel with device debug info), `np.asarray(...)` coercion in `renderer.py` (nanobind
   requires CPU numpy arrays).

## Environment (no sudo, isolated micromamba env)

Pins: **JAX < 0.6** (0.5.3), CUDA toolkit **12.6** (13.x removes APIs Madrona uses),
MuJoCo/MJX 3.8.1, brax 0.12.4, cmake ≥ 3.31.

```bash
export MAMBA_ROOT_PREFIX=$HOME/micromamba
micromamba create -y -n madmjx -c conda-forge \
  python=3.11 "cuda-version=12.6" "cuda-toolkit" cudnn cmake \
  xorg-xorgproto xorg-libxinerama xorg-libxrandr xorg-libxcursor xorg-libxi \
  xorg-libxext xorg-libx11 xorg-libxrender xorg-libxfixes libxkbcommon \
  libgl-devel libegl-devel libopengl-devel mesalib
# PIN cuda-version IN EVERY later install or the solver drifts nvcc to 13.x.
micromamba run -n madmjx pip install "jax[cuda12_local]==0.5.3" \
  "mujoco==3.8.1" "mujoco-mjx==3.8.1" "brax==0.12.4" ml_collections etils opencv-python-headless
micromamba run -n madmjx pip install mujoco_playground==0.1.0   # only _src.mjx_env/_src.wrapper used
```

If the sysroot's `usr/lib64/{libm,libc}.so` linker scripts break the link with
`cannot open /lib64/libm.so.6`, rewrite them to absolute sysroot paths:

```bash
SR=$HOME/micromamba/envs/madmjx/x86_64-conda-linux-gnu/sysroot
ESC=$(echo "$SR" | sed 's/[\/&]/\\&/g')
for f in usr/lib64/libm.so usr/lib64/libc.so usr/lib/libm.so usr/lib/libc.so; do
  cp -n "$SR/$f" "$SR/$f.bak"
  sed -i -E "s@ /lib64/@ ${ESC}/lib64/@g; s@ /usr/lib64/@ ${ESC}/usr/lib64/@g" "$SR/$f"
done
```

## Build

```bash
git clone --recursive -b so101-lighting https://github.com/jonirajala/madrona_mjx ~/src/madrona_mjx
cd ~/src/madrona_mjx && mkdir build && cd build
micromamba run -n madmjx cmake .. -DLOAD_VULKAN=OFF -DCMAKE_BUILD_TYPE=Release
micromamba run -n madmjx cmake --build . -j$(($(nproc)-2))
cd .. && micromamba run -n madmjx pip install -e .
```

`-DLOAD_VULKAN=OFF` for headless boxes; use the **raytracer** backend (the rasterizer needs
Vulkan and will segfault at construction with it disabled).

## Runtime env vars (8 GB card)

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false          # JAX prealloc otherwise starves Madrona's heap
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$HOME/micromamba/envs/madmjx
export MADRONA_MWGPU_KERNEL_CACHE=$HOME/.madrona_cache/cache   # cache the slow megakernel JIT
```

## Verify the build

```bash
python engine-patches/verify_lighting_patch.py   # 4-world montage: per-world light colour/ambient
python engine-patches/verify_fov_patch.py        # 3-world montage: per-world cam_fovy
python engine-patches/vram_probe.py --mjcf models/so101_pick_cube.xml   # VRAM probe
```

Training at the released scale (1024 dual-cam envs, 128px render → 16×16) peaks at ~7.3 GB on
an RTX 3060 Ti. The 1M-transition replay buffer lives in host RAM (~6 GB, fp16 pixels).
