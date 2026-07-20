"""Find the max num_envs the DUAL-CAM Madrona renderer fits on this GPU. Builds SquintEnv(dual_cam)
at --n, does reset+one step (triggers the batched render), prints peak GPU mem or fails (OOM)."""
import os, sys, argparse
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np, jax
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "envs"))

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, required=True); a = ap.parse_args()
from squint_env import SquintEnv
from so101_vision import vision_randomization_fn
from mujoco_playground._src.wrapper import wrap_for_brax_training
try:
    env = SquintEnv(num_envs=a.n, render_width=128, render_height=128, dual_cam=True)
    wenv = wrap_for_brax_training(env, vision=True, num_vision_envs=a.n,
                                  episode_length=int(env._config.episode_length),
                                  randomization_fn=vision_randomization_fn(env, a.n))
    st = jax.jit(wenv.reset)(jax.random.split(jax.random.PRNGKey(0), a.n))
    st = jax.jit(wenv.step)(st, np.zeros((a.n, env.action_size), np.float32))
    st.obs["pixels/view_wrist"].block_until_ready()
    import subprocess
    used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                                    "--format=csv,noheader"]).decode().strip()
    print(f"FIT n={a.n}  gpu={used}")
except Exception as e:
    print(f"FAIL n={a.n}  {type(e).__name__}: {str(e)[:160]}")
    sys.exit(1)
