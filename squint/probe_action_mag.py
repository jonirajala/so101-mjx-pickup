"""Bisect perception-OOD vs policy: are the policy's action magnitudes gentle IN SIM?

Logs per-dim mean |action| over a sim rollout, split by phase (approach: cube on table vs carry:
cube up). If the sim magnitudes are gentle but the real arm over-commands, the gap is the real
wrist image (perception OOD); if sim is equally aggressive, it is the policy/reward itself.
"""
import argparse, os, sys, pickle
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false"); os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, jax, jax.numpy as jp
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "envs")); sys.path.insert(0, os.path.join(_CODE, "squint"))
from squint_env import SquintEnv
from so101_vision import vision_randomization_fn
import squint_sac_net as N
from brax.training.acme import running_statistics
from brax.training import networks as bnet
from mujoco_playground._src.wrapper import wrap_for_brax_training

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="squint/runs/sac_v1/policy_best.pkl")
ap.add_argument("--n", type=int, default=64); ap.add_argument("--render", type=int, default=128)
ap.add_argument("--dual_cam", action="store_true", default=True)
args = ap.parse_args(); Nn = args.n

env = SquintEnv(num_envs=Nn, render_width=args.render, render_height=args.render, dual_cam=True)
ep_len = int(env._config.episode_length)
wenv = wrap_for_brax_training(env, vision=True, num_vision_envs=Nn, episode_length=ep_len,
                              randomization_fn=vision_randomization_fn(env, Nn))
NA = env.action_size; box_body = env._box_body
norm, enc_p, act_p = pickle.load(open(args.ckpt, "rb"))
enc, actor = N.CNNEncoder(), N.Actor(action_size=NA)

def _ns(s): return running_statistics.normalize(s, bnet.normalizer_select(norm, "state"))
@jax.jit
def act(obs):
    feat = enc.apply(enc_p, obs["pixels/view_wrist"])
    mean, _ = actor.apply(act_p, feat, _ns(obs["state"]))
    return jp.tanh(mean)

reset = jax.jit(wenv.reset); step = jax.jit(wenv.step)
state = reset(jax.random.split(jax.random.PRNGKey(0), Nn))
acts, czs = [], []
for t in range(ep_len):
    a = act(state.obs); state = step(state, a)
    acts.append(np.asarray(a)); czs.append(np.asarray(state.data.xpos[:, box_body, 2]))
acts = np.stack(acts); czs = np.stack(czs)               # (T,N,6), (T,N)
approach = czs < 0.06                                      # cube still near table = approach/descent phase

names = ["pan", "lift", "elbow", "wristF", "wristR", "jaw"]
print(f"=== ACTION MAGNITUDE  SIM  N={Nn} ===")
print("Deploy refs: gentle |pan|=0.26 (locks) vs over-commanding |pan|=0.57 (wanders)\n")
print(f"{'dim':8} {'mean|a| all':>12} {'mean|a| approach':>17} {'std a (all)':>12}")
for i, nm in enumerate(names):
    aa = np.abs(acts[..., i])
    appr = np.abs(acts[..., i][approach])
    print(f"{nm:8} {aa.mean():>12.3f} {appr.mean():>17.3f} {acts[...,i].std():>12.3f}")
print(f"\n>> SIM |pan| approach = {np.abs(acts[...,0][approach]).mean():.3f}  "
      f"(gentle deploy ref 0.26 / over-command deploy ref 0.57)")
print("If ~0.26 -> gentle in sim, real over-command is PERCEPTION-OOD. If ~0.57 -> policy inherently aggressive (reward).")
