"""Per-camera ablation for the dual-cam Squint-SAC policy: which camera drives the GRASP?

Rolls out N worlds (cube varies per world via DR) three ways, masking the policy's pixel input
(env physics untouched; only what the policy SEES changes):
  REAL        : both cameras (6-ch [overhead(0:3), wrist(3:6)])  -- what training saw
  NO_OVERHEAD : overhead channels zeroed -> wrist only
  NO_WRIST    : wrist channels zeroed    -> overhead only

Read:
  - grasp PRESERVED under NO_OVERHEAD + COLLAPSES under NO_WRIST => the WRIST drives the grasp;
    any deploy gap is wrist-cam fidelity (sim vs real wrist image).
  - grasp SURVIVES under NO_WRIST => the overhead, not the wrist, still drives terminal control.

  micromamba run -n madmjx python squint/probe_cam_ablation.py --ckpt <run>/policy_best.pkl --n 96
"""
import argparse, os, sys, pickle
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import jax, jax.numpy as jp

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "envs")); sys.path.insert(0, os.path.join(_CODE, "squint"))
from squint_env import SquintEnv
from so101_vision import vision_randomization_fn
import squint_sac_net as N
from brax.training.acme import running_statistics
from brax.training import networks as bnet
from mujoco_playground._src.wrapper import wrap_for_brax_training
from so101_pick_cube import _Z_SUCCESS, _TABLE_Z


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="squint/runs/sac_v1/policy_best.pkl")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--render", type=int, default=128)
    args = ap.parse_args()
    Nn = args.n

    env = SquintEnv(num_envs=Nn, render_width=args.render, render_height=args.render, dual_cam=True)
    ep_len = int(env._config.episode_length)
    wenv = wrap_for_brax_training(env, vision=True, num_vision_envs=Nn, episode_length=ep_len,
                                  randomization_fn=vision_randomization_fn(env, Nn))
    NA = env.action_size
    box_body = env._box_body; grip_site = env._gripper_site

    norm, enc_p, act_p = pickle.load(open(args.ckpt, "rb"))
    enc, actor = N.CNNEncoder(), N.Actor(action_size=NA)

    def _ns(s):
        return running_statistics.normalize(s, bnet.normalizer_select(norm, "state"))

    # mask: 0=REAL, 1=NO_OVERHEAD (zero 0:3), 2=NO_WRIST (zero 3:6)
    def masked(pix, mode):
        if mode == 1:
            pix = pix.at[..., 0:3].set(0.0)
        elif mode == 2:
            pix = pix.at[..., 3:6].set(0.0)
        return pix

    from functools import partial
    @partial(jax.jit, static_argnums=(1,))
    def act(obs, mode):
        feat = enc.apply(enc_p, masked(obs["pixels/view_wrist"], mode))
        mean, _ = actor.apply(act_p, feat, _ns(obs["state"]))
        return jp.tanh(mean)

    reset = jax.jit(wenv.reset); step = jax.jit(wenv.step)

    def rollout(mode):
        state = reset(jax.random.split(jax.random.PRNGKey(0), Nn))  # SAME seed/DR across modes
        gr_l, cz_l, gx_l, gy_l, bx_l, by_l = [], [], [], [], [], []
        for t in range(ep_len):
            a = act(state.obs, mode)
            state = step(state, a)
            d = state.data
            box = d.xpos[:, box_body]; grip = d.site_xpos[:, grip_site]
            gr_l.append(np.asarray(state.metrics["is_grasped"]))
            cz_l.append(np.asarray(box[:, 2]))
            gx_l.append(np.asarray(grip[:, 0])); gy_l.append(np.asarray(grip[:, 1]))
            bx_l.append(np.asarray(box[:, 0])); by_l.append(np.asarray(box[:, 1]))
        gr = np.stack(gr_l); cz = np.stack(cz_l)
        # final-frame localization: how well does gripper xy track cube xy at the end?
        gx, gy = np.array(gx_l[-1]), np.array(gy_l[-1]); bx, by = np.array(bx_l[-1]), np.array(by_l[-1])
        loc = 0.5 * (corr(gx, bx) + corr(gy, by))
        czg = np.where(gr > 0.5, cz, np.nan)
        max_czg = np.nanmax(np.where(np.isnan(czg), -1, czg), 0)
        return dict(grasp_once=float((gr.max(0) > 0.5).mean()),
                    grasp_steps=float((gr > 0.5).sum(0).mean()),
                    lifted=float((max_czg > _Z_SUCCESS).mean()),
                    max_czg_cm=float(np.nanmean(np.where(max_czg < 0, np.nan, max_czg)) * 100 - _TABLE_Z * 100),
                    loc=loc)

    print(f"=== CAMERA ABLATION  N={Nn}  ckpt={os.path.basename(args.ckpt)} ===")
    print(f"{'mode':12} {'grasp_once':>11} {'grasp_steps':>12} {'lifted':>8} {'maxCz(cm)':>10} {'loc_corr':>9}")
    for mode, name in [(0, "REAL"), (1, "NO_OVERHEAD"), (2, "NO_WRIST")]:
        r = rollout(mode)
        print(f"{name:12} {r['grasp_once']:>11.3f} {r['grasp_steps']:>12.1f} {r['lifted']:>8.3f} "
              f"{r['max_czg_cm']:>10.2f} {r['loc']:>9.3f}")


if __name__ == "__main__":
    main()
