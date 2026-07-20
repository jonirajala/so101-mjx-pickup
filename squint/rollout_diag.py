"""Roll out a trained Squint-SAC policy in MJX and log the lift dynamics. For each of N envs,
run one episode recording per step: gripper->cube reach, is_grasped, cube_z, gripper_z.
Then classify each env:
  - NEVER_GRASP: grasp never fires
  - GRASP_NO_RAISE: grasps but the cube stays ~at rest height
  - GRASP_THEN_SLIP: cube rises while grasped, then contact is lost and it drops
  - LIFTS: cube clears the success height while grasped

  python squint/rollout_diag.py --ckpt squint/runs/myrun/policy_best.pkl --dual_cam
"""
import argparse, os, sys, pickle
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # 8GB: JAX prealloc starves Madrona
import numpy as np
import jax, jax.numpy as jp

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "envs"))
from squint_env import SquintEnv
from so101_vision import vision_randomization_fn
import squint_sac_net as N
from so101_pick_cube import _TABLE_Z, _Z_SUCCESS
from brax.training.acme import running_statistics
from brax.training import networks as bnet
from mujoco_playground._src.wrapper import wrap_for_brax_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="local checkpoint produced by train_sac.py")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--render", type=int, default=128,
                    help="render resolution before 16x16 area reduction; must match training")
    ap.add_argument("--dual_cam", action="store_true", help="6-ch wrist+overhead checkpoint")
    args = ap.parse_args()
    Nn = args.n

    env = SquintEnv(num_envs=Nn, render_width=args.render, render_height=args.render,
                    dual_cam=args.dual_cam)
    ep_len = int(env._config.episode_length)
    wenv = wrap_for_brax_training(env, vision=True, num_vision_envs=Nn, episode_length=ep_len,
                                  randomization_fn=vision_randomization_fn(env, Nn))
    nstate = int(env._proprio_idx.shape[0]); NA = env.action_size
    RES, CH = env._obs_res, env._pix_channels
    box_body = env._box_body; grip_site = env._gripper_site

    norm, enc_p, act_p = pickle.load(open(args.ckpt, "rb"))
    enc, actor = N.CNNEncoder(), N.Actor(action_size=NA)

    def _ns(s):
        return running_statistics.normalize(s, bnet.normalizer_select(norm, "state"))

    @jax.jit
    def act(obs):
        feat = enc.apply(enc_p, obs["pixels/view_wrist"])
        mean, _ = actor.apply(act_p, feat, _ns(obs["state"]))
        return jp.tanh(mean)                          # deploy = mode (no exploration noise)

    reset = jax.jit(wenv.reset); step = jax.jit(wenv.step)
    k = jax.random.PRNGKey(0)
    state = reset(jax.random.split(k, Nn))

    # per-env, per-step logs
    home_arm = np.asarray(env._home_arm_q)            # 5 arm joints (rest target, gripper excluded)
    reach_l, grasp_l, cz_l, gz_l, rest_l = [], [], [], [], []
    for t in range(ep_len):
        a = act(state.obs)
        state = step(state, a)
        d = state.data
        box = d.xpos[:, box_body] if d.xpos.ndim == 3 else d.xpos[box_body]
        grip = d.site_xpos[:, grip_site] if d.site_xpos.ndim == 3 else d.site_xpos[grip_site]
        ctrl_arm = d.ctrl[:, :5] if d.ctrl.ndim == 2 else d.ctrl[:5]    # arm position targets
        cz_l.append(np.asarray(box[..., 2]))
        gz_l.append(np.asarray(grip[..., 2]))
        reach_l.append(np.asarray(jp.linalg.norm(box - grip, axis=-1)))
        grasp_l.append(np.asarray(state.metrics["is_grasped"]))
        rest_l.append(np.asarray(jp.linalg.norm(ctrl_arm - home_arm, axis=-1)))   # dist to rest pose
    cz = np.stack(cz_l); gz = np.stack(gz_l); gr = np.stack(grasp_l); rd = np.stack(reach_l)  # (T,N)
    rest = np.stack(rest_l)                                                        # (T,N) rest dist

    # classify per env
    LIFT_LINE = _Z_SUCCESS                       # cube center counted as lifted
    REST = _TABLE_Z
    n_grasp = (gr.max(0) > 0.5)                  # ever grasped
    grasp_steps = (gr > 0.5).sum(0)              # how many steps grasped
    # cube z WHILE grasped
    cz_when_grasped = np.where(gr > 0.5, cz, np.nan)
    max_cz_grasped = np.nanmax(np.where(np.isnan(cz_when_grasped), -1, cz_when_grasped), 0)
    lifted_while_grasped = (max_cz_grasped > LIFT_LINE)
    # slip: cube rose meaningfully above rest while grasped, then grasp lost & cube fell back
    rose = (np.nanmax(cz_when_grasped, 0) > REST + 0.02)
    ended_low_ungrasped = (gr[-1] < 0.5) & (cz[-1] < REST + 0.015)
    slip = rose & ended_low_ungrasped & n_grasp

    # ---- Rest-gated success: lifted & grasped & reached-rest (||arm_target - rest|| < 0.2 rad).
    # success_once = ever true; success_at_end = true at the last in-episode step (ep_len-2,
    # before the terminal-step reset clobber).
    succ_step = (gr > 0.5) & (cz > LIFT_LINE) & (rest < 0.2)   # (T,N) rest-gated success per step
    success_once = succ_step.any(0)                             # (N,) ever achieved
    success_at_end = succ_step[ep_len - 2]                      # (N,) held at episode end
    # lift-only (no rest gate) = the ENV-NATIVE success (grasped & cube above line):
    lg_step = (gr > 0.5) & (cz > LIFT_LINE)
    lift_once = lg_step.any(0)                                  # native success_once
    lift_at_end = lg_step[ep_len - 2]                           # native success_at_end (held at end)

    print(f"\n=== ROLLOUT DIAG  (N={Nn} envs, ep_len={ep_len}, ckpt={os.path.basename(args.ckpt)}) ===")
    print(f"rest cube_z={REST:.4f}  success lift line={LIFT_LINE:.3f}")
    print(f"REST-GATED success_once  (lift & grasp & at-rest, ever):  {success_once.mean()*100:5.1f}%")
    print(f"REST-GATED success_at_end(lift & grasp & at-rest, @end):  {success_at_end.mean()*100:5.1f}%")
    print(f"ENV-NATIVE success_once   (lift & grasp, ever):       {lift_once.mean()*100:5.1f}%")
    print(f"ENV-NATIVE success_at_end (lift & grasp, @end):       {lift_at_end.mean()*100:5.1f}%")
    # rest-distance probe: how close does the arm get to the rest pose WHILE lifted+grasped?
    lg = (gr > 0.5) & (cz > LIFT_LINE)                 # (T,N) lifted & grasped
    rest_when_lg = np.where(lg, rest, np.nan)
    min_rest_lg = np.nanmin(np.where(np.isnan(rest_when_lg), 9.9, rest_when_lg), 0)  # (N,) closest to rest
    min_rest_lg = min_rest_lg[lift_once]               # only envs that lifted
    rest_end = rest[ep_len - 2]                        # rest dist at episode end (all envs)
    print(f"rest-dist threshold = 0.20 rad.  WHILE lifted+grasped, closest approach to rest:")
    if min_rest_lg.size == 0:
        print("  (no envs lifted -> no rest-dist stats)")
    else:
        print(f"  min rest-dist per env: mean {min_rest_lg.mean():.3f}  p10={np.percentile(min_rest_lg,10):.3f} "
              f"p50={np.percentile(min_rest_lg,50):.3f} p90={np.percentile(min_rest_lg,90):.3f}  "
              f"(<0.20 in {(min_rest_lg<0.2).mean()*100:.0f}% of lifted envs)")
    print(f"  rest-dist at episode end: mean {rest_end.mean():.3f}  p50={np.percentile(rest_end,50):.3f}")
    print(f"ever grasped:        {n_grasp.mean()*100:5.1f}% of envs   (mean grasp steps/ep={grasp_steps.mean():.1f}/{ep_len})")
    print(f"min reach reached:   mean {rd.min(0).mean()*100:.1f} cm   (gripper got this close to cube)")
    print(f"max cube_z (all):    mean {cz.max(0).mean()*100:.2f} cm   (rest={REST*100:.2f})")
    print(f"max cube_z WHILE grasped: mean {np.where(max_cz_grasped<0,REST,max_cz_grasped).mean()*100:.2f} cm")
    print(f"LIFTED cube past success line while grasped: {lifted_while_grasped.mean()*100:5.1f}% of envs")
    print(f"GRASP-THEN-SLIP (rose>2cm then lost & fell):  {slip.mean()*100:5.1f}% of envs")
    print(f"GRASP_NO_RAISE (grasped but cube stayed within 1.5cm of rest): "
          f"{(n_grasp & (max_cz_grasped < REST+0.015)).mean()*100:5.1f}% of envs")
    # height histogram while grasped
    if n_grasp.any():
        hs = np.where(max_cz_grasped < 0, REST, max_cz_grasped)[n_grasp]
        print(f"max-cube-z-while-grasped distribution (cm): "
              f"p10={np.percentile(hs,10)*100:.2f} p50={np.percentile(hs,50)*100:.2f} "
              f"p90={np.percentile(hs,90)*100:.2f} max={hs.max()*100:.2f}")


if __name__ == "__main__":
    main()
