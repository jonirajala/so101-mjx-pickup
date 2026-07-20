"""Squint-style SAC+C51 (arXiv 2602.21203) in JAX over the Madrona-rendered MJX vision env.

Soft actor-critic with a C51 distributional critic (num_atoms 101, support [-20,20]), num_q=2
ensemble, gamma 0.9, tau 0.01, lr 3e-4, alpha autotune (target entropy -|A|), policy_frequency 4,
target update every step, shared CNN encoder (trained by the critic; actor uses DETACHED features).
Networks are defined in squint_sac_net.

The 1M replay buffer lives in HOST RAM (fp16 pixels, ~6 GB), which is what lets the full 1024-env
config fit an 8 GB GPU. Madrona render-order rule: the first render must happen inside the jitted
rollout, so all update/inference kernels are warm-compiled BEFORE it.

  micromamba run -n madmjx python train_sac.py --smoke
  micromamba run -n madmjx python train_sac.py --dual_cam --name sac_v1
"""
import argparse, functools, os, sys, pickle, time
# On the 8GB 3060 Ti, JAX's default 75% preallocation (~6GB) collides with Madrona's ~6.4GB peak
# and renderer init dies with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES (OOM disguised as a launch error).
# Disable preallocation so JAX grows on demand. MUST be set before jax is imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax, jax.numpy as jp
import optax
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training import networks as bnet

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE); sys.path.insert(0, os.path.join(_CODE, "envs"))
from so101_vision import SO101PickCubeVision, vision_randomization_fn
from so101_pick_cube import default_config
import squint_sac_net as N
from mujoco_playground._src.wrapper import wrap_for_brax_training


# ---------------------------------------------------------------- C51 projection
def _project(next_probs, reward, bootstrap, gamma):
    """C51 categorical projection. next_probs:(...,A) target dist; reward,bootstrap:(...,)."""
    A = N.NUM_ATOMS
    dz = (N.V_MAX - N.V_MIN) / (A - 1)
    tz = jp.clip(reward[..., None] + bootstrap[..., None] * gamma * N.Q_SUPPORT, N.V_MIN, N.V_MAX)
    b = (tz - N.V_MIN) / dz
    lo = jp.floor(b).astype(jp.int32); hi = jp.ceil(b).astype(jp.int32)
    lo_w = next_probs * (hi.astype(jp.float32) - b)
    hi_w = next_probs * (b - lo.astype(jp.float32))
    # handle lo==hi (integer b): all mass to lo
    eq = (lo == hi)
    lo_w = jp.where(eq, next_probs, lo_w)
    m = jp.zeros_like(next_probs)
    m = jax.vmap(lambda mm, i, w: mm.at[i].add(w))(m, lo, lo_w)
    m = jax.vmap(lambda mm, i, w: mm.at[i].add(w))(m, hi, hi_w)
    return m   # (...,A) projected target distribution


# ---------------------------------------------------------------- geometric image augmentation
# DrQ random-shift + RSA random-scale on the replay-batch pixels. DrQ/RAD show random shift is THE
# pixel-SAC sample-efficiency trick; RSA shows random scale stops the policy memorizing absolute
# pixel scale (which stalls the real-arm descent). One vmapped bilinear grid-sample does both:
# per-image scale s~U(1-a,1+a) and shift (dx,dy)~U(-shift,shift) px, edge-clamped (= replicate pad,
# DrQ-v2). Applied to BOTH pix and npix.
def _aug_one(key, img, max_shift, scale_amt):
    H, W, C = img.shape
    ks, kxy = jax.random.split(key)
    s = jax.random.uniform(ks, (), minval=1.0 - scale_amt, maxval=1.0 + scale_amt)  # >1 = zoom in
    d = jax.random.uniform(kxy, (2,), minval=-max_shift, maxval=max_shift)           # (dx, dy) px
    yy, xx = jp.meshgrid(jp.arange(H), jp.arange(W), indexing="ij")
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    xs = cx + (xx - cx) / s + d[0]      # source coords; /s zooms, +d shifts
    ys = cy + (yy - cy) / s + d[1]
    x0 = jp.floor(xs); y0 = jp.floor(ys)
    wx = xs - x0; wy = ys - y0
    x0i = jp.clip(x0, 0, W - 1).astype(jp.int32); x1i = jp.clip(x0 + 1, 0, W - 1).astype(jp.int32)
    y0i = jp.clip(y0, 0, H - 1).astype(jp.int32); y1i = jp.clip(y0 + 1, 0, H - 1).astype(jp.int32)
    Ia = img[y0i, x0i]; Ib = img[y1i, x0i]; Ic = img[y0i, x1i]; Id = img[y1i, x1i]
    wa = ((1 - wx) * (1 - wy))[..., None]; wb = ((1 - wx) * wy)[..., None]
    wc = (wx * (1 - wy))[..., None]; wd = (wx * wy)[..., None]
    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def _augment(key, imgs, max_shift, scale_amt):
    """imgs:(B,H,W,C) -> independently shifted+scaled per image (DrQ-v2 single draw)."""
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(_aug_one, in_axes=(0, 0, None, None))(keys, imgs, max_shift, scale_amt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="sac_v1")
    ap.add_argument("--num_envs", type=int, default=1024)   # fits the 8GB 3060 Ti at ~7.3GB peak
    #   (dual-cam, host replay buffer). Keep UTD 0.25 -> updates_per_step = num_envs/4.
    ap.add_argument("--render", type=int, default=128,
                    help="Madrona render res, area-downsampled to the 16x16 policy input. Render 128 "
                         "-> area-pool 16 is Squint's NAMED 'squinting': natural anti-aliasing that "
                         "aids sim2real (arXiv 2602.21203). 128@256 envs fits the 8GB card (~6.7GB).")
    ap.add_argument("--episode_length", type=int, default=200)
    ap.add_argument("--total_steps", type=int, default=3_000_000)   # env steps (success was still
    #   climbing at 3M -- longer is better if you have the time)
    ap.add_argument("--buffer", type=int, default=1_000_000)  # HOST/CPU buffer (fp16 pixels, ~6GB
    #   host RAM) keeps the 1M buffer off the 8GB GPU.
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--raw_env", action="store_true",
                    help="use the raw 50Hz UNNORMALIZED base env instead of the default SquintEnv "
                         "(10Hz, normalized reward, horizon 90); for comparison only. NOTE the C51 "
                         "support [-20,20] is sized for the normalized reward; --raw_env (return "
                         "~280) would need a much wider support.")
    ap.add_argument("--updates_per_step", type=int, default=256,    # UTD 0.25 at N=1024 (256/1024);
                    #   updates = num_envs/4.
                    help="gradient steps per env step; UTD = updates_per_step/num_envs (~0.25)")
    ap.add_argument("--learning_starts", type=int, default=5000)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--policy_frequency", type=int, default=4)
    ap.add_argument("--reward_scaling", type=float, default=1.0,
                    help="env reward scale. Keep 1.0: the SquintEnv reward is already normalized "
                         "(~1/step) to match the C51 support [-20,20].")
    ap.add_argument("--obs_noise_qpos", type=float, default=0.087)
    # geometric aug on the replay batch (DrQ shift + RSA scale). Defaults ON.
    ap.add_argument("--aug_shift", type=float, default=2.0,
                    help="DrQ random-shift max in PIXELS applied to replay pix/npix (edge-clamped "
                         "bilinear = replicate pad). 0 disables. At 16x16, 2 px ~ DrQ-v2's pad-4.")
    ap.add_argument("--aug_scale", type=float, default=0.3,
                    help="RSA random-scale half-range: per-image scale ~ U(1-x, 1+x) (e.g. 0.3 -> "
                         "0.7..1.3x). Targets the real-arm descent stall (scale overfit). 0 disables.")
    ap.add_argument("--eval_every", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dual_cam", action="store_true",
                    help="wrist+OVERHEAD dual-cam (6-ch [overhead,wrist]) + wide fan spawn")
    # Overhead robustness knobs (dual-cam only). Keep both 0: the overhead should stay ALWAYS
    # visible with only its POSE jittered per control step (±6cm/±4deg 3D extrinsic parallax in
    # so101_vision._jitter_cam_pose). Image-space pan or blanking STARVES the localization the
    # dual-cam policy needs (grasp plateaus ~0.5, lift ~0).
    ap.add_argument("--overhead_pan", type=float, default=0.0,
                    help="dual-cam: per-step overhead image-pan max (frame fraction); 0=off")
    ap.add_argument("--overhead_dropout", type=float, default=0.0,
                    help="dual-cam: prob the overhead channels are blanked per step; 0=off")
    args = ap.parse_args()
    if args.smoke:
        args.num_envs, args.total_steps, args.buffer = 64, 40_000, 20_000
        args.learning_starts, args.batch, args.episode_length = 2000, 256, 60
        args.updates_per_step = 8       # keep smoke fast (the full-scale default is far too slow at 64 envs)
    Nenv = args.num_envs

    if not args.raw_env:
        from squint_env import SquintEnv                  # DEFAULT: 10Hz normalized env
        env = SquintEnv(num_envs=Nenv, render_width=args.render, render_height=args.render,
                        obs_noise_qpos=args.obs_noise_qpos, dual_cam=args.dual_cam,
                        overhead_pan=(args.overhead_pan if args.dual_cam else 0.0),
                        overhead_dropout=(args.overhead_dropout if args.dual_cam else 0.0))
        ep_len = int(env._config.episode_length)          # SquintEnv horizon (90), NOT args.episode_length
    else:
        cfg = default_config(); cfg.obs_noise.qpos = args.obs_noise_qpos
        env = SO101PickCubeVision(num_envs=Nenv, render_width=args.render, render_height=args.render,
                                  vision=True, compute_reward=True, config=cfg)
        ep_len = args.episode_length
    wenv = wrap_for_brax_training(env, vision=True, num_vision_envs=Nenv, episode_length=ep_len,
                                  randomization_fn=vision_randomization_fn(env, Nenv),
                                  full_reset=True)  # re-run reset() per episode -> per-episode render DR
    nstate = int(env._proprio_idx.shape[0]); NA = env.action_size
    RES, CH = env._obs_res, env._pix_channels
    # Print the ACTUAL config (ep_len/ctrl_dt decide the horizon, not args.episode_length).
    print(f"[sac] env={'raw50Hz' if args.raw_env else 'squint10Hz'} N={Nenv} render={args.render} "
          f"ep={ep_len} ctrl_dt={float(env._config.ctrl_dt):.3f} gamma={args.gamma} "
          f"support=[{N.V_MIN},{N.V_MAX}]/{N.NUM_ATOMS} buffer={args.buffer} "
          f"updates/step={args.updates_per_step}"
          + (f" ovh_pan={args.overhead_pan} ovh_drop={args.overhead_dropout}" if args.dual_cam else ""),
          flush=True)

    # -------- networks + params --------
    enc, actor, critic = N.CNNEncoder(), N.Actor(action_size=NA), N.Critic()
    k = jax.random.PRNGKey(args.seed)
    k, ke, ka, kc = jax.random.split(k, 4)
    dpix = jp.zeros((1, RES, RES, CH)); dstate = jp.zeros((1, nstate)); dact = jp.zeros((1, NA))
    enc_p = enc.init(ke, dpix)
    feat0 = enc.apply(enc_p, dpix)
    act_p = actor.init(ka, feat0, dstate)
    crit_p = critic.init(kc, feat0, dstate, dact)
    tcrit_p = crit_p                              # target critic starts = critic
    log_alpha = jp.zeros(())
    target_entropy = -float(NA)

    cri_opt = optax.adam(args.lr); act_opt = optax.adam(args.lr); alp_opt = optax.adam(args.lr)
    cri_os = cri_opt.init({"enc": enc_p, "crit": crit_p})
    act_os = act_opt.init(act_p)
    alp_os = alp_opt.init(log_alpha)

    # running obs normalizer over the proprio state (pixels excluded).
    norm = running_statistics.init_state({"state": specs.Array((nstate,), jp.dtype("float32"))})

    def _ns(norm, s):
        return running_statistics.normalize(s, bnet.normalizer_select(norm, "state"))

    # -------- losses --------
    AUG = (args.aug_shift > 0.0) or (args.aug_scale > 0.0)

    def critic_loss_fn(enc_p, crit_p, tcrit_p, act_p, norm, log_alpha, batch, key):
        s = _ns(norm, batch["state"]); ns = _ns(norm, batch["nstate"])
        # DrQ+RSA: augment pix and npix INDEPENDENTLY (DrQ-v2 single draw each) before the
        # encoder. The encoder is trained by this critic loss, so this is where the aug must live.
        if AUG:
            kp, knp, key = jax.random.split(key, 3)
            pix = _augment(kp, batch["pix"], args.aug_shift, args.aug_scale)
            npix = _augment(knp, batch["npix"], args.aug_shift, args.aug_scale)
        else:
            pix, npix = batch["pix"], batch["npix"]
        nfeat = enc.apply(enc_p, npix)                          # online encoder on next (no tgt enc)
        nmean, nlogstd = actor.apply(act_p, jax.lax.stop_gradient(nfeat), ns)
        na, nlogp = N.sample_action(nmean, nlogstd, key)
        tlogits = critic.apply(tcrit_p, nfeat, ns, na)          # (Q,B,A) target dist per Q-net
        tprobs = jax.nn.softmax(tlogits, -1)
        alpha = jp.exp(log_alpha)
        bootstrap = (1.0 - batch["done"])
        r = batch["reward"] - bootstrap * args.gamma * alpha * nlogp[:, 0]   # soft reward (B,)
        # C51 target (Squint recipe): EACH online Q-net regresses to its OWN target-net Q
        # distribution -- NO clipped-double-Q min/mean over the ensemble.
        tgt = jax.lax.stop_gradient(
            jax.vmap(_project, in_axes=(0, None, None, None))(tprobs, r, bootstrap, args.gamma))  # (Q,B,A)
        feat = enc.apply(enc_p, pix)
        logits = critic.apply(crit_p, feat, s, batch["action"])  # (Q,B,A)
        logp = jax.nn.log_softmax(logits, -1)
        loss = (-(tgt * logp).sum(-1).mean(-1)).sum()           # CE per Q (mean batch), sum over Q
        q = N.expected_q(logits)                                 # (Q,B) online Q estimate
        return loss, {"closs": loss, "q_mean": q.mean(), "q_max": q.max(),
                      "r_mean": batch["reward"].mean(), "tgt_mean": (tgt * N.Q_SUPPORT).sum(-1).mean()}

    def actor_loss_fn(act_p, enc_p, crit_p, norm, log_alpha, batch, key):
        s = _ns(norm, batch["state"])
        # Augment the actor's pixels too (own draw) so the POLICY — not just the encoder — is
        # trained scale/shift-invariant.
        kp, key = jax.random.split(key)
        pix = _augment(kp, batch["pix"], args.aug_shift, args.aug_scale) if AUG else batch["pix"]
        feat = jax.lax.stop_gradient(enc.apply(enc_p, pix))            # detached encoder
        mean, logstd = actor.apply(act_p, feat, s)
        a, logp = N.sample_action(mean, logstd, key)
        q = N.expected_q(critic.apply(crit_p, feat, s, a)).mean(0)     # mean over Q ensemble (no CDQ)
        alpha = jp.exp(log_alpha)
        return (alpha * logp[:, 0] - q).mean(), logp

    def alpha_loss_fn(log_alpha, logp):
        return (-jp.exp(log_alpha) * (logp[:, 0] + target_entropy)).mean()

    @jax.jit
    def update(carry, batch, key):
        enc_p, crit_p, tcrit_p, act_p, log_alpha, cri_os, act_os, alp_os, norm, step = carry
        kc, ka = jax.random.split(key)
        # critic + encoder
        gl = jax.value_and_grad(critic_loss_fn, argnums=(0, 1), has_aux=True)
        (_, diag), (g_enc, g_crit) = gl(enc_p, crit_p, tcrit_p, act_p, norm, log_alpha, batch, kc)
        upd, cri_os = cri_opt.update({"enc": g_enc, "crit": g_crit}, cri_os,
                                     {"enc": enc_p, "crit": crit_p})
        merged = optax.apply_updates({"enc": enc_p, "crit": crit_p}, upd)
        enc_p, crit_p = merged["enc"], merged["crit"]
        # actor (delayed) + alpha
        def do_actor(act_p, log_alpha, act_os, alp_os):
            (al, logp), g_act = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                act_p, enc_p, crit_p, norm, log_alpha, batch, ka)
            u, act_os = act_opt.update(g_act, act_os, act_p); act_p = optax.apply_updates(act_p, u)
            g_alp = jax.grad(alpha_loss_fn)(log_alpha, logp)
            u2, alp_os = alp_opt.update(g_alp, alp_os, log_alpha)
            log_alpha = optax.apply_updates(log_alpha, u2)
            return act_p, log_alpha, act_os, alp_os
        do = (step % args.policy_frequency) == 0
        act_p, log_alpha, act_os, alp_os = jax.lax.cond(
            do, do_actor, lambda *a: (a[0], a[1], a[2], a[3]), act_p, log_alpha, act_os, alp_os)
        # target update (every step)
        tcrit_p = optax.incremental_update(crit_p, tcrit_p, args.tau)
        return (enc_p, crit_p, tcrit_p, act_p, log_alpha, cri_os, act_os, alp_os, norm, step + 1), diag

    @jax.jit
    def policy_action(enc_p, act_p, norm, obs, key):
        feat = enc.apply(enc_p, obs["pixels/view_wrist"])
        mean, logstd = actor.apply(act_p, feat, _ns(norm, obs["state"]))
        a, _ = N.sample_action(mean, logstd, key)
        return a

    @jax.jit
    def eval_action(enc_p, act_p, norm, obs):
        # EVAL = deterministic MODE action (tanh(mean), no exploration noise), same as rollout_diag.
        # The training-rollout success_once running mean uses the NOISY sampled policy, so it reads
        # lower/noisier than this clean held-out measurement.
        feat = enc.apply(enc_p, obs["pixels/view_wrist"])
        mean, _ = actor.apply(act_p, feat, _ns(norm, obs["state"]))
        return jp.tanh(mean)

    # -------- replay buffer (HOST / CPU RAM, fp16 pixels) --------
    # Keeps the full (1M) obs buffer OFF the GPU so it fits ALONGSIDE 1024 envs + Madrona on the
    # 8GB card — a 1M *device* buffer OOMs (1.5GB pix alloc). Cost: the sampled batch is copied
    # host->device per update (np gather -> jp.asarray); fp16 pixels keep that copy small.
    C = args.buffer
    _bdt = {"pix": np.float16, "state": np.float16, "action": np.float16, "reward": np.float32,
            "npix": np.float16, "nstate": np.float16, "done": np.float32}
    _bsh = {"pix": (RES, RES, CH), "state": (nstate,), "action": (NA,), "reward": (),
            "npix": (RES, RES, CH), "nstate": (nstate,), "done": ()}
    hbuf = {kk: np.zeros((C,) + _bsh[kk], _bdt[kk]) for kk in _bdt}
    _np_rng = np.random.default_rng(args.seed)

    def insert(idx, tr):
        n = int(tr["reward"].shape[0]); ii = (idx + np.arange(n)) % C
        for kk in hbuf:
            hbuf[kk][ii] = np.asarray(tr[kk], dtype=_bdt[kk])   # device -> host
        return (idx + n) % C

    def sample(filled):
        ii = _np_rng.integers(0, filled, size=args.batch)
        b = {kk: jp.asarray(hbuf[kk][ii]) for kk in hbuf}        # host -> device
        for kk in ("pix", "state", "action", "npix", "nstate"):
            b[kk] = b[kk].astype(jp.float32)
        return b

    # -------- WARMUP: the first Madrona render must happen inside the jitted rollout, so
    # warm-compile all update/inference kernels BEFORE any render --------
    db = {"pix": jp.zeros((args.batch, RES, RES, CH)), "state": jp.zeros((args.batch, nstate)),
          "action": jp.zeros((args.batch, NA)), "reward": jp.zeros((args.batch,)),
          "npix": jp.zeros((args.batch, RES, RES, CH)), "nstate": jp.zeros((args.batch, nstate)),
          "done": jp.zeros((args.batch,))}
    carry = (enc_p, crit_p, tcrit_p, act_p, log_alpha, cri_os, act_os, alp_os, norm, jp.array(0))
    jax.block_until_ready(update(carry, db, k))
    _dobs = {"pixels/view_wrist": jp.zeros((Nenv, RES, RES, CH)), "state": jp.zeros((Nenv, nstate))}
    jax.block_until_ready(policy_action(enc_p, act_p, norm, _dobs, k))
    jax.block_until_ready(eval_action(enc_p, act_p, norm, _dobs))   # warm the eval mode too
    # (host replay buffer: sample() isn't jitted, so no pre-render compile needed for it.)
    print("[sac] warm-compiled update/policy/eval before first render", flush=True)

    reset = jax.jit(wenv.reset); step_env = jax.jit(wenv.step)

    # -------- held-out deterministic eval --------
    # Roll the MODE policy for one full horizon from a FRESH reset (separate RNG stream, no buffer
    # insert, no grad). Reuses the SAME Nenv envs (no extra GPU alloc -> safe on 8 GB); all envs
    # share a fixed horizon -> exactly one episode each. at_end is snapshot at ep_len-2 to avoid
    # the auto-reset clobber on the terminal step.
    #
    # FULL SUCCESS = lifted & grasped & reached_rest, where reached_rest = ||arm_target - rest_qpos||
    # < 0.2 (gripper joint excluded). env.metrics["success"] already = lifted & grasped; the rest
    # gate is ANDed in. The env-native lift (grasp & cube-above-line, NO rest gate) is reported
    # separately (lift_once/lift_end) so a policy that lifts but drifts off rest is still legible.
    _home_arm = env._home_arm_q                       # 5 arm joints, gripper excluded
    def _full_success(es):
        target_arm = es.data.ctrl[..., :5]            # arm position TARGETS
        reached_rest = (jp.linalg.norm(target_arm - _home_arm, axis=-1) < 0.2).astype(jp.float32)
        return es.metrics["success"] * reached_rest   # lifted & grasped & at-rest
    def run_eval(eval_key):
        es = reset(jax.random.split(eval_key, Nenv))
        s_once = jp.zeros(Nenv); l_once = jp.zeros(Nenv); g_once = jp.zeros(Nenv)
        ret = jp.zeros(Nenv); s_end = jp.zeros(Nenv); l_end = jp.zeros(Nenv)
        for t in range(ep_len):
            a = eval_action(carry[0], carry[3], carry[8], es.obs)
            es = step_env(es, a)
            ph3 = _full_success(es); lift = es.metrics["lifted"]   # full vs env-native (no rest gate)
            s_once = jp.maximum(s_once, ph3)                     # full success_once (incl. at-rest)
            l_once = jp.maximum(l_once, lift)                    # env-native lift_once
            g_once = jp.maximum(g_once, es.metrics["is_grasped"])
            ret = ret + es.reward
            if t == ep_len - 2:
                s_end = ph3; l_end = lift                        # full / env-native at episode end
        return (float(jp.mean(s_once)), float(jp.mean(s_end)), float(jp.mean(g_once)),
                float(jp.mean(ret)), float(jp.mean(l_once)), float(jp.mean(l_end)))

    # -------- training loop --------
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.name)
    os.makedirs(out, exist_ok=True)
    k, kr = jax.random.split(k)
    state = reset(jax.random.split(kr, Nenv))
    idx = 0; env_steps = 0; t0 = time.time(); best = -1.0
    succ_hist = []
    while env_steps < args.total_steps:
        obs = state.obs
        k, kp = jax.random.split(k)
        if env_steps < args.learning_starts:
            action = jax.random.uniform(kp, (Nenv, NA), minval=-1.0, maxval=1.0)
        else:
            action = policy_action(carry[0], carry[3], carry[8], obs, kp)
        nstate_env = step_env(state, action)
        # Store the REAL done. The auto-reset wrapper clobbers nstate_env.obs with the RESET obs on
        # done steps, so the true terminal obs is gone; done=real makes bootstrap=(1-done)=0 there,
        # ZEROING the reset-obs out of the C51 target (Tz=r) instead of bootstrapping a garbage
        # cross-episode Bellman target.
        tr = {"pix": obs["pixels/view_wrist"], "state": obs["state"], "action": action,
              "reward": nstate_env.reward * args.reward_scaling,   # normalized reward fits support [-20,20]
              "npix": nstate_env.obs["pixels/view_wrist"],
              "nstate": nstate_env.obs["state"], "done": nstate_env.done}
        idx = insert(idx, tr)
        # track success_once from the env metrics (rate per episode)
        if "success_once" in nstate_env.metrics:
            succ_hist.append(float(jp.mean(nstate_env.metrics["success_once"])))
        state = nstate_env
        env_steps += Nenv

        diag = None
        if env_steps >= args.learning_starts:
            filled = min(env_steps, C)            # valid transitions stored so far (host buffer)
            for u in range(args.updates_per_step):
                k, ku = jax.random.split(k)
                batch = sample(filled)
                carry, diag = update(carry, batch, ku)

        if env_steps % args.eval_every < Nenv:
            sps = env_steps / (time.time() - t0)
            recent = float(np.mean(succ_hist[-200:])) if succ_hist else 0.0   # noisy train-rollout proxy
            la = float(carry[4])
            d = {kk: round(float(v), 3) for kk, v in diag.items()} if diag else {}
            # Held-out deterministic eval. success_once/at_end = full metric (lift & grasp &
            # at-rest); lift_once/at_end = the env-native lift alone (no rest gate) for context.
            k, kev = jax.random.split(k)
            ev_once, ev_end, ev_grasp, ev_ret, ev_lonce, ev_lend = run_eval(kev)
            print(f"[sac] step {env_steps}  EVAL[full] success_once={ev_once:.3f} "
                  f"success_at_end={ev_end:.3f}  (lift_once={ev_lonce:.3f} lift_end={ev_lend:.3f}) "
                  f"grasp_once={ev_grasp:.3f} return={ev_ret:.2f}  | train_succ_once~{recent:.3f} "
                  f"alpha={np.exp(la):.3f}  {sps:.0f} sps  {d}", flush=True)
            # CSV log: success_* = full (rest-gated) metric; lift_* = env-native.
            csv_path = f"{out}/eval_metrics.csv"
            if not os.path.exists(csv_path):
                with open(csv_path, "w") as f:
                    f.write("env_steps,success_once,success_at_end,lift_once,lift_at_end,"
                            "grasp_once,return,train_succ_once,alpha,sps\n")
            with open(csv_path, "a") as f:
                f.write(f"{env_steps},{ev_once:.4f},{ev_end:.4f},{ev_lonce:.4f},{ev_lend:.4f},"
                        f"{ev_grasp:.4f},{ev_ret:.4f},{recent:.4f},{np.exp(la):.4f},{sps:.1f}\n")
            params = (carry[8], carry[0], carry[3])    # (norm, encoder, actor) for deploy
            # Always write the latest (the deploy recipe points at policy_params.pkl) AND keep a
            # best-by-eval checkpoint. Best gates on LIFT_END (cube grasped+lifted at the LAST step,
            # env-native, no rest gate) = the deploy-relevant "held the lift" metric; the rest-gated
            # success can sit at 0 for a policy that lifts but never returns to home.
            with open(f"{out}/policy_params.pkl", "wb") as f:
                pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
            if ev_lend > best:
                best = ev_lend
                with open(f"{out}/policy_best.pkl", "wb") as f:
                    pickle.dump(jax.tree_util.tree_map(np.asarray, params), f)
                print(f"[sac]   ^ new best lift_end={best:.3f} -> policy_best.pkl", flush=True)
    print(f"[sac] DONE in {time.time()-t0:.0f}s  best_eval_lift_end={best:.3f}  -> "
          f"{out}/policy_params.pkl (+ policy_best.pkl)", flush=True)


if __name__ == "__main__":
    main()
