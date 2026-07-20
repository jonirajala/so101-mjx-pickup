"""SO101PickCubeVision — Madrona-rendered RGB vision env (single or dual camera) with
domain randomization, consumed by squint/train_sac.py.

Subclasses the state env (`SO101PickCube`) and adds batched Madrona rendering in
reset/step. Observation dict:

  obs = {
    'state'            : proprio — the full state minus the 3 gripper-relative cube
                          dims (the policy must infer the cube position from pixels),
    'privileged_state' : the full state (for the critic),
    'pixels/view_wrist': (16,16,3) RGB in [0,1] (single cam) or (16,16,6)
                          channel-concat [overhead(0:3), wrist(3:6)] (dual cam).
  }

Wire with `mujoco_playground.wrapper.wrap_for_brax_training(env, vision=True,
num_vision_envs=N, randomization_fn=vision_randomization_fn(N))`.
"""
import os, sys
import numpy as np
import jax
import jax.numpy as jp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from so101_pick_cube import SO101PickCube, default_config
import vision_dr
from randomize import domain_randomize

# The gripper-relative cube position is dims [21,22,23] of the state (see
# SO101PickCube._get_obs); everything else is pixel-inferable proprio. The jaw-load dim,
# when present, is KEPT (it is observable on the real arm); the proprio index set is
# computed from the env's actual _state_dim below, not hardcoded.
_CUBE_REL = (21, 22, 23)
# Pixel-obs resolution. Madrona renders at render_width (64) but the policy sees a 16x16
# AREA-downsample of it (Squint, arXiv 2602.21203: render high -> area-resize -> 16).
# Low-res (a) washes out the sharp synthetic render signature that a higher-res CNN
# overfits to and that is absent in a real photo, and (b) forces a compressed,
# task-aligned feature instead of pixel memorization.
_OBS_RES = 16
# PER-STEP camera-pose DR: a FRESH pose offset is re-sampled EVERY control step around
# each camera's base/FK world pose. With a static overhead the policy integrates it into
# a precise position estimate and over-trusts it — a slightly-off real overhead then
# drives lateral hunting at deploy. Per-step parallax destroys the overhead's metric
# reliability, so the policy must close the terminal grasp with the stable wrist cam.
_OVH_STEP_DPOS = 0.03            # overhead eye-shift box: uniform(-1,1)*0.03 = ±3 cm/axis
_OVH_STEP_DROT = np.deg2rad(4)   # overhead ROLL about the view axis (aim-preserving)
_OVH_TARGET = np.array([0.28, 0.0, 0.0])   # overhead look-at target (workspace centre, m)
_OVH_LOOKAT_NOISE = 0.01        # look-at target jitter std (m)
_WRI_STEP_DPOS = 0.002           # wrist ±2 mm
_WRI_STEP_DROT = np.deg2rad(1)   # wrist ±1°


def _jitter_cam_pose(data, cam_id, dpos, drot, key):
    """Re-sample a fresh per-step pose offset around a camera's current (base/FK) world
    pose. Perturbs the world-frame camera transform the Madrona renderer reads —
    data.cam_xpos (position parallax) + data.cam_xmat (small rotation via a Rodrigues
    small-angle rotation). Position is the dominant de-reliance lever."""
    kp, kr = jax.random.split(key)
    pos = data.cam_xpos[cam_id] + dpos * jax.random.uniform(kp, (3,), minval=-1.0, maxval=1.0)
    w = drot * jax.random.uniform(kr, (3,), minval=-1.0, maxval=1.0)   # rotation vector (rad)
    ang = jp.linalg.norm(w) + 1e-12
    ax = w / ang
    K = jp.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
    R = jp.eye(3) + jp.sin(ang) * K + (1.0 - jp.cos(ang)) * (K @ K)    # Rodrigues
    orig = data.cam_xmat[cam_id]                                       # (3,3) or (9,)
    mat = (R @ orig.reshape(3, 3)).reshape(orig.shape)
    return data.replace(cam_xpos=data.cam_xpos.at[cam_id].set(pos),
                        cam_xmat=data.cam_xmat.at[cam_id].set(mat))


def _jitter_overhead_lookat(data, cam_id, base_pos, target, dpos, look_at_noise, roll_noise, key):
    """AIM-PRESERVING overhead jitter: (1) shift the EYE in a ±dpos box, (2) RE-AIM the
    camera at the target (itself jittered by N(0,look_at_noise)) so the frame centre
    stays LOCKED on the workspace, and (3) roll ONLY about the view axis by
    N(0,roll_noise). A general ±4° rotation would instead wobble the AIM (~±3.5 cm
    view-shift at 0.5 m) every step, making the overhead an unreliable off-centre
    localiser that the policy learns to distrust. cam_xmat cols = [x_right, y_up,
    z_back]; the MuJoCo camera looks along -z, verified to reproduce the XML overhead
    orientation for the base pose."""
    kp, kt, kr = jax.random.split(key, 3)
    eye = base_pos + dpos * jax.random.uniform(kp, (3,), minval=-1.0, maxval=1.0)   # ±dpos box (prism)
    tgt = target + look_at_noise * jax.random.normal(kt, (3,))                       # N(0,look_at_noise)
    f = tgt - eye
    f = f / (jp.linalg.norm(f) + 1e-9)                # view direction (camera looks along -z = f)
    z = -f
    up = jp.array([0.0, 0.0, 1.0])
    x = jp.cross(up, z); x = x / (jp.linalg.norm(x) + 1e-9)
    y = jp.cross(z, x)
    mat = jp.stack([x, y, z], axis=1)                 # cols = camera axes in world (camera->world)
    ang = roll_noise * jax.random.normal(kr)          # ROLL about the view axis f
    K = jp.array([[0.0, -f[2], f[1]], [f[2], 0.0, -f[0]], [-f[1], f[0], 0.0]])
    Rroll = jp.eye(3) + jp.sin(ang) * K + (1.0 - jp.cos(ang)) * (K @ K)
    mat = (Rroll @ mat).reshape(data.cam_xmat[cam_id].shape)
    return data.replace(cam_xpos=data.cam_xpos.at[cam_id].set(eye),
                        cam_xmat=data.cam_xmat.at[cam_id].set(mat))


def _linear_to_srgb(x):
    """Encode Madrona's LINEAR raytracer output to sRGB, matched to a real webcam's sRGB
    response.

    Madrona's raytracer writes RAW linear RGB (no gamma/tonemap), while a real webcam
    ISP (e.g. the OV2710) outputs gamma-encoded sRGB. Training on linear but deploying
    on sRGB frames leaves a systematic dark/low-contrast offset that brightness DR
    scatters around but cannot remove. A plain pow(1/2.2) is used (not the piecewise
    sRGB curve), applied BEFORE the area-downsample so downsample + ColorJitter run in
    sRGB space and the deploy path needs NO change (the real frame is already sRGB).
    x: (H,W,3) float 0-255 -> 0-255."""
    return jp.power(jp.clip(x / 255.0, 0.0, 1.0), 1.0 / 2.2) * 255.0


def _blur5(x):
    """Light 5-tap (plus) blur on an (H,W,3) image — softens the crisp raytraced edges
    that are absent in a real photo. Edge-padded."""
    up = jp.pad(x[:-1], ((1, 0), (0, 0), (0, 0)), mode="edge")
    dn = jp.pad(x[1:], ((0, 1), (0, 0), (0, 0)), mode="edge")
    lf = jp.pad(x[:, :-1], ((0, 0), (1, 0), (0, 0)), mode="edge")
    rt = jp.pad(x[:, 1:], ((0, 0), (0, 1), (0, 0)), mode="edge")
    return (x + up + dn + lf + rt) / 5.0


class SO101PickCubeVision(SO101PickCube):
    def __init__(self, num_envs, render_width=64, render_height=64, vision=True,
                 rgb=True, compute_reward=False, dual_cam=False, jitter_kwargs=None,
                 overhead_pan=0.0, overhead_dropout=0.0,
                 config=None, config_overrides=None):
        super().__init__(config or default_config(), config_overrides)
        self._vision = vision
        # Dual-cam-only extras: per-step image pan + overhead channel dropout (default off).
        self._overhead_pan = float(overhead_pan)
        self._overhead_dropout = float(overhead_dropout)
        # Photometric DR ranges passed to vision_dr.rgb_brightness; None = its (wide)
        # defaults. The SAC trainer passes a narrower ColorJitter (hue ±0.05).
        self._jitter_kwargs = jitter_kwargs or {}
        # dual_cam: render BOTH cams and channel-concat [overhead(0:3), wrist(3:6)] into
        # a 6-ch 16x16 obs. RGB-path only.
        self._dual_cam = bool(dual_cam)
        self._pix_channels = 6 if self._dual_cam else 3
        self._obs_res = _OBS_RES
        # The SAC trainer (squint/train_sac.py) needs the dense reward -> pass True;
        # obs-only rollouts can skip the 7-term reward with False.
        self._compute_reward = compute_reward
        # Drop the 3 gripper-relative cube dims (inferred from pixels); KEEP everything
        # else, incl. the jaw-load dim. Derived from the env's real _state_dim so the
        # index set is correct whether or not _OBS_JAW_LOAD is on.
        self._proprio_idx = jp.array([i for i in range(self._state_dim) if i not in _CUBE_REL])
        self._wrist_cam = self._mj_model.camera("wrist").id
        # Scene ids for the PER-EPISODE render DR (vision_dr.sample_render_dr, called in
        # reset()), resolved by NAME exactly like vision_randomization_fn so an XML edit
        # can't mis-target the DR.
        import mujoco
        mjm = self._mj_model
        _geom_names = {mjm.geom(i).name for i in range(mjm.ngeom)}
        _cam_names = {mjm.camera(i).name for i in range(mjm.ncam)}
        self._box_geom = mjm.geom("box").id
        self._floor_geom = mjm.geom("floor").id
        self._table_geom = mjm.geom("table").id if "table" in _geom_names else None
        self._tabletop_geom = mjm.geom("tabletop_tex").id if "tabletop_tex" in _geom_names else None
        self._wood_matid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "wood052")
        self._wall_geoms = jp.array([], dtype=jp.int32)   # no backdrop walls in this scene
        self._prong_geoms = jp.array([mjm.geom("fixed_prong_visual").id,
                                      mjm.geom("moving_prong_visual").id])
        self._overhead_cam = mjm.camera("overhead").id if "overhead" in _cam_names else None
        # ORIGINAL (unjittered) base cam_fovy. sample_render_dr ADDS its ±deg FOV jitter
        # to the cam_fovy of the model it is handed; in reset() self._mjx_model already
        # carries the randomization_fn's per-env cam_fovy jitter, so feed this base to
        # avoid double-jitter (the other DR'd fields SET their indices, so they are
        # base-independent).
        self._base_cam_fovy = jp.asarray(self._mjx_model.cam_fovy)
        # Dual-cam render order = [overhead, wrist] so rgb[0]=overhead, rgb[1]=wrist.
        # overhead is declared FIRST in the XML (id 0) so this holds whether Madrona
        # preserves the enabled_cameras order or sorts by id.
        if self._dual_cam:
            _cams = np.array([self._overhead_cam, self._wrist_cam])
        else:
            _cams = np.array([self._wrist_cam])
        # Per-step camera-pose DR; on for RGB vision.
        self._cam_step_jitter = bool(vision) and bool(rgb)
        self._rw, self._rh = render_width, render_height
        if vision:
            from madrona_mjx.renderer import BatchRenderer
            self.renderer = BatchRenderer(
                self._mjx_model, gpu_id=0, num_worlds=num_envs,
                batch_render_view_width=render_width,
                batch_render_view_height=render_height,
                enabled_geom_groups=np.array([0, 1, 2]),
                # Wrist cam alone (single) or [overhead, wrist] (dual); debug_ext is tuning-only.
                enabled_cameras=_cams,
                add_cam_debug_geo=False, use_rasterizer=False, viz_gpu_hdls=None,
            )

    # ---- vision obs assembly ----
    def _process_cam(self, img4, rng, fov_key, exp_key, fov_max_deg, fov_deg, pan_max=0.0):
        """One rendered camera -> (16,16,3) sRGB float in [0,1]: linear->sRGB, optional
        per-step pan, random blur, area-downsample to 16, ColorJitter. img4: (rw,rw,4)
        Madrona LINEAR RGBA 0-255. FOV DR is geometric per-world (cam_fovy, drawn in
        vision_dr and baked by the renderer), so fov_key/fov_max_deg/fov_deg are unused
        — kept for call-site compatibility."""
        kb, kj, kp = jax.random.split(rng, 3)
        x = img4[..., :3].astype(jp.float32)                 # LINEAR RGB 0-255
        x = _linear_to_srgb(x)                               # -> sRGB (match the real cam)
        if pan_max > 0.0:
            x = vision_dr.img_pan(x, kp, max_frac=pan_max)   # per-step image pan (default off)
        b = jax.random.uniform(kb, (), minval=0.0, maxval=1.0)
        x = (1.0 - b) * x + b * _blur5(x)                    # random light blur
        f = self._rw // _OBS_RES                              # area-downsample rw->16
        x = x.reshape(_OBS_RES, f, _OBS_RES, f, 3).mean((1, 3))
        x = vision_dr.rgb_brightness(x, kj, exp_rng=exp_key, **self._jitter_kwargs)  # per-step ColorJitter
        return x.astype(jp.float32) / 255.0                  # (16,16,3) in [0,1]

    def _pixels(self, state, rgb, depth):
        """Build the (H,W,C) CNN input from the renderer's batched rgb/depth outputs.
        rgb:   (n_enabled_cams, H, W, 4) uint8 RGBA;  depth: (n_enabled_cams, H, W, 1) metres.
        Single-cam: only the wrist is enabled (axis 0). Dual-cam: [overhead, wrist] (axes 0,1)."""
        if self._dual_cam:
            # 6-ch concat [overhead(0:3), wrist(3:6)], independent DR per camera. The
            # per-step 3D extrinsic parallax in step() is the overhead's robustness DR;
            # image pan and dropout are optional extras (default 0).
            rng, r0, r1, rd = jax.random.split(state.info["rng"], 4)
            state.info["rng"] = rng
            ef0, ef1, ex0, ex1 = jax.random.split(state.info["ep_rng"], 4)  # per-EPISODE keys
            ovh = self._process_cam(rgb[0], r0, ef0, ex0, fov_max_deg=9.0, fov_deg=60.0, pan_max=self._overhead_pan)
            wri = self._process_cam(rgb[1], r1, ef1, ex1, fov_max_deg=1.0, fov_deg=80.0)
            if self._overhead_dropout > 0.0:
                keep = (jax.random.uniform(rd, ()) >= self._overhead_dropout).astype(ovh.dtype)
                ovh = ovh * keep                              # blank overhead -> wrist must carry the grasp
            return jp.concatenate([ovh, wri], axis=-1)       # (16,16,6) in [0,1]
        rng, k, kb = jax.random.split(state.info["rng"], 3)
        state.info["rng"] = rng
        x = rgb[0, ..., :3].astype(jp.float32)           # (rw,rw,3) Madrona LINEAR RGB 0-255
        x = _linear_to_srgb(x)                           # -> sRGB (match the real cam)
        exk = jax.random.fold_in(state.info["ep_rng"], 1) # per-EPISODE key
        b = jax.random.uniform(kb, (), minval=0.0, maxval=1.0)
        x = (1.0 - b) * x + b * _blur5(x)                # random light blur
        f = self._rw // _OBS_RES                          # area-downsample rw->16
        x = x.reshape(_OBS_RES, f, _OBS_RES, f, 3).mean((1, 3))
        x = vision_dr.rgb_brightness(x, k, exp_rng=exk, **self._jitter_kwargs)  # per-step ColorJitter
        return x.astype(jp.float32) / 255.0              # (16,16,3) in [0,1]

    def _obs(self, state, pixels):
        full_state = state.obs["state"]                         # the env's full state vector
        return {
            "state": full_state[self._proprio_idx],             # proprio (minus the 3 cube-relative dims)
            "privileged_state": full_state,                     # full state, for the critic
            "pixels/view_wrist": pixels,
        }

    def reset(self, rng):
        state = super().reset(rng)
        if not self._vision:
            return state
        # DO NOT render here. The first Madrona render in a process must happen INSIDE
        # the jitted rollout (step), never before it: after any render, XLA can no longer
        # LOAD a newly-compiled CUDA module (cuModuleLoadData -> cudaErrorInvalidValue),
        # so a setup render here would make every later-compiled kernel fail. The initial
        # obs therefore carries a blank frame (one step); step() renders from t=1 on.
        state.info["render_token"] = jp.zeros((), dtype=jp.bool)  # scalar, matches init()'s token
        # A PER-EPISODE key, fixed for the whole episode (step() never touches it), so
        # per-episode DR is sampled once per episode.
        state.info["ep_rng"] = jax.random.fold_in(state.info["rng"], 0x5217)
        # Render-DR timing: cube colour, table colour/texture, LIGHTING and camera FOV
        # are drawn ONCE PER ENV (in vision_randomization_fn -> self._mjx_model) and NOT
        # re-rolled per episode; ONLY the off-table/FLOOR colour is per-episode.
        # Re-rolling everything per episode over-randomizes — the per-episode FOV re-roll
        # in particular blurs the overhead depth cue and hurts off-centre localization.
        dr_model = self._mjx_model.tree_replace({"cam_fovy": self._base_cam_fovy})
        vdr = vision_dr.sample_render_dr(
            dr_model, state.info["ep_rng"], box_geom=self._box_geom,
            table_geom=self._table_geom, tabletop_geom=self._tabletop_geom,
            wood_matid=self._wood_matid, floor_geom=self._floor_geom,
            wall_geoms=self._wall_geoms, prong_geoms=self._prong_geoms,
            wrist_cam=self._wrist_cam, overhead_cam=self._overhead_cam)
        state.info["vdr_floor_rgba"] = vdr["geom_rgba"][self._floor_geom]   # per-episode FLOOR colour only
        blank = jp.zeros((self._obs_res, self._obs_res, self._pix_channels))
        return state.replace(obs=self._obs(state, blank))

    def step(self, state, action):
        if not self._vision:
            return super().step(state, action)
        state = super().step(state, action)
        # PER-STEP camera-pose DR on a RENDER-ONLY copy of data (physics/proprio use the
        # unjittered state.data). Overhead parallax forces the terminal grasp onto the wrist.
        render_data = state.data
        if self._cam_step_jitter:
            rng, kj = jax.random.split(state.info["rng"]); state.info["rng"] = rng
            kwr, kov = jax.random.split(kj)
            render_data = _jitter_cam_pose(render_data, self._wrist_cam, _WRI_STEP_DPOS, _WRI_STEP_DROT, kwr)
            if self._dual_cam:
                # AIM-PRESERVING jitter (eye shift + re-aim at target + roll), NOT a
                # general rotation, so the overhead stays a reliable off-centre
                # localiser. The base overhead pose is fixed, so cam_xpos[overhead] is
                # the unjittered eye.
                render_data = _jitter_overhead_lookat(
                    render_data, self._overhead_cam, render_data.cam_xpos[self._overhead_cam],
                    jp.asarray(_OVH_TARGET), _OVH_STEP_DPOS, _OVH_LOOKAT_NOISE, _OVH_STEP_DROT, kov)
        # Per-episode render DR: override ONLY the FLOOR colour; cube colour, table
        # colour/texture, lighting and camera FOV stay at their PER-ENV-FIXED
        # randomization_fn values in self._mjx_model. The floor matid is already -2
        # (per-env), so only rgba changes.
        render_model = self._mjx_model.tree_replace({
            "geom_rgba": self._mjx_model.geom_rgba.at[self._floor_geom].set(state.info["vdr_floor_rgba"]),
        })
        # Use init() (not render()) so the very first render lives inside this jitted
        # step — i.e. the executable's module is loaded BEFORE any render runs. init()
        # re-establishes the scene each frame and returns rgb+depth identical to
        # render(), in one batched render pass. (render() would need a token from a
        # prior, pre-scan init() — exactly the render to avoid.)
        token, rgb, depth = self.renderer.init(render_data, render_model)
        state.info["render_token"] = token
        return state.replace(obs=self._obs(state, self._pixels(state, rgb, depth)))


def vision_randomization_fn(env, num_envs, seed=0):
    """randomization_fn for wrap_for_brax_training(vision=True): per-world dynamics DR
    (kp/damping/friction/mass) applied in sequence with the visual DR.
    _supplement_vision_randomization_fn tiles the remaining render fields (lights).

    Scene ids are resolved by NAME from the env's mj_model (not hardcoded), so XML edits
    can't silently mis-target the DR. Both randomizers vmap over a *batch* of num_envs
    keys and MUST touch disjoint model fields, so applying them in sequence is safe;
    in_axes is the union of both."""
    # Union of the fields the two randomizers batch (None-vs-0 tree merges are brittle).
    # light_pos/light_dir are batched here so the per-world light DR is read consistently —
    # MadronaWrapper._supplement tiles the remaining light fields (type/castshadow/cutoff).
    batched = ["actuator_gainprm", "actuator_biasprm", "dof_damping", "dof_frictionloss",
               "dof_armature", "geom_friction", "body_mass",            # dynamics DR
               "geom_rgba", "geom_matid", "geom_size", "cam_pos", "cam_quat",  # vision DR
               "light_pos", "light_dir", "cam_fovy"]     # light-direction DR + geometric FOV DR
    import mujoco
    mj = env.mj_model
    box_body, box_geom = mj.body("box").id, mj.geom("box").id
    wrist_cam = mj.camera("wrist").id
    cam_names = {mj.camera(i).name for i in range(mj.ncam)}
    overhead_cam = mj.camera("overhead").id if "overhead" in cam_names else None  # dual-cam pose DR
    geom_names = {mj.geom(i).name for i in range(mj.ngeom)}
    table_geom = mj.geom("table").id if "table" in geom_names else None  # the surface the cube sits on
    floor_geom = mj.geom("floor").id
    # No backdrop walls; the off-table surround is just the floor.
    wall_geoms = jp.array([], dtype=jp.int32)
    prong_geoms = jp.array([mj.geom("fixed_prong_visual").id, mj.geom("moving_prong_visual").id])
    # Floor/table texture candidates (matid randomized per world; resolved by name).
    # Madrona has no normal/roughness maps — only the albedo texture renders — so the
    # split is ⅔ flat / ⅓ textured; geom_rgba tints whichever is picked. (The texture
    # image is a cardboard photo, matching the real deploy table; at 16x16 the exact
    # grain is a sub-pixel residual.)
    floor_matids = jp.array([mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_MATERIAL, n)
                             for n in ("floor_flat", "floor_flat", "floor_cardboard")])
    # The visual quad `tabletop_tex` (a mesh, so it has UVs; the table BOX has none and
    # can't texture) carries the Wood052 grain; vision_dr DRs its matid per-world
    # (~⅓ wood / ~⅔ flat).
    tabletop_geom = mj.geom("tabletop_tex").id if "tabletop_tex" in geom_names else None
    wood_matid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_MATERIAL, "wood052")

    def fn(model):
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
        model, _ = domain_randomize(model, jax.random.split(k1, num_envs),
                                    box_body=box_body, box_geom=box_geom)
        model, _ = vision_dr.vision_randomize(model, jax.random.split(k2, num_envs),
                                              box_geom=box_geom, wrist_cam=wrist_cam,
                                              overhead_cam=overhead_cam, table_geom=table_geom,
                                              floor_geom=floor_geom, wall_geoms=wall_geoms,
                                              floor_matids=floor_matids, prong_geoms=prong_geoms,
                                              tabletop_geom=tabletop_geom, wood_matid=wood_matid)
        in_axes = jax.tree_util.tree_map(lambda x: None, model)
        in_axes = in_axes.tree_replace({f: 0 for f in batched})
        return model, in_axes
    return fn
