"""Visual domain randomization for the Madrona RGB vision env (envs/so101_vision.py).

Two layers:

1. `vision_randomize(model, rng)` — per-world *model* fields that Madrona tiles across
   worlds: cube colour/material, table colour + texture, off-table colour, light
   direction + diffuse/ambient, camera FOV. Same (batched_model, in_axes) contract as
   `randomize.domain_randomize`, so the two apply in sequence (see
   so101_vision.vision_randomization_fn).

2. `rgb_brightness` — per-step image-space photometric aug (torchvision-style
   ColorJitter) applied to the rendered obs.
"""
import jax
import jax.numpy as jp

# Default scene indices (current model) — fallbacks only. Pass explicit ids resolved by
# name (the env does `mj.geom("box").id` / `mj.camera("wrist").id`) so a future
# geom/camera insertion can't silently mis-target the DR.
_BOX_GEOM = 32          # the cube geom
_WRIST_CAM = 1          # cameras: 0=debug_ext, 1=wrist

# Box half-extents band — ANISOTROPIC (independent x/y/z) so the DR spans cubes AND the
# real deploy object, a rectangular 5x4x2.5 cm box (half-extents [25, 20, 12.5] mm).
# Each axis is drawn independently in [11, 27] mm, so the real box is in-distribution
# for ANY face-down orientation, and 15 mm cubes still are. Max ~54 mm edge stays
# graspable (gripper opens to ~89 mm inner gap, measured). NOTE the box spawns
# axis-aligned (no yaw DR) — a rectangular object must be placed so a graspable
# dimension lines up with the jaw; add box-yaw DR + box dims in the privileged obs if
# alignment proves to matter.
_HALF_LO = jp.array([0.011, 0.011, 0.011])
_HALF_HI = jp.array([0.027, 0.027, 0.027])
# Wrist-camera mount jitter — deliberately wider than the per-step ±2 mm/±1° pose
# jitter, to absorb residual wrist-cam extrinsic/FOV sim2real error while staying modest
# enough that the task remains learnable.
_CAM_DPOS = 0.005       # ~±5 mm
_CAM_DROT = 0.087       # rad; 0.5x per axis => ~±2.5°
# Overhead-camera mount jitter — much wider than the wrist: it is a hand-placed STATIC
# cam, so DR absorbs the rough-mount slop. The matching FOV DR (±9°) is drawn in
# sample_render_dr.
_CAM_OVH_DPOS = 0.06    # ±6 cm per axis
_CAM_OVH_DROT = 0.0698  # deg2rad(4)
# Prong (fixed + moving jaw) VISUAL geoms — the dominant always-in-frame anchor in the
# 16x16 wrist view. Resolved by NAME in the env; these are fallbacks.
_PRONG_GEOMS = (33, 34)


# Backdrop walls + floor-material candidates are resolved by NAME in the env (fallbacks here).
_FLOOR_GEOM = 31
_WALL_GEOMS = (32, 33, 34, 35)


def _hsv_to_rgb(h, s, v):
    """Scalar HSV->RGB in JAX (matches colorsys.hsv_to_rgb), h,s,v in [0,1]."""
    i = jp.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    ii = (i.astype(jp.int32)) % 6
    r = jp.select([ii == 0, ii == 1, ii == 2, ii == 3, ii == 4, ii == 5], [v, q, p, p, t, v])
    g = jp.select([ii == 0, ii == 1, ii == 2, ii == 3, ii == 4, ii == 5], [t, v, v, q, p, p])
    b = jp.select([ii == 0, ii == 1, ii == 2, ii == 3, ii == 4, ii == 5], [p, p, t, v, v, q])
    return jp.array([r, g, b])


def _table_rgb(rng):
    """Table-surface colour family. The real deploy surface is a warm-tan cardboard
    sheet, so the colour is drawn from a family centred on it — NOT a flat U(0.12,0.88)
    RGB, which would make the real tan an unlikely draw (OOD).
      fam<0.70 cardboard/wood: hue U(.05,.11) sat U(.25,.55) val U(.45,.80)
      fam<0.90 neutral:        hue U(0,1)     sat U(0,.08)   val U(.15,.90)
      else     muted olive:    hue U(.20,.40) sat U(.10,.30) val U(.25,.65)"""
    kf, kh, ks, kv = jax.random.split(rng, 4)
    fam = jax.random.uniform(kf)
    card = fam < 0.70
    neut = (fam >= 0.70) & (fam < 0.90)
    hlo = jp.where(card, 0.05, jp.where(neut, 0.0, 0.20))
    hhi = jp.where(card, 0.11, jp.where(neut, 1.0, 0.40))
    slo = jp.where(card, 0.25, jp.where(neut, 0.0, 0.10))
    shi = jp.where(card, 0.55, jp.where(neut, 0.08, 0.30))
    vlo = jp.where(card, 0.45, jp.where(neut, 0.15, 0.25))
    vhi = jp.where(card, 0.80, jp.where(neut, 0.90, 0.65))
    h = jax.random.uniform(kh, minval=hlo, maxval=hhi)
    s = jax.random.uniform(ks, minval=slo, maxval=shi)
    v = jax.random.uniform(kv, minval=vlo, maxval=vhi)
    return _hsv_to_rgb(h, s, v)


def img_pan(img, rng, max_frac=0.10):
    """Image-space PER-STEP pan (translation), edge-clamped. Approximates the dominant
    visual effect of a ±6 cm lateral eye shift at the ~0.53 m overhead view distance
    (fovy 60): a ~±0.10-of-frame pan. Keeps the policy from building a precise, static
    overhead expectation it would over-trust at deploy. img: (H,W,C) float."""
    H, W = img.shape[0], img.shape[1]
    d = jax.random.uniform(rng, (2,), minval=-max_frac, maxval=max_frac)  # (dy,dx) as frame fractions
    yy, xx = jp.meshgrid(jp.arange(H), jp.arange(W), indexing="ij")
    sy = yy - d[0] * H
    sx = xx - d[1] * W
    chans = [jax.scipy.ndimage.map_coordinates(img[..., k], [sy, sx], order=1, mode="nearest")
             for k in range(img.shape[-1])]
    return jp.stack(chans, -1)


def sample_render_dr(model, rng, box_geom=_BOX_GEOM, table_geom=None, tabletop_geom=None,
                     wood_matid=-1, floor_geom=_FLOOR_GEOM, wall_geoms=_WALL_GEOMS,
                     prong_geoms=_PRONG_GEOMS, wrist_cam=_WRIST_CAM, overhead_cam=None):
    """Single-rng (NOT vmapped) draw of the render-DR model fields, returned as a dict
    of 6 fields. cam_pos/cam_quat/light_pos are left at their base values and NOT drawn
    here — `vision_randomize` re-adds them for its batched (randomization_fn) contract.

    `prong_geoms` is accepted for signature compatibility but unused.

    NOTE the cam_fovy `.add()` below jitters the cam_fovy of the PASSED model. For the
    per-episode render path the caller must pass a model whose cam_fovy is the ORIGINAL
    (unjittered) base, or the jitter double-applies on top of the randomization_fn's
    per-env cam_fovy jitter.
    """
    nlight = model.light_pos.shape[0]
    wall_geoms = jp.asarray(wall_geoms)
    del prong_geoms  # unused; kept in signature for compatibility

    # --- Cube COLOUR: 60% fully random U(0,1), with the real rig colours GUARANTEED
    # (20% red, 20% purple) so the deploy cube is in-distribution and high-contrast — a
    # flat uniform draw makes red/purple unlikely and favours mid-grey cubes that vanish
    # on the tan table. (Box SIZE is owned by randomize.domain_randomize, which runs
    # FIRST and batches geom_size; the two randomizers MUST touch disjoint fields or the
    # second mis-indexes the first's batched arrays.) ---
    rng, k, ksel = jax.random.split(rng, 3)
    col = jax.random.uniform(k, (3,), minval=0.0, maxval=1.0)
    sel = jax.random.uniform(ksel)
    col = jp.where(sel < 0.20, jp.array([1.0, 0.0, 0.0]), col)                     # red
    col = jp.where((sel >= 0.20) & (sel < 0.40), jp.array([0.5, 0.0, 0.5]), col)   # purple
    geom_rgba = model.geom_rgba.at[box_geom, :3].set(col)

    # --- TABLE surface (the surface the cube sits on) colour + texture: colour from the
    # cardboard/neutral/olive family (_table_rgb); matid picks grain texture vs flat;
    # geom_rgba tints the choice. Falls back to floor_geom if no table geom is present
    # (single-plane scene). ---
    surf_geom = floor_geom if table_geom is None else table_geom
    rng, kc, km = jax.random.split(rng, 3)
    table_rgb = _table_rgb(kc)
    geom_rgba = geom_rgba.at[surf_geom, :3].set(table_rgb)
    # The table BOX is always flat (matid=-2 so the per-world colour renders). It owns
    # the collision, the box sides and the flat colour modes; the VISUAL tabletop QUAD
    # on top carries the texture when picked (below).
    geom_matid = model.geom_matid.at[surf_geom].set(-2)
    # Tabletop texture split: Madrona has no normal/roughness maps — only the albedo
    # texture renders — so ~⅔ of worlds are flat and ~⅓ (U >= 0.66) show the Wood052
    # albedo (sampled raw/linearized). Flat -> quad matid=-2 with the SAME table_rgb as
    # the box, so the tabletop is a seamless flat surface.
    if tabletop_geom is not None:
        textured = jax.random.uniform(km) >= 0.66
        geom_rgba = geom_rgba.at[tabletop_geom, :3].set(table_rgb)
        geom_matid = geom_matid.at[tabletop_geom].set(jp.where(textured, wood_matid, -2))
    else:
        del km  # legacy scene without the quad
    # Madrona applies the per-world geom_rgba colour override ONLY when matid == -2
    # (UseOverrideColor). The box ships with matid == -1 (no material), under which the
    # colour DR is silently ignored (every world renders the base cube colour) — set
    # matid=-2 so the sampled cube colour actually renders.
    geom_matid = geom_matid.at[box_geom].set(-2)

    # --- OFF-TABLE surround (the ground beyond the table edge + any backdrop walls):
    # muted and generally DARKER than the lit table — hue U(0,1), sat U(0,0.40), val
    # U(0.05,0.70) — and DISTINCT from the table so the overhead sees a clear
    # table-edge / off-table boundary. ---
    rng, kh, ks, kv = jax.random.split(rng, 4)
    off_geoms = jp.concatenate([jp.array([floor_geom]), wall_geoms]) if table_geom is not None else wall_geoms
    nw = off_geoms.shape[0]
    wh = jax.random.uniform(kh, (nw,), minval=0.0, maxval=1.0)
    ws = jax.random.uniform(ks, (nw,), minval=0.0, maxval=0.40)
    wv = jax.random.uniform(kv, (nw,), minval=0.05, maxval=0.70)
    off_rgb = jax.vmap(_hsv_to_rgb)(wh, ws, wv)
    geom_rgba = geom_rgba.at[off_geoms, :3].set(off_rgb)
    # matid=-2 here too: with a material assigned, the per-world rgba override does not
    # render and the off-table colour washes out near-white instead of showing a
    # distinct flat band.
    geom_matid = geom_matid.at[off_geoms].set(-2)

    # --- LIGHT direction: each light gets an INDEPENDENT random azimuth (full 0..2pi)
    # and obliqueness, from above. dir = [cos(az)*horiz, sin(az)*horiz, -1]; horiz in
    # [0.2,1.1] (0 = straight down, larger = more oblique). ---
    rng, ka, kh = jax.random.split(rng, 3)
    az = jax.random.uniform(ka, (nlight,), minval=0.0, maxval=2.0 * jp.pi)
    horiz = jax.random.uniform(kh, (nlight,), minval=0.2, maxval=1.1)
    d = jp.stack([jp.cos(az) * horiz, jp.sin(az) * horiz, -jp.ones((nlight,))], axis=-1)
    light_dir = d / jp.linalg.norm(d, axis=-1, keepdims=True)

    # --- IN-RENDER LIGHTING DR:
    #   level = U(0.3,1.25);  ambient_rgb = U(0.12,0.45)*level
    #   key (light 0, shadow):  colour = warm_colour(level*U(0.6,1.6))
    #   fill (lights 1+, no sh): colour = warm_colour(level*U(0.3,0.9))
    #   warm_colour(I): w=U(-0.18,0.18) -> [I*(1+w), I, I*(1-w)]  (warm/cool tint on R/B)
    rng, klvl, kki, kfi, kw, kamb = jax.random.split(rng, 6)
    level = jax.random.uniform(klvl, (), minval=0.3, maxval=1.25)
    key_I = jax.random.uniform(kki, (1,), minval=0.6, maxval=1.6)
    fill_I = jax.random.uniform(kfi, (nlight - 1,), minval=0.3, maxval=0.9)
    inten = level * jp.concatenate([key_I, fill_I])                       # (nlight,)
    w = jax.random.uniform(kw, (nlight,), minval=-0.18, maxval=0.18)
    light_diffuse = jp.stack([inten * (1.0 + w), inten, inten * (1.0 - w)], axis=-1)  # (nlight,3)
    amb = jax.random.uniform(kamb, (3,), minval=0.12, maxval=0.45) * level
    light_ambient = jp.broadcast_to(amb, (nlight, 3))                     # per-world, tiled on lights
    # Scale ONLY the diffuse Lambert term by 1/pi (standard Lambertian normalization);
    # ambient stays RAW.
    light_diffuse = light_diffuse / jp.pi

    # --- Geometric per-world FOV DR: jitter cam_fovy (DEGREES) about the passed base so
    # the rendered base FOV is UNCHANGED — only the ±deg jitter is added (wrist ±1°,
    # overhead ±9°). ---
    rng2, kwf, kof = jax.random.split(rng, 3)
    cam_fovy = model.cam_fovy.at[wrist_cam].add(
        jax.random.uniform(kwf, (), minval=-1.0, maxval=1.0))
    if overhead_cam is not None:
        cam_fovy = cam_fovy.at[overhead_cam].add(
            jax.random.uniform(kof, (), minval=-9.0, maxval=9.0))

    return {"geom_rgba": geom_rgba, "geom_matid": geom_matid, "light_dir": light_dir,
            "light_diffuse": light_diffuse, "light_ambient": light_ambient, "cam_fovy": cam_fovy}


def vision_randomize(model, rng: jax.Array, box_geom=_BOX_GEOM, wrist_cam=_WRIST_CAM,
                     overhead_cam=None, floor_geom=_FLOOR_GEOM, wall_geoms=_WALL_GEOMS,
                     floor_matids=None, prong_geoms=_PRONG_GEOMS, table_geom=None,
                     tabletop_geom=None, wood_matid=-1):
    """Per-world visual model DR. Returns (batched_model, in_axes); applied in sequence
    with randomize.domain_randomize (see so101_vision.vision_randomization_fn).
    Randomizes cube colour/material, table colour + texture, off-table colour, light
    direction + diffuse/ambient, and camera FOV. `floor_matids` and `prong_geoms` are
    accepted for signature compatibility but unused.

    Madrona light-field batching rules:
    - MadronaWrapper._supplement_vision_randomization_fn REQUIRES every light field to
      be batched per-world (it tiles any left at in_axes=None). A field batched in the
      model but NOT listed in the in_axes union is read with in_axes=None -> garbage
      pixels. Here light_pos/light_dir are batched AND added to the union
      (so101_vision.vision_randomization_fn); light_type/castshadow/cutoff are left for
      _supplement to tile.
    - The renderer's shader computes the additive form `albedo*(ambient + sum
      light_colour*Lambert*shadow)`, so per-light INTENSITY+COLOUR and the coloured
      AMBIENT are randomized IN-RENDER via `light_diffuse`/`light_ambient` (the
      "IN-RENDER LIGHTING DR" block in sample_render_dr); the image-space
      `rgb_brightness` is only the per-sample ColorJitter on top.
    """
    nlight = model.light_pos.shape[0]
    if floor_matids is None:
        floor_matids = jp.array([model.geom_matid[floor_geom]])
    wall_geoms = jp.asarray(wall_geoms)
    prong_geoms = jp.asarray(prong_geoms)

    @jax.vmap
    def rand(rng):
        # Delegate to the module-level single-rng sampler. cam_pos/cam_quat/light_pos
        # are NOT drawn per world (the camera-pose DR is per-step, in so101_vision);
        # re-add the base values here to keep the batched (batched_model, in_axes)
        # contract.
        d = sample_render_dr(model, rng, box_geom=box_geom, table_geom=table_geom,
                             tabletop_geom=tabletop_geom, wood_matid=wood_matid,
                             floor_geom=floor_geom, wall_geoms=wall_geoms,
                             prong_geoms=prong_geoms, wrist_cam=wrist_cam,
                             overhead_cam=overhead_cam)
        return (d["geom_rgba"], d["geom_matid"], model.cam_pos, model.cam_quat,
                model.light_pos, d["light_dir"], d["light_diffuse"],
                d["light_ambient"], d["cam_fovy"])

    (geom_rgba, geom_matid, cam_pos, cam_quat, light_pos, light_dir,
     light_diffuse, light_ambient, cam_fovy) = rand(rng)

    batched = {"geom_rgba": 0, "geom_matid": 0, "cam_pos": 0, "cam_quat": 0,
               "light_pos": 0, "light_dir": 0, "light_diffuse": 0, "light_ambient": 0,
               "cam_fovy": 0}
    in_axes = jax.tree_util.tree_map(lambda x: None, model).tree_replace(batched)
    model = model.tree_replace({
        "geom_rgba": geom_rgba, "geom_matid": geom_matid,
        "cam_pos": cam_pos, "cam_quat": cam_quat, "light_pos": light_pos, "light_dir": light_dir,
        "light_diffuse": light_diffuse, "light_ambient": light_ambient,
        "cam_fovy": cam_fovy,
    })
    return model, in_axes


# --------------------------- image-space sensor DR ---------------------------

def _hue_rotation_matrix(a):
    """Luminance-preserving hue-rotation 3x3 (rotate RGB about the grey axis by angle `a`)."""
    c, s = jp.cos(a), jp.sin(a)
    return jp.array([
        [0.213 + c*0.787 - s*0.213, 0.715 - c*0.715 - s*0.715, 0.072 - c*0.072 + s*0.928],
        [0.213 - c*0.213 + s*0.143, 0.715 + c*0.285 + s*0.140, 0.072 - c*0.072 - s*0.283],
        [0.213 - c*0.213 - s*0.787, 0.715 - c*0.715 + s*0.715, 0.072 + c*0.928 + s*0.072],
    ])


def rgb_brightness(rgb, rng, *, exp_rng=None, bright=(0.4, 2.0), chan=(0.7, 1.35), contrast=(0.7, 1.4),
                   gamma=(0.7, 1.5), hue=0.5, sat=(0.5, 1.5), pixel_noise=8.0):
    """Per-step photometric aug on an RGB image (uint8 or float 0-255): torchvision-style
    ColorJitter (Squint's single most important DR — -18% real-world success without it),
    i.e. brightness/contrast/saturation/hue, plus per-channel gain, gamma and pixel noise.

      bright   global exposure multiply         hue      +/- rad hue rotation (about grey axis)
      chan     per-channel gain (white-balance) sat      saturation scale (toward/away grey)
      contrast about the image mean             gamma    tone curve
      pixel_noise  sensor grain (0-255 units)
    """
    rng, kb, kc, kk, kg, kh, ks, kn = jax.random.split(rng, 8)
    # Exposure level, warm/cool tint and coloured ambient are randomized IN-RENDER
    # (model.light_diffuse/light_ambient), so this layer is ONLY the per-sample
    # ColorJitter. exp_rng is retained for signature compatibility but UNUSED — drawing
    # exposure here would double-apply on top of the in-render level.
    del exp_rng
    x = rgb.astype(jp.float32) / 255.0
    x = x * jax.random.uniform(kb, (), minval=bright[0], maxval=bright[1])   # brightness
    x = x * jax.random.uniform(kc, (3,), minval=chan[0], maxval=chan[1])     # per-channel gain (identity if chan=(1,1))
    m = x.mean()
    x = (x - m) * jax.random.uniform(kk, (), minval=contrast[0], maxval=contrast[1]) + m
    # hue rotation (RGB about the grey axis)
    a = jax.random.uniform(kh, (), minval=-hue, maxval=hue)
    x = x @ _hue_rotation_matrix(a).T
    # saturation (blend toward per-pixel luminance)
    lum = (x * jp.array([0.299, 0.587, 0.114])).sum(-1, keepdims=True)
    x = lum + jax.random.uniform(ks, (), minval=sat[0], maxval=sat[1]) * (x - lum)
    x = jp.clip(x, 1e-4, 1.0) ** jax.random.uniform(kg, (), minval=gamma[0], maxval=gamma[1])
    x = x + (pixel_noise / 255.0) * jax.random.normal(kn, x.shape)
    return jp.clip(x * 255.0, 0, 255).astype(jp.uint8)
