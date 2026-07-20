"""Verify the Madrona additive-lighting patch: the shader computes
   color = albedo * ( ambient_rgb + sum_lights light_color * clamp(N.L,0,1) * shadow )
replacing the stock white max(0.2, sum Lambert)*albedo.

Renders 4 worlds with hand-set light_ambient / light_diffuse and asserts the shader
RESPONDS to them (it ignored both before the patch):
  W0 dark ambient (0.12)   vs  W1 bright ambient (0.45)  -> mean brightness must RISE
  W2 warm diffuse          vs  W3 cool diffuse           -> R/B ratio must flip
Also asserts every pixel is finite (no NaN from the new Vector3 accumulate).

  micromamba run -n madmjx python engine-patches/verify_lighting_patch.py
"""
import os, sys, numpy as np, jax, jax.numpy as jp
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from mujoco import mjx
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from model_loader import load_mj_model
from madrona_mjx.renderer import BatchRenderer

RES = 64
mj = load_mj_model()
print(f"nlight={mj.nlight} ncam={mj.ncam}", flush=True)
assert mj.nlight >= 1, "scene has no lights"
wrist_cam = mj.camera("wrist").id

m = mjx.put_model(mj)
qpos0 = jp.array(mj.key("home").qpos)
gid = mj.site("gripper").id
box_q = mj.jnt_qposadr[mj.joint("box_free").id]
d0 = mjx.forward(m, mjx.make_data(m).replace(qpos=qpos0))
grip = np.array(d0.site_xpos[gid])
qpos = qpos0.at[box_q:box_q + 3].set(jp.array([grip[0], grip[1], 0.015]))
d = mjx.forward(m, mjx.make_data(m).replace(qpos=qpos))

NW = 4
nl = mj.nlight
renderer = BatchRenderer(
    m, gpu_id=0, num_worlds=NW,
    batch_render_view_width=RES, batch_render_view_height=RES,
    enabled_geom_groups=np.array([0, 1, 2]),
    enabled_cameras=np.arange(mj.ncam), add_cam_debug_geo=False,
    use_rasterizer=False, viz_gpu_hdls=None,
)

# Per-world light_ambient / light_diffuse (the fields the patch consumes).
amb = np.zeros((NW, nl, 3), np.float32)
dif = np.zeros((NW, nl, 3), np.float32)
amb[0] = 0.12; amb[1] = 0.45; amb[2] = 0.20; amb[3] = 0.20          # W0 dark, W1 bright ambient
dif[0] = 0.5;  dif[1] = 0.5                                          # neutral diffuse for ambient test
dif[2] = np.array([0.9, 0.5, 0.2]); dif[3] = np.array([0.2, 0.5, 0.9])  # warm vs cool
amb = jp.asarray(amb); dif = jp.asarray(dif)

def tile(x):
    return jp.repeat(jp.expand_dims(x, 0), NW, axis=0)

batched = {
    "geom_rgba": tile(m.geom_rgba), "geom_matid": tile(m.geom_matid),
    "geom_size": tile(m.geom_size), "light_pos": tile(m.light_pos),
    "light_dir": tile(m.light_dir), "light_type": tile(m.light_type),
    "light_castshadow": tile(m.light_castshadow), "light_cutoff": tile(m.light_cutoff),
    "light_diffuse": dif, "light_ambient": amb,
}
v_model = m.tree_replace(batched)
in_axes = jax.tree_util.tree_map(lambda _: None, m).tree_replace(
    {k: 0 for k in batched})
v_data = jax.tree_util.tree_map(lambda x: tile(x), d)

_, rgb, depth = jax.vmap(renderer.init, in_axes=(0, in_axes))(v_data, v_model)
jax.block_until_ready((rgb, depth))
rgb = np.array(rgb)[:, wrist_cam, :, :, :3].astype(np.float32)   # (NW,H,W,3)

finite = np.isfinite(rgb).all()
means = rgb.reshape(NW, -1, 3).mean(1)                            # (NW,3) per-world mean RGB
b0, b1 = means[0].mean(), means[1].mean()
rb_warm = means[2][0] / max(means[2][2], 1e-6)                    # R/B under warm diffuse
rb_cool = means[3][0] / max(means[3][2], 1e-6)                    # R/B under cool diffuse

print(f"\nfinite pixels: {finite}")
print(f"W0 dark-ambient mean={b0:.1f}   W1 bright-ambient mean={b1:.1f}   (bright>dark? {b1>b0+1})")
print(f"W2 warm R/B={rb_warm:.2f}   W3 cool R/B={rb_cool:.2f}   (warm>cool? {rb_warm>rb_cool+0.1})")

for i in range(NW):
    Image.fromarray(np.array(rgb[i]).clip(0, 255).astype(np.uint8)).save(f"{HERE}/verify_light_w{i}.png")

ok = bool(finite) and (b1 > b0 + 1) and (rb_warm > rb_cool + 0.1)
print(f"\n{'PASS' if ok else 'FAIL'}: patched shader {'reads' if ok else 'does NOT read'} ambient+diffuse")
print("saved verify_light_w0..3.png (w0 dark, w1 bright, w2 warm, w3 cool)")
sys.exit(0 if ok else 1)
