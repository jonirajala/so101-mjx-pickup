"""Verify the geometric per-world FOV patch: the engine writes a
per-world CameraFovScale into PerspectiveCameraData every frame (GPU viewTransformUpdate),
fed from model.cam_fovy through renderer.py. Before the patch cam_fovy was BAKED at renderer
init and every world rendered the SAME FOV; the wrist FOV DR was faked in image space.

Renders NW worlds that are IDENTICAL except for the wrist-camera cam_fovy and asserts the
render responds GEOMETRICALLY: a WIDER FOV sees more of the scene, so the near foreground
(prongs at ~4 cm + cube at ~13 cm) occupies a SMALLER pixel fraction. Uses DEPTH (angle-only
change, lighting-independent) as the geometric probe, and asserts the RGB frames differ.

  micromamba run -n madmjx python engine-patches/verify_fov_patch.py
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
wrist_cam = mj.camera("wrist").id
print(f"nlight={mj.nlight} ncam={mj.ncam} wrist_cam={wrist_cam} "
      f"base wrist fovy={mj.cam_fovy[wrist_cam]:.1f}", flush=True)

m = mjx.put_model(mj)
qpos0 = jp.array(mj.key("home").qpos)
gid = mj.site("gripper").id
box_q = mj.jnt_qposadr[mj.joint("box_free").id]
d0 = mjx.forward(m, mjx.make_data(m).replace(qpos=qpos0))
grip = np.array(d0.site_xpos[gid])
# Put the cube right under the gripper so the wrist sees it.
qpos = qpos0.at[box_q:box_q + 3].set(jp.array([grip[0], grip[1], 0.015]))
d = mjx.forward(m, mjx.make_data(m).replace(qpos=qpos))

# NW worlds identical except wrist cam_fovy: narrow -> wide.
FOVS = [40.0, 70.0, 110.0]
NW = len(FOVS)
nl = mj.nlight

renderer = BatchRenderer(
    m, gpu_id=0, num_worlds=NW,
    batch_render_view_width=RES, batch_render_view_height=RES,
    enabled_geom_groups=np.array([0, 1, 2]),
    enabled_cameras=np.arange(mj.ncam), add_cam_debug_geo=False,
    use_rasterizer=False, viz_gpu_hdls=None,
)

def tile(x):
    return jp.repeat(jp.expand_dims(x, 0), NW, axis=0)

# Per-world cam_fovy: only the wrist camera differs across worlds.
cam_fovy = np.tile(np.asarray(mj.cam_fovy, np.float32), (NW, 1))   # (NW, ncam)
for i, f in enumerate(FOVS):
    cam_fovy[i, wrist_cam] = f
cam_fovy = jp.asarray(cam_fovy)

# Neutral lighting, identical across worlds (isolate the FOV effect).
amb = jp.asarray(np.full((NW, nl, 3), 0.25, np.float32))
dif = jp.asarray(np.full((NW, nl, 3), 0.6, np.float32))

batched = {
    "geom_rgba": tile(m.geom_rgba), "geom_matid": tile(m.geom_matid),
    "geom_size": tile(m.geom_size), "light_pos": tile(m.light_pos),
    "light_dir": tile(m.light_dir), "light_type": tile(m.light_type),
    "light_castshadow": tile(m.light_castshadow), "light_cutoff": tile(m.light_cutoff),
    "light_diffuse": dif, "light_ambient": amb, "cam_fovy": cam_fovy,
}
v_model = m.tree_replace(batched)
in_axes = jax.tree_util.tree_map(lambda _: None, m).tree_replace({k: 0 for k in batched})
v_data = jax.tree_util.tree_map(lambda x: tile(x), d)

_, rgb, depth = jax.vmap(renderer.init, in_axes=(0, in_axes))(v_data, v_model)
jax.block_until_ready((rgb, depth))
rgb = np.array(rgb)[:, wrist_cam, :, :, :3].astype(np.float32)     # (NW,H,W,3)
dep = np.array(depth)[:, wrist_cam, :, :, 0].astype(np.float32)    # (NW,H,W) metres

finite = np.isfinite(rgb).all() and np.isfinite(dep).all()
# Foreground = near geometry (prongs ~4cm, cube ~13cm). As the wrist FOV WIDENS, the near prongs
# (mounted right by the lens) sweep INTO the frame edges, so this near-depth fraction GROWS with FOV
# (confirmed by eye: fovy=40 is zoomed past the prongs onto the table; fovy=110 shows both prongs +
# the cube + surrounding table). Monotonic response == the per-world FOV is genuinely applied.
fg = ((dep > 1e-4) & (dep < 0.16)).reshape(NW, -1).mean(1)         # (NW,)
diff01 = np.abs(rgb[0] - rgb[-1]).mean()                          # narrow vs wide RGB must differ

print(f"\nfinite: {finite}")
for i, f in enumerate(FOVS):
    print(f"  wrist fovy={f:6.1f}  foreground_frac={fg[i]:.4f}")
print(f"narrow->wide foreground grows monotonically? "
      f"{fg[0] < fg[1] < fg[2]}   (Δ={fg[2]-fg[0]:+.4f})")
print(f"mean |RGB(narrow) - RGB(wide)| = {diff01:.2f} (0-255)")

for i, f in enumerate(FOVS):
    Image.fromarray(rgb[i].clip(0, 255).astype(np.uint8)).save(f"{HERE}/verify_fov_w{i}_fovy{int(f)}.png")

ok = bool(finite) and (fg[0] < fg[1] < fg[2]) and (fg[2] - fg[0] > 0.02) and (diff01 > 2.0)
print(f"\n{'PASS' if ok else 'FAIL'}: per-world geometric FOV "
      f"{'WORKS (each world renders its own cam_fovy)' if ok else 'did NOT take effect'}")
print("saved verify_fov_w0..2_*.png (narrow=40 -> wide=110)")
sys.exit(0 if ok else 1)
