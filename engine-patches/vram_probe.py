"""Minimal Madrona-MJX render + VRAM probe for the 8 GB go/no-go gate.

Batches the model per-world (geom_rgba/matid/size) exactly like the real
per-world DR path, so Madrona's custom-call batching rule is satisfied.
"""
import argparse, numpy as np, jax, jax.numpy as jp, mujoco
from mujoco import mjx
from madrona_mjx.renderer import BatchRenderer

ap = argparse.ArgumentParser()
ap.add_argument('--mjcf', required=True)
ap.add_argument('--num-worlds', type=int, default=256)
ap.add_argument('--res', type=int, default=64)
ap.add_argument('--steps', type=int, default=30)
args = ap.parse_args()
N = args.num_worlds

mj = mujoco.MjModel.from_xml_path(args.mjcf)
m = mjx.put_model(mj)
print(f"model: nq={m.nq} ncam={m.ncam} ngeom={m.ngeom}", flush=True)

renderer = BatchRenderer(
    m, gpu_id=0, num_worlds=N,
    batch_render_view_width=args.res, batch_render_view_height=args.res,
    enabled_geom_groups=np.array([0, 1, 2]),
    enabled_cameras=None, add_cam_debug_geo=False,
    use_rasterizer=False, viz_gpu_hdls=None,
)
print("renderer constructed", flush=True)

# Per-world model: batch the fields the renderer treats as per-world.
batched = {
    'geom_rgba': jp.repeat(jp.expand_dims(m.geom_rgba, 0), N, axis=0),
    'geom_matid': jp.repeat(jp.expand_dims(m.geom_matid, 0), N, axis=0),
    'geom_size': jp.repeat(jp.expand_dims(m.geom_size, 0), N, axis=0),
}
v_model = m.tree_replace(batched)
in_axes = jax.tree_util.tree_map(lambda _: None, m).tree_replace(
    {'geom_rgba': 0, 'geom_matid': 0, 'geom_size': 0})

keys = jax.random.split(jax.random.PRNGKey(0), N)
def make_one(rng):
    d = mjx.make_data(m)
    d = d.replace(qpos=d.qpos + 0.01 * jax.random.uniform(rng, (m.nq,)))
    return mjx.forward(m, d)
v_data = jax.vmap(make_one)(keys)

render_token, rgb, depth = jax.vmap(renderer.init, in_axes=(0, in_axes))(v_data, v_model)
jax.block_until_ready((rgb, depth))
print(f"init render OK  rgb={rgb.shape}{rgb.dtype}  depth={depth.shape}{depth.dtype}", flush=True)

@jax.jit
def step(carry, _):
    tok, data = carry
    tok, rgb, depth = jax.vmap(renderer.render, in_axes=(None, 0))(tok, data)
    return (tok, data), (rgb, depth)

(_, _), (rgbs, depths) = jax.lax.scan(step, (render_token, v_data), None, length=args.steps)
jax.block_until_ready((rgbs, depths))
print(f"rendered {args.steps} steps x {N} worlds @ {args.res}x{args.res}  OK", flush=True)
