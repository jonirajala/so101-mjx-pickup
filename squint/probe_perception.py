"""Is the cube even RESOLVABLE in the 16x16 wrist view across the spawn region?

Measures how many pixels the 3 cm cube occupies at 16x16 — and whether it is in frame at all —
from the start (hover) pose, across the wide FAN spawn vs the small box spawn. If the cube is
~1 px / out of frame at the fan edges, 16x16 simply can't drive localization there.

Renders the wrist camera (mujoco.Renderer) at 128, area-downsamples to 16, counts green-cube
pixels, and writes a montage. Run in the mjx venv:
  MUJOCO_GL=egl ~/venvs/mjx/bin/python squint/probe_perception.py
"""
import os, sys, math
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODE)
from model_loader import load_mj_model
from ik import solve_ik, ARM_JOINTS

_HOVER = np.array([0.03, -0.18, 0.13])
_RES = 128
_OBS = 16


def cube_mask_16(small):
    g, r, b = small[..., 1], small[..., 0], small[..., 2]
    return (g > r + 25) & (g > b + 25) & (g > 60)   # bright-green cube (rgba 0 0.8 0.2)


def main():
    m = load_mj_model()
    d = mujoco.MjData(m)
    q_home, err = solve_ik(m, _HOVER, jaw=0.6)
    assert err < 5e-3, f"hover IK err {err}"
    arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
    jaw_qadr = m.jnt_qposadr[m.joint("gripper").id]
    box_adr = m.jnt_qposadr[m.joint("box_free").id]
    rnd = mujoco.Renderer(m, height=_RES, width=_RES)

    def cube_px(cube_xy):
        d.qpos[:] = q_home
        d.qpos[box_adr:box_adr + 3] = [cube_xy[0], cube_xy[1], 0.015]
        d.qpos[jaw_qadr] = 0.6
        mujoco.mj_forward(m, d)
        rnd.update_scene(d, camera="wrist")
        rgb = rnd.render().astype(np.float32)
        small = rgb.reshape(_OBS, _RES // _OBS, _OBS, _RES // _OBS, 3).mean((1, 3))
        mask = cube_mask_16(small)
        # also high-res in-frame check
        hi = cube_mask_16(rgb)
        return int(mask.sum()), int(hi.sum()), small.astype(np.uint8)

    print("=== FAN spawn (wide, r[0.13,0.24] theta+/-50) ===")
    fan = []
    for r in (0.13, 0.18, 0.24):
        for th in (-50, -25, 0, 25, 50):
            x, y = r * math.sin(math.radians(th)), -r * math.cos(math.radians(th))
            n16, nhi, img = cube_px((x, y))
            fan.append(img)
            inframe = "in-frame" if nhi > 0 else "OUT-OF-FRAME"
            print(f"  r={r} th={th:+d}  xy=({x:+.3f},{y:+.3f})  16x16 cube px={n16:2d}  hi-res px={nhi:4d}  {inframe}")

    print("=== small BOX spawn (original, x[-0.06,0.13] y[-0.23,-0.13] — SAC learned here) ===")
    for x in (-0.06, 0.03, 0.13):
        for y in (-0.23, -0.18, -0.13):
            n16, nhi, _ = cube_px((x, y))
            inframe = "in-frame" if nhi > 0 else "OUT-OF-FRAME"
            print(f"  xy=({x:+.3f},{y:+.3f})  16x16 cube px={n16:2d}  hi-res px={nhi:4d}  {inframe}")

    # montage of the fan 16x16 views (upscaled) for eyeballing — saved if PIL is available
    try:
        from PIL import Image
        grid = np.concatenate([np.concatenate([np.kron(fan[r * 5 + c], np.ones((8, 8, 1), np.uint8))
                                               for c in range(5)], axis=1) for r in range(3)], axis=0)
        out = os.path.join(os.path.dirname(__file__), "runs", "perception_fan_16x16.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.fromarray(grid).save(out)
        print(f"\nwrote fan montage -> {out}")
    except Exception as e:
        print(f"\n(montage skipped: {e})")

    # --- analytical cube pixel size: ours vs the Squint reference camera geometry ---
    def px(cube_m, fov_deg, dist, res=16):
        return (cube_m / dist) / (2 * math.tan(math.radians(fov_deg) / 2)) * res
    print("\n=== ANALYTICAL cube apparent size at 16x16 (cube_size/dist / (2 tan(fov/2)) * 16) ===")
    print(f"  OURS    cube=0.03 m, fovy=95: @dist0.12={px(0.03,95,0.12):.2f}px  @0.20={px(0.03,95,0.20):.2f}px  @0.30={px(0.03,95,0.30):.2f}px")
    print(f"  SQUINT  cube~0.05 m, fov=71: @dist0.12={px(0.05,71,0.12):.2f}px  @0.20={px(0.05,71,0.20):.2f}px  @0.30={px(0.05,71,0.30):.2f}px")


if __name__ == "__main__":
    main()
