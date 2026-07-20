"""Load the SO-101 pick-cube model via from_xml_string + an in-memory asset dict.

Using an asset dict (rather than from_xml_path) sidesteps MuJoCo's brittle meshdir
resolution when a model is pulled in with <include>. Keys are provided in several
forms (basename, and dir-prefixed) so includes and meshdir-relative meshes both resolve.
"""
from etils import epath
import mujoco

MODELS_DIR = epath.Path(__file__).parent / "models"
SCENE_XML = MODELS_DIR / "so101_pick_cube.xml"
# The scene includes so101_real/so101_mjx.xml (the MJX-adapted SO-101 arm with box grasp pads).
_ARM_DIR = MODELS_DIR / "so101_real"


def get_assets() -> dict:
    # MuJoCo's VFS matches assets by basename (and collides if two keys share one),
    # so key everything by basename only — meshdir/include subpaths resolve to these.
    assets = {}

    def add(f):
        assert f.name not in assets, f"asset basename collision: {f.name}"
        assets[f.name] = f.read_bytes()

    for f in MODELS_DIR.glob("*.xml"):
        add(f)
    for ext in ("*.png", "*.jpg", "*.obj", "*.stl"):   # texture images + mesh assets (e.g. floor_cardboard.png, tabletop_quad.obj)
        for f in MODELS_DIR.glob(ext):
            add(f)
    for f in _ARM_DIR.glob("*.xml"):
        add(f)
    for f in (_ARM_DIR / "assets").glob("*"):
        if f.is_file():
            add(f)
    return assets


def load_mj_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(SCENE_XML.read_text(), assets=get_assets())
