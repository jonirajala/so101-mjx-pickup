"""Real wrist/overhead camera capture, framed to match the sim cameras.

crop_resize mirrors the sim framing exactly: rotate 180 deg (inverted mount), centre-crop
square, INTER_AREA resize. WristCamera wraps an OpenCV device with that framing. Probe
device indices with:  python wrist_camera.py --probe
"""
from __future__ import annotations

import numpy as np

_RENDER = 64                 # so101_vision render size (square)
_WRIST_ROT_DEG = 180         # inverted mount


def crop_resize(frame: np.ndarray, size: int = _RENDER, rot_deg: int = _WRIST_ROT_DEG,
                fov_zoom: float | None = None) -> np.ndarray:
    """Rotate, centre-crop to square, (optional fov_zoom center-crop), resize to (size,size).
    fov_zoom!=None additionally center-crops by that factor to emulate a narrower sim fovy."""
    import cv2
    if rot_deg:
        frame = np.rot90(frame, k=(rot_deg // 90) % 4).copy()
    h, w = frame.shape[:2]
    c = min(h, w)
    frame = frame[(h - c) // 2:(h - c) // 2 + c, (w - c) // 2:(w - c) // 2 + c]
    if fov_zoom is not None:
        # center-crop the square by fov_zoom, keeping the optical center fixed (= narrower fovy).
        cc = frame.shape[0]
        m = int(round(cc * fov_zoom))
        o = (cc - m) // 2
        frame = frame[o:o + m, o:o + m]
    # INTER_AREA = the anti-aliased area-downsample the policy trains on (sim renders high then
    # area-downsamples; deploy must match, else interpolation mismatch is its own sim2real gap).
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


class WristCamera:
    """OpenCV RGB grab from the gripper wrist cam, cropped to the sim wrist-cam framing.

    index: OpenCV device index. With several USB cams attached the wrist cam may NOT be 0 —
    use `python wrist_camera.py --probe` to find it, then pass --camera N.
    """

    def __init__(self, index: int = 0, size: int = _RENDER, rot_deg: int = _WRIST_ROT_DEG,
                 width: int = 640, height: int = 480, fov_zoom: float | None = None):
        import cv2
        self._cv2 = cv2
        self.size, self.rot_deg, self.fov_zoom = size, rot_deg, fov_zoom
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open wrist camera at OpenCV index {index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def grab_rgb(self) -> np.ndarray:
        """(size,size,3) uint8 RGB, framed like the sim wrist cam (fov_zoom emulates a narrower fovy)."""
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("wrist camera read() failed")
        rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
        return crop_resize(rgb, self.size, self.rot_deg, fov_zoom=self.fov_zoom)

    def close(self):
        self.cap.release()


def _probe(max_index: int = 6):
    """Save one framed RGB grab from each opened OpenCV index, to pick the wrist cam."""
    import cv2, os
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(here, exist_ok=True)
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            continue
        ok, bgr = cap.read()
        if ok:
            p = f"{here}/probe_cam{i}.png"
            cv2.imwrite(p, cv2.cvtColor(crop_resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
                                        cv2.COLOR_RGB2BGR))
            print(f"index {i}: {bgr.shape[1]}x{bgr.shape[0]} -> {p}")
        cap.release()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="wrist camera probe")
    ap.add_argument("--probe", action="store_true", help="dump a framed grab from each cam index")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    _probe()
