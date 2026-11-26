"""Low-level image helpers. The original upload is never modified; everything is lossless PNG."""
from __future__ import annotations

import io
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}


class ImageError(ValueError):
    pass


def _to_bgr(im: Image.Image) -> np.ndarray:
    im = ImageOps.exif_transpose(im).convert("RGB")   # phones store rotation in EXIF
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        with Image.open(path) as im:
            arr = _to_bgr(im)
    except Exception as exc:  # noqa: BLE001
        raise ImageError(f"Cannot read image '{path.name}': {exc}") from exc
    if min(arr.shape[:2]) < 300:
        raise ImageError("Image is too small (need at least 300 px on the short side).")
    return arr


def bytes_to_image(data: bytes, name: str = "upload") -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as im:
            arr = _to_bgr(im)
    except Exception as exc:  # noqa: BLE001
        raise ImageError(f"Cannot read '{name}': {exc}") from exc
    if min(arr.shape[:2]) < 300:
        raise ImageError("Image is too small (need at least 300 px on the short side).")
    return arr


def save_png(path: str | Path, bgr: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise ImageError(f"PNG encode failed for {path}")
    path.write_bytes(buf.tobytes())  # works with non-ASCII Windows paths
    return path


def png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ImageError("PNG encode failed")
    return buf.tobytes()


def blue_mask(bgr: np.ndarray) -> np.ndarray:
    """Mask of the printed blue/cyan table lines and text (OpenCV hue ~ 85-130)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (85, 35, 60), (132, 255, 255))


def enlarge(img: np.ndarray, factor: int, method: str = "lanczos") -> np.ndarray:
    interp = {"lanczos": cv2.INTER_LANCZOS4, "bicubic": cv2.INTER_CUBIC,
              "nearest": cv2.INTER_NEAREST}[method]
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp)


def enhance_colour(bgr: np.ndarray) -> np.ndarray:
    """Colour-preserving enhancement: CLAHE on lightness + very mild unsharp mask."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.addWeighted(out, 1.4, blur, -0.4, 0)


def line_mask_red_channel(bgr: np.ndarray, drop: float = 0.80) -> np.ndarray:
    """Alternative line mask that is robust to uneven light: blue ink absorbs red light, red ink does not.

    Divides the red channel by its own large-scale background and keeps pixels that are clearly darker.
    """
    r = bgr[..., 2].astype(np.float32)
    k = max(31, (min(r.shape) // 12) | 1)
    bg = cv2.medianBlur(cv2.dilate(bgr[..., 2], np.ones((7, 7), np.uint8)), k).astype(np.float32) + 1.0
    return ((r / bg) < drop).astype(np.uint8) * 255
