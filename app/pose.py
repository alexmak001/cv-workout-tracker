import math
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np

from .config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODEL_URL,
    MIN_KEYPOINT_CONFIDENCE,
)

# ===============================
# KEYPOINT INDEX MAP (COCO order)
# ===============================

KEYPOINT_NAMES = [
    "nose",            # 0
    "left_eye",        # 1
    "right_eye",       # 2
    "left_ear",        # 3
    "right_ear",       # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle"      # 16
]


def get_keypoints_dict(result):
    """
    Convert YOLO keypoints for the most confident person into
    {name: (x, y, conf)} dict.

    Returns None if no people detected.
    """
    if result.keypoints is None or len(result.keypoints) == 0:
        return None

    # Take the first detected person
    kpts = result.keypoints.data[0].cpu().numpy()  # shape: (17, 3) -> x, y, conf

    kp_dict = {}
    for idx, name in enumerate(KEYPOINT_NAMES):
        x, y, c = kpts[idx]
        kp_dict[name] = (float(x), float(y), float(c))

    return kp_dict


def compute_left_shoulder_y_norm(kp_dict, frame_h):
    """
    Use ONLY left shoulder since camera is on the LEFT side.
    Returns normalized y in [0,1], or None if missing.
    """
    ls = kp_dict.get("left_shoulder", None)
    if ls is None:
        return None

    _, y_l, c_l = ls
    if c_l < MIN_KEYPOINT_CONFIDENCE:
        return None

    y_norm = y_l / float(frame_h)
    return y_norm


def angle_at_left_elbow(kp_dict):
    """
    Compute elbow angle for dips on LEFT side.
    Returns angle in degrees, or None if data missing/low-confidence.
    """
    S = kp_dict.get("left_shoulder")
    E = kp_dict.get("left_elbow")
    W = kp_dict.get("left_wrist")

    if S is None or E is None or W is None:
        return None

    sx, sy, sc = S
    ex, ey, ec = E
    wx, wy, wc = W

    if sc < MIN_KEYPOINT_CONFIDENCE or ec < MIN_KEYPOINT_CONFIDENCE or wc < MIN_KEYPOINT_CONFIDENCE:
        return None

    v1 = np.array([sx - ex, sy - ey], dtype=float)
    v2 = np.array([wx - ex, wy - ey], dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None

    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = max(-1.0, min(1.0, cos_theta))  # clamp
    theta = math.degrees(math.acos(cos_theta))
    return theta


def compute_ws(kp_dict, frame_h):
    """
    Compute wrist-vs-shoulder vertical delta (normalized by frame height).
    Positive => wrist above shoulder.
    """
    ls = kp_dict.get("left_shoulder")
    lw = kp_dict.get("left_wrist")
    if ls is None or lw is None:
        return None

    _, y_s, c_s = ls
    _, y_w, c_w = lw
    if c_s < MIN_KEYPOINT_CONFIDENCE or c_w < MIN_KEYPOINT_CONFIDENCE:
        return None

    return (y_s - y_w) / float(frame_h)


def ensure_model_path(model_arg):
    model_path = DEFAULT_MODEL_PATH if model_arg is None else model_arg
    if not hasattr(model_path, "is_file"):
        model_path = Path(model_path)

    if model_path.is_file():
        return model_path

    if model_path.name == DEFAULT_MODEL_NAME:
        target = model_path
        if model_path.parent == Path("."):
            target = DEFAULT_MODEL_PATH

        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"Downloading {DEFAULT_MODEL_NAME} to {target}...")
            try:
                urlretrieve(DEFAULT_MODEL_URL, target)
            except URLError as exc:
                raise RuntimeError(
                    f"Failed to download model from {DEFAULT_MODEL_URL}: {exc}"
                ) from exc
        return target

    return model_path
