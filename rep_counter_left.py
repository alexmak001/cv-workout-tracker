import time
import math
import argparse
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2
from ultralytics import YOLO
import numpy as np


# ===============================
# CONFIG / TUNING PARAMETERS
# ===============================

# --- Model download/cache ---
MODEL_DIR = Path("models")
DEFAULT_MODEL_NAME = "yolov8n-pose.pt"
DEFAULT_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt"
)
DEFAULT_MODEL_PATH = MODEL_DIR / DEFAULT_MODEL_NAME

# --- Pose & detection ---
MIN_KEYPOINT_CONFIDENCE = 0.4   # TUNE_ME: raise if jittery/missing keypoints

# --- Smoothing ---
SMOOTHING_ALPHA = 0.2           # TUNE_ME: 0.1–0.3; higher = smoother but more lag

# --- Pull-up thresholds (normalized left shoulder Y in [0,1]) ---
# IMPORTANT: Y increases downward in images.
# So "top" (chin over bar) = smaller y, "bottom" (dead hang) = larger y.
PULLUP_TOP_THRESH    = 0.33   # above your top band
PULLUP_BOTTOM_THRESH = 0.38   # below your bottom band

# --- Dip thresholds (left elbow angle in degrees) ---
# Typical: top of dip ~160–180°, bottom ~70–110° (depends on your form)
DIP_TOP_ANGLE = 150.0           # TUNE_ME: angle at/near lockout
DIP_BOTTOM_ANGLE = 90.0        # TUNE_ME: angle at deepest part

# --- Rep debounce ---
MIN_TIME_BETWEEN_REPS = 0.35    # TUNE_ME: seconds; avoid double-counting fast jitters

# --- Auto-detect tuning ---
DETECT_WINDOW_SECONDS = 2.0     # TUNE_ME: seconds to collect detection samples
REST_SECONDS = 4.0              # TUNE_ME: seconds of rest to reset auto mode
SHOULDER_RANGE_THRESH = 0.06    # TUNE_ME: y_norm range to indicate pullups
ELBOW_RANGE_THRESH = 30.0       # TUNE_ME: angle range to indicate dips
DOMINANCE_K = 1.3               # TUNE_ME: shoulder range dominance vs elbow range
MOTION_EPS_Y = 0.01             # TUNE_ME: y_norm delta to count as movement
MOTION_EPS_ANGLE = 3.0          # TUNE_ME: angle delta (deg) to count as movement
MIN_DETECT_SAMPLES = 5          # TUNE_ME: minimum samples per signal for detection


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


def exponential_smooth(new_value, prev_value, alpha):
    if prev_value is None:
        return new_value
    return alpha * new_value + (1.0 - alpha) * prev_value


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


class RepCounterPullup:
    """
    Simple FSM for pull-up reps based on LEFT shoulder vertical position.
    """

    def __init__(self):
        self.state = "WAITING_FOR_REP"
        self.rep_count = 0
        self.last_rep_time = 0.0
        self.smooth_y = None

    def update(self, y_norm, timestamp):
        if y_norm is None:
            return self.rep_count, self.state

        self.smooth_y = exponential_smooth(y_norm, self.smooth_y, SMOOTHING_ALPHA)
        y = self.smooth_y

        # States:
        # WAITING_FOR_REP -> AT_BOTTOM once we see bottom-ish position
        # AT_BOTTOM -> AT_TOP when cross TOP threshold (up)
        # AT_TOP -> AT_BOTTOM when cross BOTTOM threshold (down) => count rep

        if self.state == "WAITING_FOR_REP":
            if y > PULLUP_BOTTOM_THRESH:
                self.state = "AT_BOTTOM"

        elif self.state == "AT_BOTTOM":
            if y < PULLUP_TOP_THRESH:
                self.state = "AT_TOP"

        elif self.state == "AT_TOP":
            if y > PULLUP_BOTTOM_THRESH:
                if timestamp - self.last_rep_time > MIN_TIME_BETWEEN_REPS:
                    self.rep_count += 1
                    self.last_rep_time = timestamp
                self.state = "AT_BOTTOM"

        return self.rep_count, self.state


class RepCounterDip:
    """
    Simple FSM for dip reps based on LEFT elbow angle.
    """

    def __init__(self):
        self.state = "WAITING_FOR_REP"
        self.rep_count = 0
        self.last_rep_time = 0.0
        self.smooth_angle = None

    def update(self, angle, timestamp):
        if angle is None:
            return self.rep_count, self.state

        self.smooth_angle = exponential_smooth(angle, self.smooth_angle, SMOOTHING_ALPHA)
        a = self.smooth_angle

        # States:
        # WAITING_FOR_REP -> AT_TOP when we see angle near top
        # AT_TOP -> AT_BOTTOM when angle < BOTTOM_ANGLE
        # AT_BOTTOM -> AT_TOP when angle > TOP_ANGLE => count rep

        if self.state == "WAITING_FOR_REP":
            if a > DIP_TOP_ANGLE:
                self.state = "AT_TOP"

        elif self.state == "AT_TOP":
            if a < DIP_BOTTOM_ANGLE:
                self.state = "AT_BOTTOM"

        elif self.state == "AT_BOTTOM":
            if a > DIP_TOP_ANGLE:
                if timestamp - self.last_rep_time > MIN_TIME_BETWEEN_REPS:
                    self.rep_count += 1
                    self.last_rep_time = timestamp
                self.state = "AT_TOP"

        return self.rep_count, self.state


def draw_overlay(frame, mode, rep_count, state, extra_value=None):
    """
    Draw basic info on frame.
    extra_value: y_norm for pullups or angle for dips, for debugging.
    """
    h, w = frame.shape[:2]

    # translucent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    text_mode = f"Mode: {mode.upper()} (LEFT SIDE VIEW)"
    text_reps = f"Reps: {rep_count}"
    text_state = f"State: {state}"

    cv2.putText(frame, text_mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(frame, text_reps, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, text_state, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if extra_value is not None:
        if mode == "pullup":
            t = f"Left shoulder y_norm: {extra_value:.3f}"
        else:
            t = f"Left elbow angle: {extra_value:.1f} deg"
        cv2.putText(frame, t, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 3)

    return frame


def ensure_model_path(model_arg):
    model_path = Path(model_arg)
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


def main():
    parser = argparse.ArgumentParser(description="YOLO-based rep counter (pullups & dips, LEFT-side view).")
    parser.add_argument("--mode", choices=["pullup", "dip"], required=False,
                        help="Exercise mode: pullup or dip")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Path to YOLO pose model (e.g., yolov8n-pose.pt)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file. If not set, use webcam 0.")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (webcam mode only).")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["avfoundation", "any", "default"],
                        help="OpenCV backend: avfoundation, any, or default.")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect exercise (pullup or dip).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Calibration mode (pullup or dip). Press b/t to sample, r to reset.")
    args = parser.parse_args()

    if not args.auto and args.mode is None:
        parser.error("--mode is required unless --auto is set.")

    model_path = ensure_model_path(args.model)
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    def open_camera(camera_index, backend_choice):
        if backend_choice == "avfoundation" and sys.platform == "darwin":
            print(f"Opening camera index {camera_index} with backend=avfoundation")
            return cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        print(f"Opening camera index {camera_index} with backend={backend_choice}")
        return cv2.VideoCapture(camera_index)

    if args.video is None:
        if args.backend is None:
            backend_choice = "avfoundation" if sys.platform == "darwin" else "default"
        else:
            backend_choice = args.backend
        cap = open_camera(args.camera, backend_choice)
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    if args.video is None:
        start_warmup = time.time()
        warmup_frames = 0
        while warmup_frames < 20 and (time.time() - start_warmup) < 1.0:
            ret, _ = cap.read()
            if ret:
                warmup_frames += 1

    if args.auto:
        pullup_counter = RepCounterPullup()
        dip_counter = RepCounterDip()
    else:
        if args.mode == "pullup":
            counter = RepCounterPullup()
        else:
            counter = RepCounterDip()
    if args.auto and args.mode in ("pullup", "dip"):
        counter = pullup_counter if args.mode == "pullup" else dip_counter

    paused = False
    if args.calibrate:
        print("Calibration mode: press 'b' (bottom), 't' (top), 'r' (reset), 'q' (quit).")
    else:
        print("Press 'q' to quit.")

    # --- Calibration-only state (pullups only) ---
    bottom_samples = []
    top_samples = []
    last_suggested = None
    last_frame = None
    last_result = None
    last_kp_dict = None
    last_extra_val = None
    last_rep_count = 0
    last_state = "WAITING_FOR_REP"
    last_suggested_msg = None
    last_suggested_time = 0.0
    auto_state = "IDLE"
    locked_mode = None
    detect_start_time = None
    detect_shoulder_vals = []
    detect_elbow_vals = []
    auto_smooth_y = None
    auto_smooth_angle = None
    last_auto_smooth_y = None
    last_auto_smooth_angle = None
    last_motion_time = None
    last_detect_shoulder_range = 0.0
    last_detect_elbow_range = 0.0
    last_overlay_mode = "auto" if args.auto else args.mode
    last_auto_state = auto_state
    last_locked_mode = locked_mode
    button_rect = (0, 0, 0, 0)
    window_name = "Rep Counter (Left Side)"

    def on_mouse(event, x, y, flags, userdata):
        nonlocal paused
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                paused = not paused

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        if not paused:
            if args.video is None:
                ret, frame = cap.read()
                if not ret:
                    print("Frame read failed, retrying for up to 2 seconds...")
                    start_retry = time.time()
                    while (time.time() - start_retry) < 2.0:
                        time.sleep(0.05)
                        ret, frame = cap.read()
                        if ret:
                            break
                    if not ret:
                        print("Reopening camera after read failures...")
                        cap.release()
                        cap = open_camera(args.camera, backend_choice)
                        if not cap.isOpened():
                            print("Error: Could not reopen camera. Exiting.")
                            break
                        start_warmup = time.time()
                        warmup_frames = 0
                        while warmup_frames < 20 and (time.time() - start_warmup) < 1.0:
                            ret, _ = cap.read()
                            if ret:
                                warmup_frames += 1
                        ret, frame = cap.read()
                        if not ret:
                            print("Error: Camera read failed after reopen. Exiting.")
                            break
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            frame_h, frame_w = frame.shape[:2]
            last_frame = frame.copy()

            results = model(frame, verbose=False)
            result = results[0]
            last_result = result

            kp_dict = get_keypoints_dict(result)
            last_kp_dict = kp_dict

            now = time.time()
            extra_val = None
            y_norm = None
            angle = None
            if kp_dict is not None:
                y_norm = compute_left_shoulder_y_norm(kp_dict, frame_h)
                angle = angle_at_left_elbow(kp_dict)

            if args.auto:
                if y_norm is not None:
                    auto_smooth_y = exponential_smooth(y_norm, auto_smooth_y, SMOOTHING_ALPHA)
                if angle is not None:
                    auto_smooth_angle = exponential_smooth(angle, auto_smooth_angle, SMOOTHING_ALPHA)

                moved = False
                if auto_smooth_y is not None and last_auto_smooth_y is not None:
                    if abs(auto_smooth_y - last_auto_smooth_y) > MOTION_EPS_Y:
                        moved = True
                if auto_smooth_angle is not None and last_auto_smooth_angle is not None:
                    if abs(auto_smooth_angle - last_auto_smooth_angle) > MOTION_EPS_ANGLE:
                        moved = True
                if moved:
                    last_motion_time = now

                last_auto_smooth_y = auto_smooth_y
                last_auto_smooth_angle = auto_smooth_angle

                if auto_state == "IDLE":
                    if moved:
                        auto_state = "DETECTING"
                        detect_start_time = now
                        detect_shoulder_vals = []
                        detect_elbow_vals = []

                if auto_state == "DETECTING":
                    if auto_smooth_y is not None:
                        detect_shoulder_vals.append(auto_smooth_y)
                    if auto_smooth_angle is not None:
                        detect_elbow_vals.append(auto_smooth_angle)

                    if len(detect_shoulder_vals) >= 2:
                        last_detect_shoulder_range = max(detect_shoulder_vals) - min(detect_shoulder_vals)
                    else:
                        last_detect_shoulder_range = 0.0
                    if len(detect_elbow_vals) >= 2:
                        last_detect_elbow_range = max(detect_elbow_vals) - min(detect_elbow_vals)
                    else:
                        last_detect_elbow_range = 0.0

                    elapsed = now - (detect_start_time or now)
                    if elapsed >= DETECT_WINDOW_SECONDS:
                        enough_samples = (
                            len(detect_shoulder_vals) >= MIN_DETECT_SAMPLES
                            and len(detect_elbow_vals) >= MIN_DETECT_SAMPLES
                        )
                        if not enough_samples and elapsed < (DETECT_WINDOW_SECONDS + 1.0):
                            pass
                        else:
                            shoulder_range = last_detect_shoulder_range
                            elbow_range = last_detect_elbow_range
                            if (
                                shoulder_range >= SHOULDER_RANGE_THRESH
                                and shoulder_range > (elbow_range * DOMINANCE_K)
                            ):
                                locked_mode = "pullup"
                            elif elbow_range >= ELBOW_RANGE_THRESH:
                                locked_mode = "dip"
                            else:
                                locked_mode = "unknown"

                            auto_state = "LOCKED"
                            pullup_counter = RepCounterPullup()
                            dip_counter = RepCounterDip()
                            last_motion_time = now

                if auto_state == "LOCKED":
                    if last_motion_time is not None and (now - last_motion_time) > REST_SECONDS:
                        auto_state = "IDLE"
                        locked_mode = None
                        detect_start_time = None
                        detect_shoulder_vals = []
                        detect_elbow_vals = []
                        pullup_counter = RepCounterPullup()
                        dip_counter = RepCounterDip()
                        rep_count = 0
                        state = "IDLE"
                        last_overlay_mode = "auto"
                    elif locked_mode == "pullup":
                        extra_val = y_norm
                        rep_count, state = pullup_counter.update(y_norm, now)
                        last_overlay_mode = "pullup"
                    elif locked_mode == "dip":
                        extra_val = angle
                        rep_count, state = dip_counter.update(angle, now)
                        last_overlay_mode = "dip"
                    else:
                        rep_count = 0
                        state = "UNKNOWN"
                        last_overlay_mode = "unknown"
                else:
                    rep_count = 0
                    state = auto_state
                    last_overlay_mode = "auto"
            else:
                if args.mode == "pullup":
                    extra_val = y_norm
                    rep_count, state = counter.update(y_norm, now)
                else:  # dip
                    extra_val = angle
                    rep_count, state = counter.update(angle, now)
                last_overlay_mode = args.mode

            last_extra_val = extra_val
            last_rep_count = rep_count
            last_state = state
            last_auto_state = auto_state
            last_locked_mode = locked_mode

        if last_frame is None:
            continue

        frame = last_frame.copy()
        frame_h, frame_w = frame.shape[:2]
        rep_count = last_rep_count
        state = last_state
        extra_val = last_extra_val
        result = last_result
        kp_dict = last_kp_dict
        overlay_mode = last_overlay_mode
        auto_state = last_auto_state
        locked_mode = last_locked_mode

        # Draw left-side keypoints for debugging
        if result is not None and result.keypoints is not None and len(result.keypoints) > 0:
            for person in result.keypoints.xy:
                for x, y in person:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        frame = draw_overlay(frame, overlay_mode, rep_count, state, extra_value=extra_val)

        if args.auto:
            auto_label = f"Auto: {auto_state}"
            cv2.putText(frame, auto_label, (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            if auto_state == "LOCKED":
                detected = locked_mode.upper() if locked_mode else "UNKNOWN"
                cv2.putText(frame, f"Detected: {detected}", (10, 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                if locked_mode == "unknown":
                    cv2.putText(frame, "UNKNOWN - keep moving or press 1/2 to force mode",
                                (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            elif auto_state == "DETECTING":
                cv2.putText(frame,
                            f"ranges: shoulder={last_detect_shoulder_range:.3f}  elbow={last_detect_elbow_range:.1f}",
                            (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- Calibration-only behavior (pullups) ---
        if args.calibrate and args.mode == "pullup":
            smooth_y = counter.smooth_y
            if smooth_y is not None:
                cv2.putText(
                    frame,
                    f"y_norm (smoothed): {smooth_y:.3f}",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 0),
                    3,
                )

            if len(bottom_samples) >= 3 and len(top_samples) >= 3:
                bottom_mean = float(np.mean(bottom_samples))
                bottom_min = float(np.min(bottom_samples))
                bottom_max = float(np.max(bottom_samples))
                top_mean = float(np.mean(top_samples))
                top_min = float(np.min(top_samples))
                top_max = float(np.max(top_samples))

                suggested_top = (top_max + bottom_min) / 2.0
                suggested_bottom = max(0.0, min(1.0, bottom_min - 0.05))
                last_suggested = (suggested_top, suggested_bottom)

                y0 = 175
                step = 26
                cv2.putText(frame, f"bottom mean/min/max: {bottom_mean:.3f} / {bottom_min:.3f} / {bottom_max:.3f}",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
                cv2.putText(frame, f"top    mean/min/max: {top_mean:.3f} / {top_min:.3f} / {top_max:.3f}",
                            (10, y0 + step), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
                cv2.putText(frame, f"suggested top/bottom: {suggested_top:.3f} / {suggested_bottom:.3f}",
                            (10, y0 + 2 * step), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 0), 2)
        elif args.calibrate and args.mode == "dip":
            smooth_angle = counter.smooth_angle
            if smooth_angle is not None:
                cv2.putText(
                    frame,
                    f"elbow angle (smoothed): {smooth_angle:.1f} deg",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 0),
                    3,
                )

            if len(bottom_samples) >= 3 and len(top_samples) >= 3:
                bottom_mean = float(np.mean(bottom_samples))
                bottom_min = float(np.min(bottom_samples))
                bottom_max = float(np.max(bottom_samples))
                top_mean = float(np.mean(top_samples))
                top_min = float(np.min(top_samples))
                top_max = float(np.max(top_samples))

                suggested_bottom = bottom_max + 5.0
                suggested_top = top_min - 5.0
                if suggested_bottom >= suggested_top:
                    suggested_bottom = bottom_max
                    suggested_top = top_min
                last_suggested = (suggested_top, suggested_bottom)

                y0 = 175
                step = 26
                cv2.putText(frame, f"bottom mean/min/max: {bottom_mean:.1f} / {bottom_min:.1f} / {bottom_max:.1f}",
                            (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
                cv2.putText(frame, f"top    mean/min/max: {top_mean:.1f} / {top_min:.1f} / {top_max:.1f}",
                            (10, y0 + step), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)
                cv2.putText(frame, f"suggested top/bottom: {suggested_top:.1f} / {suggested_bottom:.1f}",
                            (10, y0 + 2 * step), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 0), 2)

        if paused:
            cv2.putText(frame, "PAUSED", (10, frame_h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Pause/Play button (top-right)
        btn_w, btn_h = 150, 50
        margin = 10
        x2 = frame_w - margin
        y1 = margin
        x1 = x2 - btn_w
        y2 = y1 + btn_h
        button_rect = (x1, y1, x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
        label = "PLAY" if paused else "PAUSE"
        cv2.putText(frame, label, (x1 + 15, y1 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            paused = not paused
        if args.auto:
            if key == ord("1"):
                auto_state = "LOCKED"
                locked_mode = "pullup"
                detect_start_time = None
                detect_shoulder_vals = []
                detect_elbow_vals = []
                last_detect_shoulder_range = 0.0
                last_detect_elbow_range = 0.0
                pullup_counter = RepCounterPullup()
                dip_counter = RepCounterDip()
                last_motion_time = time.time()
                last_rep_count = 0
                last_state = "WAITING_FOR_REP"
                last_overlay_mode = "pullup"
            elif key == ord("2"):
                auto_state = "LOCKED"
                locked_mode = "dip"
                detect_start_time = None
                detect_shoulder_vals = []
                detect_elbow_vals = []
                last_detect_shoulder_range = 0.0
                last_detect_elbow_range = 0.0
                pullup_counter = RepCounterPullup()
                dip_counter = RepCounterDip()
                last_motion_time = time.time()
                last_rep_count = 0
                last_state = "WAITING_FOR_REP"
                last_overlay_mode = "dip"
            elif key == ord("0"):
                auto_state = "IDLE"
                locked_mode = None
                detect_start_time = None
                detect_shoulder_vals = []
                detect_elbow_vals = []
                last_detect_shoulder_range = 0.0
                last_detect_elbow_range = 0.0
                pullup_counter = RepCounterPullup()
                dip_counter = RepCounterDip()
                last_motion_time = None
                last_rep_count = 0
                last_state = "IDLE"
                last_overlay_mode = "auto"
            last_auto_state = auto_state
            last_locked_mode = locked_mode
        if args.calibrate:
            if args.mode == "pullup":
                if key == ord("b") and counter.smooth_y is not None:
                    bottom_samples.append(counter.smooth_y)
                elif key == ord("t") and counter.smooth_y is not None:
                    top_samples.append(counter.smooth_y)
            elif args.mode == "dip":
                if key == ord("b") and counter.smooth_angle is not None:
                    bottom_samples.append(counter.smooth_angle)
                elif key == ord("t") and counter.smooth_angle is not None:
                    top_samples.append(counter.smooth_angle)

            if key == ord("r"):
                bottom_samples = []
                top_samples = []
                last_suggested = None
                last_suggested_msg = None
            elif key == ord("q"):
                break
        else:
            if key == ord("q"):
                break

        now = time.time()
        if args.calibrate and last_suggested is not None:
            if args.mode == "pullup":
                msg = (f"Suggested thresholds: top={last_suggested[0]:.3f}, "
                       f"bottom={last_suggested[1]:.3f}")
            else:
                msg = (f"Suggested dip thresholds: top_angle={last_suggested[0]:.1f}, "
                       f"bottom_angle={last_suggested[1]:.1f}")
            if msg != last_suggested_msg or (now - last_suggested_time) >= 1.0:
                print(msg)
                last_suggested_msg = msg
                last_suggested_time = now

    cap.release()
    cv2.destroyAllWindows()
    print("Finished.")


if __name__ == "__main__":
    main()
