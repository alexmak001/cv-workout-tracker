import time
import math
import argparse

import cv2
from ultralytics import YOLO
import numpy as np


# ===============================
# CONFIG / TUNING PARAMETERS
# ===============================

# --- Pose & detection ---
MIN_KEYPOINT_CONFIDENCE = 0.4   # TUNE_ME: raise if jittery/missing keypoints

# --- Smoothing ---
SMOOTHING_ALPHA = 0.2           # TUNE_ME: 0.1–0.3; higher = smoother but more lag

# --- Pull-up thresholds (normalized left shoulder Y in [0,1]) ---
# IMPORTANT: Y increases downward in images.
# So "top" (chin over bar) = smaller y, "bottom" (dead hang) = larger y.
PULLUP_BOTTOM_THRESH = 0.65     # TUNE_ME: y_norm near the bottom position
PULLUP_TOP_THRESH = 0.45        # TUNE_ME: y_norm near the top position

# --- Dip thresholds (left elbow angle in degrees) ---
# Typical: top of dip ~160–180°, bottom ~70–110° (depends on your form)
DIP_TOP_ANGLE = 150.0           # TUNE_ME: angle at/near lockout
DIP_BOTTOM_ANGLE = 100.0        # TUNE_ME: angle at deepest part

# --- Rep debounce ---
MIN_TIME_BETWEEN_REPS = 0.35    # TUNE_ME: seconds; avoid double-counting fast jitters


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

    cv2.putText(frame, text_mode, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, text_reps, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, text_state, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if extra_value is not None:
        if mode == "pullup":
            t = f"Left shoulder y_norm: {extra_value:.3f}"
        else:
            t = f"Left elbow angle: {extra_value:.1f} deg"
        cv2.putText(frame, t, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    return frame


def main():
    parser = argparse.ArgumentParser(description="YOLO-based rep counter (pullups & dips, LEFT-side view).")
    parser.add_argument("--mode", choices=["pullup", "dip"], required=True,
                        help="Exercise mode: pullup or dip")
    parser.add_argument("--model", type=str, default="yolov8n-pose.pt",
                        help="Path to YOLO pose model (e.g., yolov8n-pose.pt)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file. If not set, use webcam 0.")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    if args.video is None:
        cap = cv2.VideoCapture(0)  # TUNE_ME: change index if wrong camera
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    if args.mode == "pullup":
        counter = RepCounterPullup()
    else:
        counter = RepCounterDip()

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]

        results = model(frame, verbose=False)
        result = results[0]

        kp_dict = get_keypoints_dict(result)

        now = time.time()
        extra_val = None

        if args.mode == "pullup":
            y_norm = None
            if kp_dict is not None:
                y_norm = compute_left_shoulder_y_norm(kp_dict, frame_h)
            extra_val = y_norm
            rep_count, state = counter.update(y_norm, now)

        else:  # dip
            angle = None
            if kp_dict is not None:
                angle = angle_at_left_elbow(kp_dict)
            extra_val = angle
            rep_count, state = counter.update(angle, now)

        # Draw left-side keypoints for debugging
        if result.keypoints is not None and len(result.keypoints) > 0:
            for person in result.keypoints.xy:
                for x, y in person:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        frame = draw_overlay(frame, args.mode, rep_count, state, extra_value=extra_val)

        cv2.imshow("Rep Counter (Left Side)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Finished.")


if __name__ == "__main__":
    main()
