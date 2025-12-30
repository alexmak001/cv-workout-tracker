from pathlib import Path

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
PULLUP_TOP_THRESH = 0.33        # above your top band
PULLUP_BOTTOM_THRESH = 0.38     # below your bottom band

# --- Dip thresholds (left elbow angle in degrees) ---
# Typical: top of dip ~160–180°, bottom ~70–110° (depends on your form)
DIP_TOP_ANGLE = 150.0           # TUNE_ME: angle at/near lockout
DIP_BOTTOM_ANGLE = 90.0         # TUNE_ME: angle at deepest part

# --- Rep debounce ---
MIN_TIME_BETWEEN_REPS = 0.35    # TUNE_ME: seconds; avoid double-counting fast jitters

# --- Auto-detect tuning ---
WS_PULLUP_THRESH = 0.05         # TUNE_ME: wrist above shoulder
WS_DIP_THRESH = -0.05           # TUNE_ME: wrist below shoulder
SWITCH_CONFIRM_FRAMES = 10      # TUNE_ME: frames required to confirm a switch
WS_MOTION_EPS = 0.01            # TUNE_ME: ws delta that counts as motion
REST_SECONDS = 1.0              # TUNE_ME: time without motion before switching
