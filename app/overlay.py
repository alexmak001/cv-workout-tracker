import cv2


def draw_overlay(frame, mode, rep_count, state, extra_value=None):
    """
    Draw basic info on frame.
    extra_value: y_norm for pullups or angle for dips, for debugging.
    """
    h, w = frame.shape[:2]

    # translucent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 140), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    text_mode = f"Mode: {mode.upper()} (LEFT SIDE VIEW)"
    text_reps = f"Reps: {rep_count}"
    text_state = f"State: {state}"

    cv2.putText(frame, text_mode, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
    cv2.putText(frame, text_reps, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 0), 4)
    cv2.putText(frame, text_state, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    if extra_value is not None:
        if mode == "pullup":
            t = f"Left shoulder y_norm: {extra_value:.3f}"
            cv2.putText(frame, t, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 200, 0), 4)
        elif mode == "dip":
            t = f"Left elbow angle: {extra_value:.1f} deg"
            cv2.putText(frame, t, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 200, 0), 4)

    return frame


def draw_paused_label(frame):
    h, _ = frame.shape[:2]
    cv2.putText(frame, "PAUSED", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 4)


def draw_pause_button(frame, paused):
    """Draw the pause/play button and return its rectangle."""
    frame_h, frame_w = frame.shape[:2]
    btn_w, btn_h = 180, 60
    margin = 10
    x2 = frame_w - margin
    y1 = margin
    x1 = x2 - btn_w
    y2 = y1 + btn_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 30, 30), -1)
    label = "PLAY" if paused else "PAUSE"
    cv2.putText(frame, label, (x1 + 18, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return (x1, y1, x2, y2)
