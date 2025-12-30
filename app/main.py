import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from . import overlay
from .config import (
    DEFAULT_MODEL_PATH,
    REST_SECONDS,
    WS_DIP_THRESH,
    WS_MOTION_EPS,
    WS_PULLUP_THRESH,
    SWITCH_CONFIRM_FRAMES,
)
from .export import Exporter, HomeAssistantExporterStub, MQTTExporterStub, NoOpExporter
from .pose import (
    angle_at_left_elbow,
    compute_left_shoulder_y_norm,
    compute_ws,
    ensure_model_path,
    get_keypoints_dict,
)
from .rep_counter import RepCounterDip, RepCounterPullup
from .tts import PersistentChatterboxAnnouncer


def build_exporter(args) -> Exporter:
    if not args.export:
        return NoOpExporter()

    target = (args.export_target or "").lower()
    if target == "homeassistant":
        return HomeAssistantExporterStub(enabled=True, target=target, destination=args.export_destination)
    if target == "mqtt":
        return MQTTExporterStub(enabled=True, target=target, destination=args.export_destination)
    return Exporter(enabled=True, target=target or None, destination=args.export_destination)


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
    parser.add_argument("--tts", action="store_true",
                        help="Enable TTS via persistent worker.")
    parser.add_argument("--tts-python", type=str, default=".venv_tts/bin/python",
                        help="Python path for the TTS environment.")
    parser.add_argument("--tts-cooldown", type=float, default=0.4,
                        help="Minimum seconds between TTS launches.")
    parser.add_argument("--tts-model", choices=["chatterbox", "turbo"], default="turbo",
                        help="TTS model selection.")
    parser.add_argument("--tts-audio-prompt", type=str, default=None,
                        help="Optional audio prompt for turbo mode.")
    parser.add_argument("--tts-test", type=str, default=None,
                        help="Speak this text once and exit.")
    parser.add_argument("--export", action="store_true",
                        help="Enable stub export logging.")
    parser.add_argument("--export-target", type=str, default=None,
                        help="Export target (homeassistant or mqtt).")
    parser.add_argument("--export-destination", type=str, default=None,
                        help="Export destination (unused for now).")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug overlay/logs.")
    args = parser.parse_args()

    window_name = "Rep Counter (Left Side)"
    paused = False
    button_rect = (0, 0, 0, 0)

    def on_mouse(event, x, y, flags, userdata):
        nonlocal paused
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1, x2, y2 = button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                paused = not paused

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    if not args.auto and args.mode is None:
        parser.error("--mode is required unless --auto is set.")

    announcer = PersistentChatterboxAnnouncer(
        enabled=args.tts,
        tts_python=args.tts_python,
        model=args.tts_model,
        audio_prompt=args.tts_audio_prompt,
        cooldown_s=args.tts_cooldown,
    )
    if args.tts and args.tts_test:
        announcer.start_nonblocking()
        announcer.on_rep(args.tts_test, 1)
        announcer.stop()
        return

    model_path = ensure_model_path(Path(args.model))
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    def open_camera(camera_index, backend_choice):
        if backend_choice == "avfoundation" and sys.platform == "darwin":
            print(f"Opening camera index {camera_index} with backend=avfoundation")
            return cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        print(f"Opening camera index {camera_index} with backend={backend_choice}")
        return cv2.VideoCapture(camera_index)

    if args.video is None:
        backend_choice = "avfoundation" if sys.platform == "darwin" else "default"
        if args.backend is not None:
            backend_choice = args.backend
        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            print("AVFoundation open failed, retrying with default backend...")
            cap = open_camera(args.camera, backend_choice)
    else:
        cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    try:
        backend_name = cap.getBackendName()
    except Exception:
        backend_name = "unknown"
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"cap.isOpened()={cap.isOpened()} backend={backend_name} size={int(width)}x{int(height)} fps={fps:.1f}")
    if args.video is None:
        start_warmup = time.time()
        warmup_frames = 0
        while warmup_frames < 20 and (time.time() - start_warmup) < 1.0:
            ret, _ = cap.read()
            if ret:
                warmup_frames += 1
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if args.auto:
        if args.mode in ("pullup", "dip"):
            counter = RepCounterPullup() if args.mode == "pullup" else RepCounterDip()
    else:
        if args.mode == "pullup":
            counter = RepCounterPullup()
        else:
            counter = RepCounterDip()

    exporter = build_exporter(args)

    paused = False
    if args.calibrate:
        print("Calibration mode: press 'b' (bottom), 't' (top), 'r' (reset), 'q' (quit).")
    if args.auto:
        print("Auto mode: press '1' for pullup, '2' for dip, '0' to return to auto-detect.")
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
    current_exercise = None
    candidate_exercise = None
    candidate_count = 0
    active_counter = None
    active_mode = "auto"
    last_ws = None
    last_motion_time = None
    last_overlay_mode = "auto" if args.auto else args.mode
    last_current_exercise = current_exercise
    last_candidate_exercise = candidate_exercise
    last_candidate_count = candidate_count
    last_locked_moving = False
    last_active_mode = active_mode
    tts_started = False
    frames_ok = 0
    consecutive_failures = 0
    last_fail_log = 0.0

    try:
        while True:
            ret = True
            frame = None
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    consecutive_failures += 1
                    now = time.time()
                    if (now - last_fail_log) > 1.0:
                        print("cap.read() failed; keeping window open; press q to quit")
                        last_fail_log = now
                    if args.video is None and consecutive_failures > 30:
                        print("Reopening camera after 30 consecutive failures...")
                        cap.release()
                        cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
                        if not cap.isOpened():
                            print("AVFoundation reopen failed, retrying default backend...")
                            cap = open_camera(args.camera, backend_choice)
                        consecutive_failures = 0
                    last_frame = None
                    last_result = None
                    last_kp_dict = None
                    last_extra_val = None
                else:
                    consecutive_failures = 0
                    frame_h, frame_w = frame.shape[:2]
                    last_frame = frame.copy()
                    frames_ok += 1
                    if args.tts and not tts_started and frames_ok >= 5:
                        print(f"Starting TTS worker after {frames_ok} good frames")
                        announcer.start_nonblocking()
                        tts_started = True

                    results = model(frame, verbose=False)
                    result = results[0]
                    last_result = result

                    kp_dict = get_keypoints_dict(result)
                    last_kp_dict = kp_dict

                    now = time.time()
                    extra_val = None
                    y_norm = None
                    angle = None
                    ws = None
                    if kp_dict is not None:
                        y_norm = compute_left_shoulder_y_norm(kp_dict, frame_h)
                        angle = angle_at_left_elbow(kp_dict)
                        ws = compute_ws(kp_dict, frame_h)

                    if args.auto:
                        if ws is not None:
                            if last_ws is not None and abs(ws - last_ws) > WS_MOTION_EPS:
                                last_motion_time = now
                            last_ws = ws

                            if ws >= WS_PULLUP_THRESH:
                                inst = "pullup"
                            elif ws <= WS_DIP_THRESH:
                                inst = "dip"
                            else:
                                inst = None

                            if inst is None:
                                pass
                            elif inst != current_exercise:
                                if inst == candidate_exercise:
                                    candidate_count += 1
                                else:
                                    candidate_exercise = inst
                                    candidate_count = 1
                            else:
                                candidate_exercise = None
                                candidate_count = 0

                            rest_ok = last_motion_time is None or (now - last_motion_time) >= REST_SECONDS
                            if (
                                candidate_exercise is not None
                                and candidate_count >= SWITCH_CONFIRM_FRAMES
                                and rest_ok
                            ):
                                current_exercise = candidate_exercise
                                candidate_exercise = None
                                candidate_count = 0
                                if current_exercise == "pullup":
                                    active_counter = RepCounterPullup()
                                    active_mode = "pullup"
                                    rep_count = 0
                                    state = "WAITING_FOR_REP"
                                elif current_exercise == "dip":
                                    active_counter = RepCounterDip()
                                    active_mode = "dip"
                                    rep_count = 0
                                    state = "WAITING_FOR_REP"

                        if current_exercise == "pullup":
                            extra_val = y_norm
                            if active_counter is None:
                                active_counter = RepCounterPullup()
                            active_mode = "pullup"
                            rep_count, state, rep_inc = active_counter.update(y_norm, now)
                            if rep_inc:
                                announcer.on_rep("pullup", rep_count)
                                exporter.on_rep("pullup", rep_count, now)
                            last_overlay_mode = "pullup"
                        elif current_exercise == "dip":
                            extra_val = angle
                            if active_counter is None:
                                active_counter = RepCounterDip()
                            active_mode = "dip"
                            rep_count, state, rep_inc = active_counter.update(angle, now)
                            if rep_inc:
                                announcer.on_rep("dip", rep_count)
                                exporter.on_rep("dip", rep_count, now)
                            last_overlay_mode = "dip"
                        else:
                            active_mode = "auto"
                            rep_count = 0
                            state = "UNKNOWN"
                            last_overlay_mode = "auto"
                    else:
                        if args.mode == "pullup":
                            extra_val = y_norm
                            rep_count, state, rep_inc = counter.update(y_norm, now)
                            if rep_inc:
                                announcer.on_rep("pullup", rep_count)
                                exporter.on_rep("pullup", rep_count, now)
                        else:  # dip
                            extra_val = angle
                            rep_count, state, rep_inc = counter.update(angle, now)
                            if rep_inc:
                                announcer.on_rep("dip", rep_count)
                                exporter.on_rep("dip", rep_count, now)
                        last_overlay_mode = args.mode

                    last_extra_val = extra_val
                    last_rep_count = rep_count
                    last_state = state
                    last_active_mode = active_mode
                    last_current_exercise = current_exercise
                    last_candidate_exercise = candidate_exercise
                    last_candidate_count = candidate_count
                    last_locked_moving = (
                        last_motion_time is not None and (now - last_motion_time) < REST_SECONDS
                    )

            if last_frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                frame_h, frame_w = frame.shape[:2]
                rep_count = last_rep_count
                state = last_state
                extra_val = last_extra_val
                result = None
                kp_dict = None
                overlay_mode = last_overlay_mode
                current_exercise = last_current_exercise
                candidate_exercise = last_candidate_exercise
                candidate_count = last_candidate_count
                locked_moving = last_locked_moving
                active_mode = last_active_mode
                cv2.putText(frame, "NO FRAME", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            else:
                frame = last_frame.copy()
                frame_h, frame_w = frame.shape[:2]
                rep_count = last_rep_count
                state = last_state
                extra_val = last_extra_val
                result = last_result
                kp_dict = last_kp_dict
                overlay_mode = last_overlay_mode
                current_exercise = last_current_exercise
                candidate_exercise = last_candidate_exercise
                candidate_count = last_candidate_count
                locked_moving = last_locked_moving
                active_mode = last_active_mode

                # Draw left-side keypoints for debugging
                if result is not None and result.keypoints is not None and len(result.keypoints) > 0:
                    for person in result.keypoints.xy:
                        for x, y in person:
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

            frame = overlay.draw_overlay(frame, overlay_mode, rep_count, state, extra_value=extra_val)

            if args.auto:
                cv2.putText(frame, "Auto: ACTIVE", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                detected = current_exercise.upper() if current_exercise else "UNKNOWN"
                cv2.putText(frame, f"Detected: {detected}", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                ws_text = "n/a" if last_ws is None else f"{last_ws:.3f}"
                cv2.putText(frame,
                            f"ws: {ws_text} (pu>={WS_PULLUP_THRESH:.2f} dip<={WS_DIP_THRESH:.2f})",
                            (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cand = candidate_exercise.upper() if candidate_exercise else "NONE"
                cv2.putText(frame, f"candidate: {cand} ({candidate_count})",
                            (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                if locked_moving:
                    cv2.putText(frame, "LOCKED (moving)", (10, 310),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            if args.debug:
                debug_text = f"ret={ret} shape={getattr(frame, 'shape', None)} t={time.time():.2f}"
                cv2.putText(frame, debug_text, (10, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # --- Calibration-only behavior (pullups) ---
            if args.calibrate and args.mode == "pullup":
                smooth_y = counter.smooth_y
                if smooth_y is not None:
                    cv2.putText(
                        frame,
                        f"y_norm (smoothed): {smooth_y:.3f}",
                        (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
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
    
                    y0 = 220
                    step = 34
                    cv2.putText(frame, f"bottom mean/min/max: {bottom_mean:.3f} / {bottom_min:.3f} / {bottom_max:.3f}",
                                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 255, 200), 2)
                    cv2.putText(frame, f"top    mean/min/max: {top_mean:.3f} / {top_min:.3f} / {top_max:.3f}",
                                (10, y0 + step), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 255, 200), 2)
                    cv2.putText(frame, f"suggested top/bottom: {suggested_top:.3f} / {suggested_bottom:.3f}",
                                (10, y0 + 2 * step), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 255, 0), 2)
            elif args.calibrate and args.mode == "dip":
                smooth_angle = counter.smooth_angle
                if smooth_angle is not None:
                    cv2.putText(
                        frame,
                        f"elbow angle (smoothed): {smooth_angle:.1f} deg",
                        (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
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
    
                    y0 = 220
                    step = 34
                    cv2.putText(frame, f"bottom mean/min/max: {bottom_mean:.1f} / {bottom_min:.1f} / {bottom_max:.1f}",
                                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 255, 200), 2)
                    cv2.putText(frame, f"top    mean/min/max: {top_mean:.1f} / {top_min:.1f} / {top_max:.1f}",
                                (10, y0 + step), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 255, 200), 2)
                    cv2.putText(frame, f"suggested top/bottom: {suggested_top:.1f} / {suggested_bottom:.1f}",
                                (10, y0 + 2 * step), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 255, 0), 2)
            if args.calibrate:
                cv2.putText(frame,
                            f"wrist tune: ws_pu>={WS_PULLUP_THRESH:.3f} ws_dip<={WS_DIP_THRESH:.3f}",
                            (10, frame_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
            if paused:
                overlay.draw_paused_label(frame)
    
            button_rect = overlay.draw_pause_button(frame, paused)
    
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                paused = not paused
            if args.auto:
                if key == ord("1"):
                    current_exercise = "pullup"
                    active_counter = RepCounterPullup()
                    active_mode = "pullup"
                    candidate_exercise = None
                    candidate_count = 0
                    last_rep_count = 0
                    last_state = "WAITING_FOR_REP"
                    last_overlay_mode = "pullup"
                elif key == ord("2"):
                    current_exercise = "dip"
                    active_counter = RepCounterDip()
                    active_mode = "dip"
                    candidate_exercise = None
                    candidate_count = 0
                    last_rep_count = 0
                    last_state = "WAITING_FOR_REP"
                    last_overlay_mode = "dip"
                elif key == ord("0"):
                    current_exercise = None
                    active_counter = None
                    active_mode = "auto"
                    candidate_exercise = None
                    candidate_count = 0
                    last_rep_count = 0
                    last_state = "IDLE"
                    last_overlay_mode = "auto"
                last_current_exercise = current_exercise
                last_candidate_exercise = candidate_exercise
                last_candidate_count = candidate_count
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
                    return
            else:
                if key == ord("q"):
                    return
    
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
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if args.tts:
            announcer.stop()
        print("Finished.")


if __name__ == "__main__":
    main()
