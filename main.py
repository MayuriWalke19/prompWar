import os
import sys

# Windows / UTF-8 console output safety
if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# macOS: allow OpenCV to access the camera from a background thread
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

"""
Flask backend for Real-Time Attention Tracker.
Endpoints:
  GET  /                    -> serves the dashboard HTML
  GET  /video_feed          -> MJPEG stream (fallback)
  GET  /api/frame           -> single JPEG frame (canvas polling)
  GET  /api/status          -> current attention state & stats (JSON)
  GET  /api/session_summary -> session analytics summary (JSON)
  GET  /api/timeline        -> attention timeline records (JSON)
  POST /api/reset_session   -> reset session data
  GET  /api/sessions        -> list past sessions
"""

import cv2
import csv
import io
import time
import threading
import numpy as np
from collections import deque
from flask import Flask, Response, jsonify, render_template, make_response
from flask_cors import CORS

from face_detection.detector import FaceDetector
from eye_tracking.ear         import BlinkDetector, compute_ear
from eye_tracking.gaze        import GazeEstimator
from head_pose.estimator      import HeadPoseEstimator
from attention_logic.classifier import AttentionClassifier
from utils.drawing import (
    draw_face_box, draw_eye_landmarks, draw_iris,
    draw_attention_label, draw_stats_panel, draw_no_face, draw_fps,
)
from utils.session_logger import SessionLogger


# ── App setup ───────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)


# ── Global processing state ─────────────────────────────────────────
class ProcessingState:
    def __init__(self):
        self.lock          = threading.Lock()
        self.frame_bytes   = None
        self.status        = {
            "state":         "No Face",
            "confidence":    0.0,
            "ear":           0.0,
            "blink_rate":    0,
            "gaze":          "unknown",
            "yaw":           0.0,
            "pitch":         0.0,
            "roll":          0.0,
            "fps":           0.0,
            "face_detected": False,
        }
        self.running = False


state = ProcessingState()

# Detector singletons (initialised in the camera thread)
face_detector  = None
blink_detector = None
gaze_estimator = None
pose_estimator = None
classifier     = None
session_logger = None


def processing_loop():
    global face_detector, blink_detector, gaze_estimator
    global pose_estimator, classifier, session_logger

    face_detector  = FaceDetector(max_faces=3)  # detect up to 3 faces
    blink_detector = BlinkDetector()
    gaze_estimator = GazeEstimator()
    pose_estimator = HeadPoseEstimator()
    classifier     = AttentionClassifier()
    session_logger = SessionLogger()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,           30)

    prev_time     = time.time()
    timeline_tick = time.time()
    frame_count   = 0
    fps_samples   = deque(maxlen=10)

    while state.running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)

        now = time.time()
        fps_samples.append(1.0 / max(now - prev_time, 1e-6))
        fps = float(np.mean(fps_samples))
        prev_time = now
        frame_count += 1

        results   = face_detector.process(frame)
        landmarks = face_detector.get_landmarks(results, frame.shape)

        if landmarks is not None:
            bbox = face_detector.get_face_bbox(landmarks, frame.shape)

            left_eye, right_eye = face_detector.get_eye_landmarks(landmarks)
            left_ear  = compute_ear(left_eye)
            right_ear = compute_ear(right_eye)
            avg_ear   = (left_ear + right_ear) / 2.0
            blink_detector.update(avg_ear)

            left_iris, right_iris = face_detector.get_iris_landmarks(landmarks)
            gaze_dir, gx, gy = gaze_estimator.estimate(landmarks, left_iris, right_iris)

            # Head pose every 3rd frame to save CPU
            if frame_count % 3 == 0:
                pitch, yaw, roll, rvec, tvec = pose_estimator.estimate(landmarks, frame.shape)
            else:
                pitch, yaw, roll = pose_estimator.smoothed_angles()

            ear_stats  = blink_detector.get_stats()
            gaze_stats = {
                "looking_away": gaze_estimator.is_looking_away(),
                "direction":    gaze_dir,
            }
            pose_stats = pose_estimator.get_stats()
            attn_state, confidence, signals = classifier.classify(
                ear_stats, gaze_stats, pose_stats
            )

            session_logger.record(
                attn_state, confidence, avg_ear,
                ear_stats["blink_rate_per_min"],
                gaze_dir, yaw, pitch, roll,
            )

            if now - timeline_tick >= 1.0:
                classifier.record_timeline_point()
                timeline_tick = now

            draw_face_box(frame, bbox, attn_state)
            draw_eye_landmarks(frame, left_eye)
            draw_eye_landmarks(frame, right_eye)
            if left_iris is not None:
                draw_iris(frame, left_iris)
                draw_iris(frame, right_iris)
            draw_attention_label(frame, attn_state, confidence)
            draw_stats_panel(
                frame, avg_ear,
                ear_stats["blink_rate_per_min"],
                gaze_dir, pitch, yaw, roll,
            )
            draw_fps(frame, fps)

            # Draw bystander faces (grey box, no analysis)
            secondary_bboxes = face_detector.get_secondary_face_bboxes(
                results, frame.shape, landmarks
            )
            for sbbox in secondary_bboxes:
                sx1, sy1, sx2, sy2 = sbbox
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (100, 100, 100), 1)
                cv2.putText(
                    frame, "Bystander",
                    (sx1 + 4, sy1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (140, 140, 140), 1, cv2.LINE_AA,
                )

            with state.lock:
                state.status = {
                    "state":         attn_state,
                    "confidence":    confidence,
                    "ear":           round(float(avg_ear), 3),
                    "blink_rate":    ear_stats["blink_rate_per_min"],
                    "gaze":          gaze_dir,
                    "yaw":           round(float(yaw),   1),
                    "pitch":         round(float(pitch), 1),
                    "roll":          round(float(roll),  1),
                    "fps":           round(float(fps),   1),
                    "face_detected": True,
                    "face_count":    1 + len(secondary_bboxes),
                }
        else:
            draw_no_face(frame)
            draw_fps(frame, fps)
            with state.lock:
                state.status["face_detected"] = False
                state.status["state"]         = "No Face"
                state.status["fps"]           = round(fps, 1)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with state.lock:
            state.frame_bytes = jpeg.tobytes()

    cap.release()
    face_detector.close()
    if session_logger:
        session_logger.save()


def generate_frames():
    """MJPEG generator (kept as fallback)."""
    while True:
        with state.lock:
            frame = state.frame_bytes
        if frame is None:
            time.sleep(0.033)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.033)


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/frame")
def api_frame():
    """Return the latest processed frame as a single JPEG (canvas polling)."""
    with state.lock:
        frame = state.frame_bytes
    if frame is None:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode(".jpg", blank)
        frame = jpeg.tobytes()
    resp = Response(frame, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/status")
def api_status():
    with state.lock:
        return jsonify(state.status)


@app.route("/api/session_summary")
def api_session_summary():
    if classifier is None:
        return jsonify({})
    return jsonify(classifier.session_summary())


@app.route("/api/timeline")
def api_timeline():
    if classifier is None:
        return jsonify([])
    return jsonify(classifier.get_timeline())


@app.route("/api/reset_session", methods=["POST"])
def api_reset_session():
    if classifier:
        classifier.reset_session()
    if blink_detector:
        blink_detector.blink_count = 0
        blink_detector.blink_times.clear()
    if session_logger:
        session_logger.save()
    return jsonify({"ok": True})


@app.route("/api/sessions")
def api_sessions():
    if session_logger is None:
        return jsonify([])
    return jsonify(session_logger.list_sessions())


@app.route("/api/export_csv")
def api_export_csv():
    """Export full session data as a downloadable CSV file."""
    if classifier is None:
        return jsonify({"error": "Session not started"}), 503

    summary  = classifier.session_summary()
    timeline = classifier.get_timeline()

    output = io.StringIO()
    writer = csv.writer(output)

    # ── Summary section ──────────────────────────────────────
    writer.writerow(["AttentionAI Session Report"])
    writer.writerow(["Generated", time.strftime("%Y-%m-%d %H:%M:%S")])
    writer.writerow([])
    writer.writerow(["=== SESSION SUMMARY ==="])
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Total Duration (s)",    summary.get("total_seconds", 0)])
    writer.writerow(["Focused (%)",            summary.get("focused_pct", 0)])
    writer.writerow(["Distracted (%)",         summary.get("distracted_pct", 0)])
    writer.writerow(["Drowsy (%)",             summary.get("drowsy_pct", 0)])
    writer.writerow(["Focused Time (s)",       summary.get("focused_seconds", 0)])
    writer.writerow(["Distracted Time (s)",    summary.get("distracted_seconds", 0)])
    writer.writerow(["Drowsy Time (s)",        summary.get("drowsy_seconds", 0)])
    writer.writerow(["Best Focus Streak (s)",  summary.get("best_focus_streak_sec", 0)])
    writer.writerow(["Longest Distraction (s)",summary.get("longest_distract_sec", 0)])
    writer.writerow(["Attention Score (0-100)",summary.get("attention_score", 0)])
    writer.writerow([])

    # ── Timeline section ──────────────────────────────────────
    writer.writerow(["=== ATTENTION TIMELINE ==="])
    writer.writerow(["Elapsed (s)", "State"])
    for point in timeline:
        writer.writerow([point.get("ts", ""), point.get("state", "")])

    csv_bytes = output.getvalue().encode("utf-8")
    filename  = time.strftime("attentionai_session_%Y%m%d_%H%M%S.csv")

    resp = make_response(csv_bytes)
    resp.headers["Content-Type"]        = "text/csv; charset=utf-8"
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return resp


# ── Startup ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    state.running = True
    t = threading.Thread(target=processing_loop, daemon=True)
    t.start()

    print("\n" + "=" * 55)
    print("  🧠  Attention Tracker  |  http://localhost:5050  ")
    print("=" * 55 + "\n")

    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
