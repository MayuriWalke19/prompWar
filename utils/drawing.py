"""
Frame drawing utilities: overlay bounding boxes, landmarks, labels, gauges.
"""

import cv2
import numpy as np


# ── Color palettes (BGR) ────────────────────────────────────────────
COLORS = {
    "Focused":    (0,   210,  80),   # Vivid green
    "Distracted": (0,   200, 255),   # Amber/yellow
    "Drowsy":     (0,    50, 220),   # Red
    "white":      (255, 255, 255),
    "black":      (0,     0,   0),
    "gray":       (150, 150, 150),
    "iris":       (255, 180,  30),
    "landmark":   ( 80, 220, 255),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.45
FONT_MED   = 0.65
FONT_LARGE = 0.9


def draw_face_box(frame, bbox, state):
    x1, y1, x2, y2 = bbox
    color = COLORS.get(state, COLORS["white"])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner accents
    corner_len = 18
    t = 3
    for cx, cy, dx, dy in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, t)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, t)


def draw_eye_landmarks(frame, eye_landmarks, color=None):
    if color is None:
        color = COLORS["landmark"]
    pts = eye_landmarks[:, :2].astype(int)
    for pt in pts:
        cv2.circle(frame, tuple(pt), 2, color, -1)
    # Draw eye outline
    hull = cv2.convexHull(pts)
    cv2.polylines(frame, [hull], True, color, 1)


def draw_iris(frame, iris_landmarks, color=None):
    if color is None:
        color = COLORS["iris"]
    pts = iris_landmarks[:, :2].astype(int)
    center = pts.mean(axis=0).astype(int)
    radius = int(np.linalg.norm(pts - center, axis=1).mean()) + 1
    cv2.circle(frame, tuple(center), radius, color, 1)
    cv2.circle(frame, tuple(center), 2, color, -1)


def draw_attention_label(frame, state, confidence):
    color  = COLORS.get(state, COLORS["white"])
    h, w   = frame.shape[:2]

    # Background pill
    label  = f"{state}  {int(confidence * 100)}%"
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_LARGE, 2)
    pad    = 12
    x0     = w // 2 - tw // 2 - pad
    y0     = 18
    x1_b   = w // 2 + tw // 2 + pad
    y1_b   = y0 + th + pad * 2

    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1_b, y1_b), (20, 20, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.rectangle(frame, (x0, y0), (x1_b, y1_b), color, 2)
    cv2.putText(frame, label, (x0 + pad, y1_b - pad),
                FONT, FONT_LARGE, color, 2, cv2.LINE_AA)


def draw_stats_panel(frame, ear, blink_rate, gaze_dir, pitch, yaw, roll):
    """Draw a semi-transparent info panel on the left side."""
    h, w = frame.shape[:2]
    panel_w = 200
    panel_h = 175
    x0, y0 = 10, 60

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), COLORS["gray"], 1)

    lines = [
        f"EAR:      {ear:.3f}",
        f"Blinks:   {blink_rate}/min",
        f"Gaze:     {gaze_dir}",
        f"Yaw:      {yaw:.1f}°",
        f"Pitch:    {pitch:.1f}°",
        f"Roll:     {roll:.1f}°",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0 + 8, y0 + 22 + i * 24),
                    FONT, FONT_SMALL, COLORS["white"], 1, cv2.LINE_AA)


def draw_no_face(frame):
    h, w = frame.shape[:2]
    msg = "No Face Detected"
    (tw, th), _ = cv2.getTextSize(msg, FONT, FONT_MED, 2)
    cv2.putText(frame, msg,
                (w // 2 - tw // 2, h // 2),
                FONT, FONT_MED, COLORS["Drowsy"], 2, cv2.LINE_AA)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                FONT, FONT_SMALL, COLORS["gray"], 1, cv2.LINE_AA)
