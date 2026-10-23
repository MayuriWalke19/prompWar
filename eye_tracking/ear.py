"""
Eye Aspect Ratio (EAR) calculation and blink detection.
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
When eyes are open, EAR ≈ 0.25-0.35. When closed, EAR ≈ 0.0
"""

import numpy as np
from collections import deque
import time


EAR_THRESHOLD = 0.22       # Below this → eye considered closed
BLINK_FRAMES = 2           # Consecutive frames below threshold = blink
DROWSY_EAR_THRESHOLD = 0.18  # Sustained low EAR → drowsy
DROWSY_CONSEC_FRAMES = 20  # ~0.66s at 30fps sustained closure → drowsy


def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)


def compute_ear(eye_landmarks):
    """
    Compute EAR given 6 landmark points.
    Points order: [outer_corner, top1, top2, inner_corner, bottom1, bottom2]
    """
    p1, p2, p3, p4, p5, p6 = eye_landmarks[:, :2]
    vertical1 = euclidean(p2, p6)
    vertical2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return float(ear)


class BlinkDetector:
    def __init__(self, window_seconds=60):
        self.blink_count = 0
        self.consecutive_closed = 0
        self.eye_closed = False
        self.blink_times = deque()
        self.window_seconds = window_seconds
        self.ear_history = deque(maxlen=90)  # 3s at 30fps
        self.session_start = time.time()

    def update(self, ear):
        """Update blink state with current EAR value."""
        now = time.time()
        self.ear_history.append(ear)

        if ear < EAR_THRESHOLD:
            self.consecutive_closed += 1
        else:
            if self.consecutive_closed >= BLINK_FRAMES:
                self.blink_count += 1
                self.blink_times.append(now)
            self.consecutive_closed = 0

        # Remove old blinks outside the window
        cutoff = now - self.window_seconds
        while self.blink_times and self.blink_times[0] < cutoff:
            self.blink_times.popleft()

    def blink_rate_per_minute(self):
        """Blinks counted in the last minute."""
        now = time.time()
        recent = [t for t in self.blink_times if t > now - 60]
        return len(recent)

    def is_drowsy_by_closure(self):
        """Detect drowsiness by sustained eye closure."""
        return self.consecutive_closed >= DROWSY_CONSEC_FRAMES

    def avg_ear(self):
        """Average EAR over recent history."""
        if not self.ear_history:
            return 0.3
        return float(np.mean(self.ear_history))

    def get_stats(self):
        return {
            "blink_count": self.blink_count,
            "blink_rate_per_min": self.blink_rate_per_minute(),
            "consecutive_closed_frames": self.consecutive_closed,
            "is_drowsy_by_closure": self.is_drowsy_by_closure(),
            "avg_ear": round(self.avg_ear(), 3),
        }
