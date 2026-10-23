"""
Gaze Direction Estimation using iris position relative to eye bounding box.
Returns: 'center', 'left', 'right', 'up', 'down'

───────────────────────────────────────────────────────────────
WHY BOTH CORRECTIONS ARE NEEDED (important for future maintainers)
───────────────────────────────────────────────────────────────
main.py does cv2.flip(frame,1) BEFORE MediaPipe, so the frame is mirrored.

In a mirrored frame, the outer/inner corner x-order is REVERSED between eyes:
  LEFT eye:  outer(263).x > inner(362).x  → denominator POSITIVE
  RIGHT eye: outer(33).x  < inner(133).x  → denominator NEGATIVE

This means raw ratio_x pulls in OPPOSITE directions for the two eyes
when the user moves their gaze. Averaging them always gives ~0.5 → "center".

FIX STEP 1 — per-eye normalisation
  Flip ratio_x for the LEFT eye (1 - ratio_x).
  Now BOTH eyes produce a LARGE ratio_x when the user looks LEFT,
  and a SMALL ratio_x when the user looks RIGHT.

FIX STEP 2 — label swap at the classification step
  After step 1, small avg_x still maps to the WRONG label under the naive
  convention ("left") because of the mirror flip.
  Swapping the labels gives the correct user-centric result:
    small avg_x → "right"
    large avg_x → "left"
    small avg_y → "down"
    large avg_y → "up"
"""

import numpy as np
from collections import deque, Counter


# ── Sensitivity ──────────────────────────────────────────────────────
GAZE_H_THRESH    = 0.35   # horizontal threshold (symmetric for left/right)
GAZE_V_DOWN_THRESH = 0.32  # avg_y below this → looking DOWN
GAZE_V_UP_THRESH   = 0.60  # avg_y above this → looking UP
# Note: UP threshold is lower (0.60 vs 1-0.32=0.68) because looking UP
# moves the iris less than looking DOWN (upper eyelid covers the iris).

# ── Landmark indices (MediaPipe 478-point mesh) ───────────────────────
LEFT_EYE_OUTER   = 263   # anatomical left eye, temporal canthus
LEFT_EYE_INNER   = 362   # anatomical left eye, medial canthus
RIGHT_EYE_OUTER  = 33    # anatomical right eye, temporal canthus
RIGHT_EYE_INNER  = 133   # anatomical right eye, medial canthus

LEFT_EYE_TOP     = 386
LEFT_EYE_BOTTOM  = 374
RIGHT_EYE_TOP    = 159
RIGHT_EYE_BOTTOM = 145


class GazeEstimator:
    def __init__(self):
        self.gaze_history = deque(maxlen=10)
        self.raw_ratio_x  = 0.5
        self.raw_ratio_y  = 0.5

    def estimate(self, landmarks, left_iris, right_iris):
        """Estimate gaze direction from iris positions."""
        if left_iris is None or right_iris is None:
            return "unknown", 0.5, 0.5

        ratios_x, ratios_y = [], []

        for iris, outer_idx, inner_idx, top_idx, bot_idx, invert_x in [
            # LEFT eye: denominator is positive, which produces opposite ratio direction
            # vs right eye → must invert so both eyes agree before averaging.
            (left_iris,  LEFT_EYE_OUTER,  LEFT_EYE_INNER,
             LEFT_EYE_TOP,    LEFT_EYE_BOTTOM,  True),
            # RIGHT eye: denominator is negative (outer.x < inner.x in mirrored frame).
            # ratio_x already moves in the "correct" (from our label-swapped convention)
            # direction — do not invert.
            (right_iris, RIGHT_EYE_OUTER, RIGHT_EYE_INNER,
             RIGHT_EYE_TOP,   RIGHT_EYE_BOTTOM, False),
        ]:
            iris_c = iris[:, :2].mean(axis=0)
            outer  = landmarks[outer_idx, :2]
            inner  = landmarks[inner_idx, :2]
            top    = landmarks[top_idx,   :2]
            bottom = landmarks[bot_idx,   :2]

            eye_w = float(np.linalg.norm(outer - inner))
            eye_h = float(np.linalg.norm(top   - bottom))
            if eye_w < 2 or eye_h < 2:
                continue

            dx = float(outer[0]) - float(inner[0])
            dy = float(bottom[1]) - float(top[1])
            if abs(dx) < 1e-6 or abs(dy) < 1e-6:
                continue

            rx = (float(iris_c[0]) - float(inner[0])) / dx
            ry = (float(iris_c[1]) - float(top[1]))   / dy

            # Clamp — iris should remain within the eye bounding box
            rx = max(-0.1, min(1.1, rx))
            ry = max(-0.1, min(1.1, ry))

            # ── Step 1: normalise left eye so both eyes agree ──────
            if invert_x:
                rx = 1.0 - rx

            ratios_x.append(rx)
            ratios_y.append(ry)

        if not ratios_x:
            return "unknown", 0.5, 0.5

        avg_x = float(np.mean(ratios_x))
        avg_y = float(np.mean(ratios_y))
        self.raw_ratio_x = avg_x
        self.raw_ratio_y = avg_y

        # ── Step 2: classify with swapped labels ───────────────────
        # See module docstring for why labels are swapped.
        direction = "center"
        if avg_x < GAZE_H_THRESH:
            direction = "right"          # small ratio_x → user looking RIGHT
        elif avg_x > (1.0 - GAZE_H_THRESH):
            direction = "left"           # large ratio_x → user looking LEFT
        elif avg_y < GAZE_V_DOWN_THRESH:
            direction = "down"           # small ratio_y → user looking DOWN
        elif avg_y > GAZE_V_UP_THRESH:
            direction = "up"             # large ratio_y → user looking UP

        self.gaze_history.append(direction)
        return direction, avg_x, avg_y

    def smoothed_direction(self):
        if not self.gaze_history:
            return "center"
        return Counter(self.gaze_history).most_common(1)[0][0]

    def is_looking_away(self):
        if len(self.gaze_history) < 5:
            return False
        away = sum(1 for g in self.gaze_history if g not in ("center", "unknown"))
        return away >= len(self.gaze_history) * 0.55
