"""
Head Pose Estimation using facial landmarks and solvePnP.
Estimates Yaw (left-right), Pitch (up-down), Roll (tilt) angles.
Uses direct Rodrigues → rotation-matrix → Euler decomposition (reliable scalar output).
"""

import cv2
import numpy as np
from collections import deque


# 3D model reference points (standard face model in mm)
MODEL_POINTS = np.array([
    (0.0,     0.0,    0.0),    # Nose tip
    (0.0,  -330.0,  -65.0),   # Chin
    (-225.0, 170.0, -135.0),  # Left corner of left eye
    (225.0,  170.0, -135.0),  # Right corner of right eye
    (-150.0,-150.0, -125.0),  # Left corner of mouth
    (150.0, -150.0, -125.0),  # Right corner of mouth
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
POSE_LANDMARK_IDS = [1, 152, 226, 446, 57, 287]

# Thresholds (degrees) — tuned for realistic head-turn detection
YAW_THRESHOLD   = 25  # left/right head turn
PITCH_THRESHOLD = 20  # up/down head nod
ROLL_THRESHOLD  = 30  # head tilt


def _rotation_matrix_to_euler(R):
    """
    Decompose 3x3 rotation matrix to Euler angles (pitch, yaw, roll) in degrees.
    Uses ZYX convention with normalization to fix the ~+/-180deg wrap-around:
    solvePnP places 'frontal face' near +/-180 pitch (coordinate frame artefact).
    Normalize: if abs(pitch) > 90, reflect back into [-90, 90].
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = float(np.degrees(np.arctan2( R[2, 1],  R[2, 2])))
        yaw   = float(np.degrees(np.arctan2(-R[2, 0],  sy)))
        roll  = float(np.degrees(np.arctan2( R[1, 0],  R[0, 0])))
    else:
        pitch = float(np.degrees(np.arctan2(-R[1, 2],  R[1, 1])))
        yaw   = float(np.degrees(np.arctan2(-R[2, 0],  sy)))
        roll  = 0.0

    # Normalize pitch & yaw out of the +-180 wrap-around zone.
    # e.g. pitch=-176.3 (frontal face) -> -(180+(-176.3)) = -3.7 (correct)
    if pitch < -90.0:
        pitch = -(180.0 + pitch)
    elif pitch > 90.0:
        pitch =  (180.0 - pitch)

    if yaw < -90.0:
        yaw = -(180.0 + yaw)
    elif yaw > 90.0:
        yaw =  (180.0 - yaw)

    return pitch, yaw, roll


class HeadPoseEstimator:
    def __init__(self):
        self.yaw_history   = deque(maxlen=15)
        self.pitch_history = deque(maxlen=15)
        self.roll_history  = deque(maxlen=15)
        self._last = (0.0, 0.0, 0.0)

    def _camera_matrix(self, frame_shape):
        h, w = frame_shape[:2]
        f = w  # approximate focal length
        return np.array([
            [f, 0, w / 2],
            [0, f, h / 2],
            [0, 0, 1   ],
        ], dtype=np.float64)

    def estimate(self, landmarks, frame_shape):
        """
        Returns (pitch, yaw, roll, rvec, tvec).
        Falls back to last known values on failure.
        """
        try:
            image_points = np.array(
                [landmarks[i, :2] for i in POSE_LANDMARK_IDS],
                dtype=np.float64,
            )

            cam_mat    = self._camera_matrix(frame_shape)
            dist_coeff = np.zeros((4, 1))

            ok, rvec, tvec = cv2.solvePnP(
                MODEL_POINTS,
                image_points,
                cam_mat,
                dist_coeff,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                return (*self._last, None, None)

            R, _ = cv2.Rodrigues(rvec)
            pitch, yaw, roll = _rotation_matrix_to_euler(R)

            self.pitch_history.append(pitch)
            self.yaw_history.append(yaw)
            self.roll_history.append(roll)
            self._last = (pitch, yaw, roll)

            return pitch, yaw, roll, rvec, tvec

        except Exception as e:
            return (*self._last, None, None)

    def smoothed_angles(self):
        pitch = float(np.mean(self.pitch_history)) if self.pitch_history else 0.0
        yaw   = float(np.mean(self.yaw_history))   if self.yaw_history   else 0.0
        roll  = float(np.mean(self.roll_history))  if self.roll_history  else 0.0
        return pitch, yaw, roll

    def is_looking_away(self):
        pitch, yaw, roll = self.smoothed_angles()
        return (
            abs(yaw)   > YAW_THRESHOLD   or
            abs(pitch) > PITCH_THRESHOLD or
            abs(roll)  > ROLL_THRESHOLD
        )

    def get_stats(self):
        pitch, yaw, roll = self.smoothed_angles()
        return {
            "pitch":        round(pitch, 1),
            "yaw":          round(yaw,   1),
            "roll":         round(roll,  1),
            "looking_away": self.is_looking_away(),
        }
