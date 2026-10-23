"""
Face Detection & Landmark Tracking using MediaPipe Tasks API (FaceLandmarker).
Works with mediapipe >= 0.10.30.
Extracts 478 facial landmarks (468 face + 10 iris) per frame.
"""

import os
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode


# Path to the .task model file (downloaded alongside this project)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")


class FaceDetector:
    # Key eye landmark indices (MediaPipe 478-point mesh)
    # ── EAR calculation: 6 points per eye ────────────────────────
    LEFT_EAR_LANDMARKS  = [362, 385, 387, 263, 373, 380]
    RIGHT_EAR_LANDMARKS = [33,  160, 158, 133, 153, 144]

    # ── Iris landmarks (indices 468-477) ──────────────────────────
    LEFT_IRIS  = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    # ── For eye outline drawing ───────────────────────────────────
    LEFT_EYE_OUTLINE  = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                         466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_OUTLINE = [33,  7,  163, 144, 145, 153, 154, 155, 133,
                         173, 157, 158, 159, 160, 161, 246]

    # ── Head pose key landmark indices ───────────────────────────
    POSE_LANDMARK_IDS = [1, 152, 226, 446, 57, 287]

    def __init__(self, max_faces: int = 3):
        model_path = os.path.abspath(_MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face Landmarker model not found at: {model_path}\n"
                "Run: curl -sL https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task "
                "-o face_landmarker.task"
            )

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)

    def process(self, frame_bgr: np.ndarray):
        """
        Run face landmarking on a BGR frame.
        Returns the raw FaceLandmarkerResult object.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect(mp_image)

    def get_landmarks(self, result, frame_shape) -> "np.ndarray | None":
        """
        Returns landmarks for the PRIMARY face (largest = closest to camera).
        Returns shape (N, 3) or None.
        """
        if not result.face_landmarks:
            return None

        h = int(frame_shape[0])
        w = int(frame_shape[1])

        # Build landmarks for all detected faces
        all_faces = []
        for face_lms in result.face_landmarks:
            coords = np.array(
                [[lm.x * w, lm.y * h, lm.z * w] for lm in face_lms],
                dtype=np.float32,
            )
            all_faces.append(coords)

        if len(all_faces) == 1:
            return all_faces[0]

        # Select the face with the largest bounding-box area (= primary user)
        def _bbox_area(lms):
            x0 = float(np.min(lms[:, 0]))
            x1 = float(np.max(lms[:, 0]))
            y0 = float(np.min(lms[:, 1]))
            y1 = float(np.max(lms[:, 1]))
            return (x1 - x0) * (y1 - y0)

        return max(all_faces, key=_bbox_area)

    def get_secondary_face_bboxes(self, result, frame_shape, primary_landmarks):
        """
        Returns list of bounding boxes for NON-primary faces.
        Used to draw a 'bystander' indicator without analysing them.
        """
        if not result.face_landmarks or len(result.face_landmarks) <= 1:
            return []

        h = int(frame_shape[0])
        w = int(frame_shape[1])
        padding = 20
        bboxes = []

        # Primary face fingerprint: centroid x,y
        px = float(np.mean(primary_landmarks[:, 0]))
        py = float(np.mean(primary_landmarks[:, 1]))

        for face_lms in result.face_landmarks:
            coords = np.array(
                [[lm.x * w, lm.y * h, lm.z * w] for lm in face_lms],
                dtype=np.float32,
            )
            cx = float(np.mean(coords[:, 0]))
            cy = float(np.mean(coords[:, 1]))
            # Skip the primary face (close centroid match)
            if abs(cx - px) < 30 and abs(cy - py) < 30:
                continue
            x_min = max(0, int(np.min(coords[:, 0])) - padding)
            y_min = max(0, int(np.min(coords[:, 1])) - padding)
            x_max = min(w, int(np.max(coords[:, 0])) + padding)
            y_max = min(h, int(np.max(coords[:, 1])) + padding)
            bboxes.append((x_min, y_min, x_max, y_max))

        return bboxes

    def get_face_bbox(self, landmarks, frame_shape):
        """Compute padded bounding box from landmarks."""
        h = int(frame_shape[0])
        w = int(frame_shape[1])
        padding = 20
        x_min = max(0, int(np.min(landmarks[:, 0])) - padding)
        y_min = max(0, int(np.min(landmarks[:, 1])) - padding)
        x_max = min(w, int(np.max(landmarks[:, 0])) + padding)
        y_max = min(h, int(np.max(landmarks[:, 1])) + padding)
        return x_min, y_min, x_max, y_max

    def get_eye_landmarks(self, landmarks: np.ndarray):
        """Return left and right eye EAR landmark arrays (shape 6×3 each)."""
        return landmarks[self.LEFT_EAR_LANDMARKS], landmarks[self.RIGHT_EAR_LANDMARKS]

    def get_iris_landmarks(self, landmarks: np.ndarray):
        """Return left and right iris landmark arrays (shape 4×3 each)."""
        try:
            return landmarks[self.LEFT_IRIS], landmarks[self.RIGHT_IRIS]
        except IndexError:
            return None, None

    def close(self):
        self._landmarker.close()
