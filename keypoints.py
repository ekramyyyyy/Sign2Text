"""PoseLandmarker + HandLandmarker (MediaPipe Tasks). The browser app uses the SAME .task files,
so training features == deployment features. Run `python download_models.py` once."""
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
from config import MODEL_DIR
from features import build_vec

POSE_TASK = MODEL_DIR / "pose_landmarker_lite.task"
HAND_TASK = MODEL_DIR / "hand_landmarker.task"


def _arr(lms):
    return np.array([[p.x, p.y, p.z] for p in lms], np.float32)


class KeypointExtractor:
    def __init__(self):
        self._make()

    def _make(self):
        self.t = 0
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(POSE_TASK)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=str(HAND_TASK)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2))

    def reset(self):            # between videos, so tracking state does not leak
        self.close()
        self._make()

    def close(self):
        self.pose.close()
        self.hands.close()

    def process(self, rgb):
        """rgb: HxWx3 uint8, NOT mirrored. Returns (vec[201], hands_visible)."""
        H, W = rgb.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        self.t += 33
        pr = self.pose.detect_for_video(img, self.t)
        hr = self.hands.detect_for_video(img, self.t)
        pose = _arr(pr.pose_landmarks[0]) if pr.pose_landmarks else None
        return build_vec(pose, [_arr(h) for h in hr.hand_landmarks], W, H)
