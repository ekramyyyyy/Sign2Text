import os
from pathlib import Path

ROOT = Path(__file__).parent


def _p(env_name, default):
    v = os.environ.get(env_name)
    return Path(v) if v else default


DATA = _p("SIGN2TEXT_DATA", ROOT / "data")
ASL_CITIZEN_ZIP = _p("SIGN2TEXT_ASL_CITIZEN_ZIP", DATA / "ASL_Citizen.zip")
VIDEO_DIR = _p("SIGN2TEXT_VIDEO_DIR", DATA / "videos")        # optional local MP4 directory

KP_DIR = _p("SIGN2TEXT_KP_DIR", DATA / "keypoints")            # {video_id}.npy
META_CSV = _p("SIGN2TEXT_META_CSV", DATA / "meta.csv")
LABELS_JSON = _p("SIGN2TEXT_LABELS_JSON", DATA / "labels.json")
CKPT_DIR = _p("SIGN2TEXT_CKPT_DIR", ROOT / "checkpoints")
MODEL_DIR = _p("SIGN2TEXT_MODEL_DIR", ROOT / "web" / "model")  # MediaPipe .task files + exported ONNX model

T_POSE = 32                        # frames fed to the pose transformer
T_RGB = 16                         # frames fed to VideoMAE
IMG = 224
KP_DIM = 67 * 3                    # 25 pose + 21 left hand + 21 right hand landmarks, xyz
POSE_IN = KP_DIM * 2               # + frame-to-frame deltas
VIDEOMAE = "MCG-NJU/videomae-base-finetuned-kinetics"
