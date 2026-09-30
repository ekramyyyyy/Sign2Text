"""Download the MediaPipe models once. They land in web/model/ and are used by BOTH Python (training) and the browser."""
import urllib.request
from config import MODEL_DIR

BASE = "https://storage.googleapis.com/mediapipe-models"
FILES = {
    "pose_landmarker_lite.task": f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}
MODEL_DIR.mkdir(parents=True, exist_ok=True)
for name, url in FILES.items():
    dst = MODEL_DIR / name
    if not dst.exists():
        print("downloading", name)
        urllib.request.urlretrieve(url, dst)
print("ok:", [p.name for p in MODEL_DIR.iterdir()])
