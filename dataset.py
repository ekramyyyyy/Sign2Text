import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset
from config import *
from preprocess import pose_features


class ASLCitizen(Dataset):
    """ASL Citizen dataset backed by pre-extracted MediaPipe keypoints."""
    def __init__(self, split, mode="pose", train=False):
        if mode != "pose":
            raise NotImplementedError(
                "Colab remote-dataset mode currently trains the pose branch only. "
                "RGB/hybrid still require local video files to avoid re-downloading the remote ZIP every epoch."
            )
        df = pd.read_csv(META_CSV, dtype={"video_id": str})
        def kp_path(v):
            from pathlib import Path
            return KP_DIR / f"{Path(str(v)).stem}.npy"
        df = df[(df.split == split) & df.video_id.map(lambda v: kp_path(v).exists())]
        self.rows, self.mode, self.train = df.to_dict("records"), mode, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        from pathlib import Path
        vid = Path(str(r["video_id"])).stem
        pose = torch.from_numpy(pose_features(np.load(KP_DIR / f"{vid}.npy"), self.train))
        return {"label": int(r["label"]), "pose": pose}


