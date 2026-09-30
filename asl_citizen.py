"""ASL Citizen access helpers.

Supports the official Microsoft ASL Citizen ZIP either locally or remotely.
For Colab, the default is HTTP range access: the 42.8 GB archive is NOT downloaded
as a whole. Only the CSV metadata and the individual MP4 members needed for
keypoint extraction are fetched.
"""
from __future__ import annotations

import os
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

ASL_CITIZEN_URL = os.environ.get(
    "SIGN2TEXT_ASL_CITIZEN_URL",
    "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip",
)
LOCAL_ZIP = os.environ.get("SIGN2TEXT_ASL_CITIZEN_ZIP", "")


def open_archive():
    """Open ASL Citizen from a local ZIP or via HTTP range requests."""
    path = Path(LOCAL_ZIP) if LOCAL_ZIP else None
    if path and path.exists():
        return zipfile.ZipFile(path, "r")
    try:
        from remotezip import RemoteZip
    except ImportError as e:
        raise RuntimeError("Install remotezip first: pip install -q remotezip") from e
    return RemoteZip(ASL_CITIZEN_URL, initial_buffer_size=16 * 1024 * 1024,
                     timeout=(30, 120), headers={"User-Agent": "sign2text-asl-citizen/1.0"})




def find_split_member(z, split: str) -> str:
    candidates = [n for n in z.namelist() if n.lower().endswith(f"/{split.lower()}.csv") or n.lower() == f"{split.lower()}.csv"]
    if not candidates:
        raise FileNotFoundError(f"Could not find {split}.csv inside ASL Citizen ZIP")
    return candidates[0]


def load_split_csv(z, split: str) -> pd.DataFrame:
    member = find_split_member(z, split)
    with z.open(member) as f:
        df = pd.read_csv(f)
    # Official column names are: Participant ID, Video file, Gloss, ASL-LEX Code.
    rename = {c: c.strip() for c in df.columns}
    df = df.rename(columns=rename)
    required = {"Participant ID", "Video file", "Gloss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{split}.csv is missing columns {sorted(missing)}; got {list(df.columns)}")
    return df


def build_video_map(z):
    """Map an MP4 basename to its archive member path."""
    out = {}
    for name in z.namelist():
        if name.lower().endswith(".mp4"):
            out[Path(name).name] = name
    return out


def extract_member_to_temp(z, member: str):
    """Extract one archive member to a temporary file and return its path."""
    suffix = Path(member).suffix or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with z.open(member) as src:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
        tmp.close()
        return Path(tmp.name)
    except Exception:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise


def read_video_bytes(z, member: str) -> bytes:
    with z.open(member) as f:
        return f.read()
