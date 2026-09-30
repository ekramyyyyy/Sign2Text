"""Bundle / restore everything a Colab run generates into ONE .tar file (no Google Drive needed).

    python backup.py save    [/content/sign2text_backup.tar] [--no-keypoints]
    python backup.py restore  /content/sign2text_backup.tar

Contents: meta.csv, labels.json, extract_failures.csv, keypoints/*.npy, checkpoints/*.
A Colab runtime reset wipes /content, so download this file after extraction and after training.
"""
import argparse, tarfile
from pathlib import Path
from config import *

DEFAULT = "/content/sign2text_backup.tar"
FILES = {"meta.csv": META_CSV, "labels.json": LABELS_JSON, "extract_failures.csv": DATA / "extract_failures.csv"}
DIRS = {"keypoints": KP_DIR, "checkpoints": CKPT_DIR}


def save(path, keypoints=True):
    n = 0
    with tarfile.open(path, "w") as tar:                       # uncompressed: .npy files barely compress
        for arc, p in FILES.items():
            if Path(p).exists():
                tar.add(p, arcname=arc); n += 1
        for arc, d in DIRS.items():
            if arc == "keypoints" and not keypoints:
                continue
            for p in sorted(Path(d).glob("*")):
                if p.is_file() and not p.name.endswith(".tmp"):
                    tar.add(p, arcname=f"{arc}/{p.name}"); n += 1
    print(f"saved {n} files -> {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")


def restore(path):
    n = 0
    with tarfile.open(path, "r") as tar:
        for m in tar:
            if not m.isfile():
                continue
            parts = Path(m.name).parts
            if ".." in parts or m.name.startswith("/"):
                continue                                       # never write outside the target dirs
            if len(parts) == 1 and parts[0] in FILES:
                dest = Path(FILES[parts[0]])
            elif len(parts) == 2 and parts[0] in DIRS:
                dest = Path(DIRS[parts[0]]) / parts[1]
            else:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            n += 1
    print(f"restored {n} files from {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["save", "restore"])
    ap.add_argument("path", nargs="?", default=DEFAULT)
    ap.add_argument("--no-keypoints", action="store_true")
    a = ap.parse_args()
    save(a.path, not a.no_keypoints) if a.cmd == "save" else restore(a.path)
