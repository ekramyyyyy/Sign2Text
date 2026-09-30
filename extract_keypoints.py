"""Run MediaPipe over ASL Citizen videos and cache keypoints as .npy (resumable).

The official 42.8 GB ZIP is read remotely with HTTP range requests. Each MP4 is fetched into a
temporary file on the VM, turned into keypoints, and deleted.

Robustness/speed notes
- Downloads run in a small thread pool (SIGN2TEXT_DL_WORKERS, default 3; set 1 to disable) so the
  network fetch of the next videos overlaps with MediaPipe on the current one. Each thread has its own
  archive handle, and a failed fetch is retried with a fresh connection.
- .npy files are written atomically (tmp file + rename). A session killed mid-write can therefore never
  leave a truncated .npy that a later resume would treat as "done".
- Frames are streamed, not loaded into RAM.
- Clips where the pose is (almost) never detected are skipped and reported instead of silently
  becoming all-zero training samples.
- Every skip is recorded with its reason in DATA/extract_failures.csv.
"""
import csv
import os
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd
from tqdm import tqdm
from config import *
from keypoints import KeypointExtractor
from video_io import iter_frames
from asl_citizen import open_archive, build_video_map, extract_member_to_temp

DL_WORKERS = max(1, int(os.environ.get("SIGN2TEXT_DL_WORKERS", "3")))
RETRIES = 4
MIN_FRAMES = 4
MIN_POSE_FRAC = 0.2          # skip clips where the pose is found in fewer than 20% of frames

_tls = threading.local()
_opened, _opened_lock = [], threading.Lock()


def _archive(fresh=False):
    z = getattr(_tls, "z", None)
    if z is not None and not fresh:
        return z
    if z is not None:
        try:
            z.close()
        except Exception:
            pass
    _tls.z = open_archive()
    with _opened_lock:
        _opened.append(_tls.z)
    return _tls.z


def fetch(member):
    """Download one archive member to a temp file, retrying on a fresh connection."""
    last = None
    for attempt in range(RETRIES):
        try:
            return extract_member_to_temp(_archive(fresh=attempt > 0), member)
        except Exception as e:
            last = e
            time.sleep(min(2 ** attempt, 20))
    raise last


def keypoints_for(ext, path):
    ext.reset()
    seq, pose_seen = [], 0
    for frame in iter_frames(path):
        vec = ext.process(frame)[0]
        seq.append(vec)
        pose_seen += int(bool(vec.any()))
    return (np.stack(seq) if seq else None), pose_seen


def save_atomic(out, arr):
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, out)


def main():
    KP_DIR.mkdir(parents=True, exist_ok=True)
    for stale in KP_DIR.glob("*.tmp"):
        stale.unlink(missing_ok=True)

    df = pd.read_csv(META_CSV, dtype={"video_id": str})
    with open_archive() as z:
        video_map = build_video_map(z)

    todo, failed = [], []
    already = 0
    for r in df.to_dict("records"):
        vid = str(r["video_id"])
        out = KP_DIR / f"{Path(vid).stem}.npy"
        if out.exists():
            already += 1
            continue
        member = video_map.get(Path(vid).name)
        if member is None:
            failed.append((vid, "not_in_zip", ""))
            continue
        todo.append((vid, out, member))
    print(f"{len(df)} videos in meta | {already} already extracted | {len(todo)} to do | "
          f"{len(failed)} missing from ZIP | {DL_WORKERS} download thread(s)")

    ext = KeypointExtractor()
    processed = 0
    pending, it = deque(), iter(todo)

    def submit_next(pool):
        try:
            item = next(it)
        except StopIteration:
            return
        pending.append((item, pool.submit(fetch, item[2])))

    try:
        with ThreadPoolExecutor(DL_WORKERS) as pool:
            for _ in range(DL_WORKERS * 2):
                submit_next(pool)
            bar = tqdm(total=len(todo), desc="ASL Citizen keypoints")
            while pending:
                (vid, out, _member), fut = pending.popleft()
                submit_next(pool)
                tmp = None
                try:
                    tmp = fut.result()
                    arr, pose_seen = keypoints_for(ext, tmp)
                    if arr is None or len(arr) < MIN_FRAMES:
                        failed.append((vid, "too_short", ""))
                    elif pose_seen / len(arr) < MIN_POSE_FRAC:
                        failed.append((vid, "no_pose_detected", f"{pose_seen}/{len(arr)} frames"))
                    else:
                        save_atomic(out, arr)
                        processed += 1
                except Exception as e:
                    failed.append((vid, type(e).__name__, str(e)[:200]))
                finally:
                    if tmp:
                        Path(tmp).unlink(missing_ok=True)
                    bar.update(1)
            bar.close()
    finally:
        for (_item, fut) in pending:                 # interrupted: don't leak prefetched temp files
            if not fut.cancel() and fut.done() and not fut.exception():
                Path(fut.result()).unlink(missing_ok=True)
        ext.close()
        for z in _opened:
            try:
                z.close()
            except Exception:
                pass

    if failed:
        DATA.mkdir(parents=True, exist_ok=True)
        with open(DATA / "extract_failures.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "reason", "detail"])
            w.writerows(failed)
    print(f"done | newly processed {processed} | skipped {len(failed)} "
          f"{dict(Counter(r for _, r, _ in failed))}"
          + (f" | details in {DATA / 'extract_failures.csv'}" if failed else ""))
    print("re-run this cell to retry skipped videos (existing .npy files are not redone)")


if __name__ == "__main__":
    main()
