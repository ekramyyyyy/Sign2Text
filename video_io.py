import cv2


def _crop(frame, bbox):
    if not bbox:
        return frame
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if x2 - x1 < 32 or y2 - y1 < 32:
        return frame
    return frame[y1:y2, x1:x2]


def iter_frames(path):
    """Yield RGB frames one at a time. Nothing is buffered, so memory stays flat even for
    long or high-resolution clips (ASL Citizen webcam videos are not all small)."""
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def read_frames(path, bbox=None, start=1, end=-1):
    """Read RGB frames of one clip into a list (frame_start is 1-indexed, end=-1 means till the end).
    Prefer iter_frames() for keypoint extraction; this loads every frame into RAM."""
    cap = cv2.VideoCapture(str(path))
    s, end = max(int(start) - 1, 0), int(end)
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i >= s and (end == -1 or i < end):
            frames.append(cv2.cvtColor(_crop(bgr, bbox), cv2.COLOR_BGR2RGB))
        i += 1
        if end != -1 and i >= end:
            break
    cap.release()
    return frames
