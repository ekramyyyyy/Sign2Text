"""Keypoint vector construction. Pure numpy; web/features.js is a line-by-line mirror (see check_parity.py)."""
import numpy as np


def build_vec(pose, hands, W, H):
    """pose: (33,3) normalized landmarks or None; hands: list of (21,3); W,H: frame size in pixels.
    Returns (vec[201], hands_seen). Coordinates are converted to pixels first so the result does not depend
    on the frame's aspect ratio (training clips are signer crops, the browser feed is a full frame)."""
    vec = np.zeros((67, 3), np.float32)
    if pose is None:
        return vec.reshape(-1), False
    k = np.array([W, H, W], np.float32)
    pose = pose * k
    hands = [h * k for h in hands[:2]]
    c = (pose[11] + pose[12]) / 2                        # center on mid-shoulders
    d = float(np.linalg.norm(pose[11, :2] - pose[12, :2]))
    s = d if d > 1e-3 else 1.0                           # scale by shoulder width
    vec[:25] = (pose[:25] - c) / s

    lw, rw = pose[15, :2], pose[16, :2]                  # assign hands to left/right by nearest pose wrist
    cost = lambda h, w: float(np.linalg.norm(h[0, :2] - w))
    left = right = None
    if len(hands) == 1:
        if cost(hands[0], lw) <= cost(hands[0], rw):
            left = hands[0]
        else:
            right = hands[0]
    elif len(hands) >= 2:
        h0, h1 = hands
        if cost(h0, lw) + cost(h1, rw) <= cost(h0, rw) + cost(h1, lw):
            left, right = h0, h1
        else:
            left, right = h1, h0
    if left is not None:
        vec[25:46] = (left - c) / s
    if right is not None:
        vec[46:67] = (right - c) / s
    return vec.reshape(-1), (left is not None or right is not None)
