"""Shared by training and real-time inference so both see identical inputs."""
import cv2
import numpy as np
from config import T_POSE, T_RGB, IMG

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def sample_indices(n, k, train=False):
    if train and n >= k:        # jittered uniform sampling
        e = np.linspace(0, n, k + 1)
        return np.array([np.random.randint(int(e[i]), max(int(e[i]) + 1, int(e[i + 1]))) for i in range(k)])
    return np.linspace(0, n - 1, k).round().astype(int)


def augment_pose(p):
    m = p != 0                  # keep missing landmarks at zero
    q = p.reshape(len(p), -1, 3).copy()
    a = np.random.uniform(-0.2, 0.2)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    q[..., :2] = q[..., :2] @ R.T
    q *= np.random.uniform(0.9, 1.1)
    q += np.random.normal(0, 0.01, q.shape).astype(np.float32)
    return q.reshape(len(p), -1) * m


def pose_features(kp, train=False):
    x = kp[sample_indices(len(kp), T_POSE, train)]
    if train:
        x = augment_pose(x)
    d = np.diff(x, axis=0, prepend=x[:1])
    return np.concatenate([x, d], 1).astype(np.float32)          # (T_POSE, 402)


def rgb_tensor(frames, train=False):
    clip = [frames[i] for i in sample_indices(len(frames), T_RGB, train)]
    if train:
        h, w = clip[0].shape[:2]
        s = np.random.uniform(0.8, 1.0)
        ch, cw = int(h * s), int(w * s)
        y0, x0 = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
        clip = [f[y0:y0 + ch, x0:x0 + cw] for f in clip]
    clip = np.stack([cv2.resize(f, (IMG, IMG)) for f in clip]).astype(np.float32) / 255.0
    if train:
        clip = np.clip(clip * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.1, 0.1), 0, 1)
    clip = (clip - MEAN) / STD
    return np.ascontiguousarray(clip.transpose(0, 3, 1, 2))      # (T_RGB, 3, 224, 224)
