"""Checks that web/features.js (browser) produces the same numbers as features.py/preprocess.py (training).
Needs node. Run: python check_parity.py"""
import json, subprocess
import numpy as np
from features import build_vec
from preprocess import pose_features

rng = np.random.default_rng(0)
cases = []
for i in range(30):
    W, H = int(rng.choice([640, 480, 1280])), int(rng.choice([480, 720, 360]))
    pose = rng.random((33, 3)).astype(np.float32)
    pose[11], pose[12] = [0.6, 0.5, 0.0], [0.4, 0.5, 0.0]
    hands = [rng.random((21, 3)).astype(np.float32) for _ in range(i % 3)]
    cases.append(dict(W=W, H=H, pose=pose.tolist(), hands=[h.tolist() for h in hands]))
kps = [rng.standard_normal(201).astype(np.float32) for _ in range(57)]

js = r"""
import {buildVec, poseFeatures} from './web/features.js';
import fs from 'fs';
const d = JSON.parse(fs.readFileSync(0, 'utf8'));
const xyz = a => a.map(p => ({x: p[0], y: p[1], z: p[2]}));
const vecs = d.cases.map(c => { const r = buildVec(xyz(c.pose), c.hands.map(xyz), c.W, c.H); return {vec: Array.from(r.vec), seen: r.seen}; });
const pf = Array.from(poseFeatures(d.kps.map(k => Float32Array.from(k))));
console.log(JSON.stringify({vecs, pf}));
"""
out = subprocess.run(["node", "--input-type=module", "-e", js], input=json.dumps(dict(cases=cases, kps=[k.tolist() for k in kps])),
                     capture_output=True, text=True, check=True).stdout
res = json.loads(out)

worst = 0.0
for c, r in zip(cases, res["vecs"]):
    v, seen = build_vec(np.array(c["pose"], np.float32), [np.array(h, np.float32) for h in c["hands"]], c["W"], c["H"])
    assert seen == r["seen"], "hands_seen mismatch"
    worst = max(worst, float(np.abs(v - np.array(r["vec"])).max()))
py_pf = pose_features(np.stack(kps))
worst_pf = float(np.abs(py_pf - np.array(res["pf"]).reshape(py_pf.shape)).max())
print(f"max diff build_vec: {worst:.2e} | pose_features: {worst_pf:.2e}")
assert worst < 1e-3 and worst_pf < 1e-4, "PARITY FAILED"
print("PARITY OK")
