// Line-by-line mirror of features.py (build_vec) and preprocess.py (pose_features, eval mode).
// `python check_parity.py` verifies both give identical numbers.
export const T = 32, KP = 201, IN = 402;

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const roundHE = (x) => { const r = Math.round(x); return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r; }; // numpy rounds half-to-even

// pose: [{x,y,z}] (33) or undefined; hands: [[{x,y,z}] (21)]; W,H video size in pixels
export function buildVec(pose, hands, W, H) {
  const vec = new Float32Array(KP);
  if (!pose) return { vec, seen: false };
  const P = pose.map(p => [p.x * W, p.y * H, p.z * W]);
  const Hs = hands.slice(0, 2).map(h => h.map(p => [p.x * W, p.y * H, p.z * W]));
  const c = [0, 1, 2].map(i => (P[11][i] + P[12][i]) / 2);
  const d = dist(P[11], P[12]);
  const s = d > 1e-3 ? d : 1;
  const put = (pts, off, n) => {
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) vec[(off + i) * 3 + k] = (pts[i][k] - c[k]) / s;
  };
  put(P, 0, 25);
  const lw = P[15], rw = P[16], cost = (h, w) => dist(h[0], w);
  let L = null, R = null;
  if (Hs.length === 1) {
    if (cost(Hs[0], lw) <= cost(Hs[0], rw)) L = Hs[0]; else R = Hs[0];
  } else if (Hs.length >= 2) {
    const [h0, h1] = Hs;
    if (cost(h0, lw) + cost(h1, rw) <= cost(h0, rw) + cost(h1, lw)) { L = h0; R = h1; } else { L = h1; R = h0; }
  }
  if (L) put(L, 25, 21);
  if (R) put(R, 46, 21);
  return { vec, seen: !!(L || R) };
}

// kps: array of Float32Array(201) -> Float32Array(T*402)  [positions | frame-to-frame deltas]
export function poseFeatures(kps) {
  const n = kps.length, out = new Float32Array(T * IN);
  const idx = Array.from({ length: T }, (_, i) => roundHE(i * (n - 1) / (T - 1)));
  for (let t = 0; t < T; t++) {
    const cur = kps[idx[t]], prev = kps[idx[Math.max(t - 1, 0)]];
    for (let d = 0; d < KP; d++) { out[t * IN + d] = cur[d]; out[t * IN + KP + d] = cur[d] - prev[d]; }
  }
  return out;
}
