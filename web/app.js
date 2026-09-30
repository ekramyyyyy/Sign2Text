// Everything vision-related runs IN THE BROWSER (MediaPipe + ONNX Runtime Web): no video is uploaded, no network lag.
// Only the final gloss list goes to /api/translate (tiny JSON) for the LLM.
import { PoseLandmarker, HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import { buildVec, poseFeatures, T, IN } from "./features.js";

const CFG = {
  thresh: 0.30, endGap: 8, minLen: 10, maxLen: 120, pauseMs: 1900, topK: 3,
  // Mid-air segmentation: end a sign when the hands go briefly still, even if they never leave
  // the camera frame - so you can sign back-to-back without pulling your hand out between each one.
  // This is a heuristic: too sensitive and it can split ONE sign into two bad predictions if that
  // sign itself has a natural pause partway through. Tune motionStillFrames/motionThresh if so.
  motionStillFrames: 9,   // consecutive "still" frames (~300ms @30fps) before ending the segment
  motionThresh: 0.015,    // mean per-dim movement below this counts as "still"; raise if signs get
};                        // cut short mid-motion, lower if two signs get fused into one prediction
const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");

let pose, hands, sess, labels;
let tokens = [], segKp = [], gap = 0, still = 0, lastSignT = performance.now();
let busy = false, lastVideoTime = -1, lastTs = 0, predChain = Promise.resolve();
let frames = 0, fpsT = performance.now();

async function make(cls, vision, path, opts) {
  for (const delegate of ["GPU", "CPU"]) {
    try {
      return await cls.createFromOptions(vision, { baseOptions: { modelAssetPath: path, delegate }, runningMode: "VIDEO", ...opts });
    } catch (e) { console.warn(path, delegate, e); }
  }
  throw new Error("could not load " + path);
}

async function init() {
  $("status").textContent = "loading models…";
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/";
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  [pose, hands, labels, sess] = await Promise.all([
    make(PoseLandmarker, vision, "model/pose_landmarker_lite.task", { numPoses: 1 }),
    make(HandLandmarker, vision, "model/hand_landmarker.task", { numHands: 2 }),
    fetch("model/labels.json").then(r => r.json()),
    ort.InferenceSession.create("model/pose.onnx", { executionProviders: ["wasm"], graphOptimizationLevel: "all" }),
  ]);
  $("status").textContent = "";
}

async function predict(kps) {
  const x = poseFeatures(kps);
  const logits = (await sess.run({ pose: new ort.Tensor("float32", x, [1, T, IN]) })).logits.data;
  const mx = Math.max(...logits);
  const e = Array.from(logits, v => Math.exp(v - mx)), sum = e.reduce((a, b) => a + b, 0);
  return e.map((v, i) => [labels[i], v / sum]).sort((a, b) => b[1] - a[1]).slice(0, CFG.topK);
}

function renderTokens() {
  $("tokens").innerHTML = tokens.map(c => `<span class="chip">${c[0][0]}<small>${Math.round(c[0][1] * 100)}%</small></span>`).join("");
}

async function translate() {
  if (!tokens.length || busy) return;
  busy = true;
  const toks = tokens.splice(0);
  renderTokens();
  $("en").textContent = "…";
  const t0 = performance.now();
  try {
    const r = await fetch("/api/translate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ tokens: toks }) });
    const j = await r.json();
    console.log(`[timing] /api/translate round trip: ${(performance.now() - t0).toFixed(0)}ms`);
    $("en").textContent = j.english || ""; $("ar").textContent = j.arabic || "";
  } catch (e) {
    $("en").textContent = toks.map(c => c[0][0]).join(" ").toLowerCase(); $("ar").textContent = "";
  }
  busy = false;
}

function draw(pr, hr) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const dot = (p, col) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(p.x * canvas.width, p.y * canvas.height, 3, 0, 6.3); ctx.fill(); };
  (pr.landmarks[0] || []).slice(0, 25).forEach(p => dot(p, "#60a5fa"));
  hr.landmarks.forEach(h => h.forEach(p => dot(p, "#4ade80")));
}

function handMotion(cur, prev) {
  // mean absolute change across the hand-landmark region of the feature vector (indices 75..200:
  // both hands, 3D, already normalized by shoulder width - see features.js buildVec()).
  let sum = 0;
  for (let i = 75; i < 201; i++) sum += Math.abs(cur[i] - prev[i]);
  return sum / (201 - 75);
}

function finishSegment() {
  still = 0;
  if (segKp.length >= CFG.minLen) {
    const kps = segKp, t0 = performance.now();
    predChain = predChain.then(async () => {
      const cands = await predict(kps);
      console.log(`[timing] predict(): ${(performance.now() - t0).toFixed(0)}ms`);
      if (cands[0][1] >= CFG.thresh) { tokens.push(cands); renderTokens(); }
    });
    lastSignT = performance.now();
  }
  segKp = []; gap = 0;
}

function loop() {
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const ts = lastTs = Math.max(lastTs + 1, Math.round(performance.now()));
  const pr = pose.detectForVideo(video, ts), hr = hands.detectForVideo(video, ts);
  const { vec, seen } = buildVec(pr.landmarks[0], hr.landmarks, video.videoWidth, video.videoHeight);
  draw(pr, hr);

  if (seen) {
    if (segKp.length) {
      still = handMotion(vec, segKp[segKp.length - 1]) < CFG.motionThresh ? still + 1 : 0;
    }
    segKp.push(vec); gap = 0;
  } else if (segKp.length) gap++;
  const midAirEnd = segKp.length >= CFG.minLen && still >= CFG.motionStillFrames;
  if (segKp.length && (gap >= CFG.endGap || segKp.length >= CFG.maxLen || midAirEnd)) finishSegment();
  if (tokens.length && !segKp.length && performance.now() - lastSignT > CFG.pauseMs) translate();

  $("badge").textContent = segKp.length ? "signing…" : "ready";
  $("badge").classList.toggle("live", segKp.length > 0);
  if (++frames === 30) { $("status").textContent = `${Math.round(30000 / (performance.now() - fpsT))} fps`; frames = 0; fpsT = performance.now(); }
}

$("start").onclick = async () => {
  $("start").disabled = true;
  try {
    await init();
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: 30, facingMode: "user" }, audio: false });
    await video.play();
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    $("start").style.display = "none";
    loop();
  } catch (e) { $("status").textContent = "error: " + e.message; $("start").disabled = false; }
};
$("go").onclick = translate;
$("clear").onclick = () => { tokens = []; renderTokens(); $("en").textContent = ""; $("ar").textContent = ""; };
addEventListener("keydown", e => { if (e.code === "Space") { e.preventDefault(); translate(); } if (e.key === "c") $("clear").click(); });