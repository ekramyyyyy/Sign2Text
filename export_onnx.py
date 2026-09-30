"""Export the trained POSE model to ONNX for in-browser inference (web/model/pose.onnx + labels.json).

Uses the newer torch.export-based ("dynamo") exporter. The older legacy TorchScript tracer produces a
graph that LOOKS valid (right shape, no error) but gives numerically wrong predictions for this model's
TransformerEncoderLayer — confirmed by comparing against real test-set predictions (only ~70% agreement,
vs ~1e-6 max difference and 100% agreement with the dynamo exporter). Needs `pip install onnxscript`.
"""
import argparse, json, shutil
import numpy as np, torch, torch.nn as nn
from config import *
from model import SignNet

if hasattr(torch.backends, "mha"):          # disable TransformerEncoderLayer's fused "fast path": tracers
    torch.backends.mha.set_fastpath_enabled(False)   # can capture it incorrectly, giving a wrong (not just slow) graph

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=str(CKPT_DIR / "pose.pt"))
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu")
assert ck["mode"] == "pose", "browser deployment uses the pose model (fast, no video stream needed)"
net = SignNet(len(ck["labels"]), "pose", pretrained=False)
net.load_state_dict(ck["model"])
net.eval()


class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pose):
        return self.m(pose=pose)


MODEL_DIR.mkdir(parents=True, exist_ok=True)
out = MODEL_DIR / "pose.onnx"
dummy = torch.randn(1, T_POSE, POSE_IN)     # batch=1: matches real usage (browser + realtime.py always send one segment at a time)
args = dict(input_names=["pose"], output_names=["logits"], opset_version=17)

out.unlink(missing_ok=True)
(MODEL_DIR / "pose.onnx.data").unlink(missing_ok=True)   # stale external-data file from an older export, if any
try:
    prog = torch.onnx.export(Wrap(net), dummy, dynamo=True, **args)   # correct exporter — see module docstring
    prog.save(str(out))   # single self-contained file (no external .data sidecar) — required for onnxruntime-web
except Exception as e:
    print("dynamo export failed, falling back to the legacy exporter (may be numerically WRONG, see docstring):", e)
    torch.onnx.export(Wrap(net), dummy, str(out), dynamo=False, **args)
shutil.copy(LABELS_JSON, MODEL_DIR / "labels.json")

try:                                                                       # sanity check: ONNX == PyTorch (batch=1, matching real usage)
    import onnxruntime as ort
    x = torch.randn(1, T_POSE, POSE_IN)
    ref = net(pose=x).detach().numpy()
    got = ort.InferenceSession(str(out)).run(None, {"pose": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    print("max |torch - onnx| =", diff, "OK" if diff < 1e-3 else "!! LARGE DIFF — export may be broken !!")
except ImportError:
    print("pip install onnxruntime to verify the export")
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
