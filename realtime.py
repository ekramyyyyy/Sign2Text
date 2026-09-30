"""Webcam -> sign segments -> gloss tokens -> LLM -> English/Arabic sentence.

Keys: q quit | c clear | space/enter translate now.
Do NOT mirror the webcam feed: the models were trained on un-mirrored video.
"""
import argparse, os, threading, time
import cv2, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from keypoints import KeypointExtractor
from preprocess import pose_features, rgb_tensor
from model import SignNet
import llm

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    arabic_reshaper = None


def load_font(size=28):
    for p in [os.getenv("ARABIC_FONT", ""), "C:/Windows/Fonts/arial.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if p and os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT = load_font()


def draw_lines(bgr, lines):
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d, y = ImageDraw.Draw(img), 10
    for text, color in lines:
        if arabic_reshaper and any("\u0600" <= ch <= "\u06ff" for ch in text):
            text = get_display(arabic_reshaper.reshape(text))
        d.text((10, y), text, font=FONT, fill=color)
        y += 38
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev)
    m = SignNet(len(ck["labels"]), ck["mode"], pretrained=False)
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), ck["labels"], ck["mode"]


@torch.no_grad()
def predict(model, mode, kps, frames, labels, dev, k=3):
    pose = rgb = None
    if mode in ("pose", "hybrid"):
        pose = torch.from_numpy(pose_features(np.stack(kps))).unsqueeze(0).to(dev)
    if mode in ("rgb", "hybrid"):
        rgb = torch.from_numpy(rgb_tensor(frames)).unsqueeze(0).to(dev)
    conf, idx = model(pose, rgb).softmax(-1)[0].topk(k)
    return [(labels[i], float(c)) for c, i in zip(conf, idx)]


def translate(tokens, result):
    out = llm.gloss_to_sentence(tokens)
    result.update(out)
    print("EN:", out.get("english"), "| AR:", out.get("arabic"), "|", out.get("error", ""))
    result["busy"] = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pose.pt")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.30, help="min top-1 confidence to accept a sign")
    ap.add_argument("--end_gap", type=int, default=8, help="frames without hands that end a sign")
    ap.add_argument("--min_len", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=120)
    ap.add_argument("--pause", type=float, default=2.5, help="seconds of silence before auto-translating")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, labels, mode = load(a.ckpt, dev)
    cap, ext = cv2.VideoCapture(a.cam), KeypointExtractor()
    seg_kp, seg_fr, gap = [], [], 0
    tokens, last_t = [], time.time()
    result = {"english": "", "arabic": "", "busy": False}

    def start_translation():
        if tokens and not result["busy"]:
            result["busy"] = True
            threading.Thread(target=translate, args=(list(tokens), result), daemon=True).start()
            tokens.clear()

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        vec, hands = ext.process(rgb)
        if hands:
            seg_kp.append(vec)
            if mode != "pose":
                seg_fr.append(rgb)
            gap = 0
        elif seg_kp:
            gap += 1

        if seg_kp and (gap >= a.end_gap or len(seg_kp) >= a.max_len):
            if len(seg_kp) >= a.min_len:
                cands = predict(model, mode, seg_kp, seg_fr, labels, dev)
                if cands[0][1] >= a.thresh:
                    tokens.append(cands)
                    print("sign:", ", ".join(f"{g} {c:.2f}" for g, c in cands))
                last_t = time.time()
            seg_kp, seg_fr, gap = [], [], 0

        if tokens and not seg_kp and time.time() - last_t > a.pause:
            start_translation()

        lines = [("signing..." if seg_kp else "ready", (255, 255, 0)),
                 (" ".join(c[0][0] for c in tokens), (0, 255, 0)),
                 (result["english"], (255, 255, 255)), (result["arabic"], (255, 200, 0))]
        cv2.imshow("sign2text", draw_lines(bgr, lines))
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("c"):
            tokens.clear(); result.update(english="", arabic="")
        if k in (13, 32):
            start_translation()
    cap.release(); ext.close(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
