import argparse, json, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import *
from dataset import ASLCitizen
from model import SignNet


def fwd(model, b, dev):
    return model(b["pose"].to(dev) if "pose" in b else None, b["rgb"].to(dev) if "rgb" in b else None)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    c1 = c5 = n = 0
    for b in loader:
        y = b["label"].to(dev)
        top = fwd(model, b, dev).topk(min(5, model.head[-1].out_features), -1).indices
        c1 += (top[:, 0] == y).sum().item()
        c5 += (top == y[:, None]).any(1).sum().item()
        n += len(y)
    return c1 / max(n, 1), c5 / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pose", choices=["pose", "rgb", "hybrid"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weighted", action="store_true",
                    help="inverse-frequency class weights in the loss, to stop common classes dominating")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    labels = json.load(open(LABELS_JSON))
    mk = lambda s, t: DataLoader(ASLCitizen(s, a.mode, t), a.bs, shuffle=t, num_workers=a.workers, drop_last=t, pin_memory=dev == "cuda")
    tr, va, te = mk("train", True), mk("val", False), mk("test", False)
    print(f"train {len(tr.dataset)} | val {len(va.dataset)} | test {len(te.dataset)} | classes {len(labels)} | device {dev}")
    if min(len(tr.dataset), len(va.dataset), len(te.dataset)) == 0:
        raise SystemExit("A split has no extracted keypoints. Run extract_keypoints.py first "
                         "(or restore a backup) and check extract_failures.csv.")
    if len(tr) == 0:
        raise SystemExit(f"Only {len(tr.dataset)} training clips, fewer than one batch ({a.bs}). Lower --bs or extract more.")

    model = SignNet(len(labels), a.mode).to(dev)
    bb = [p for n, p in model.named_parameters() if n.startswith("rgb.m.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("rgb.m.") and p.requires_grad]
    groups = [{"params": rest, "lr": a.lr}] + ([{"params": bb, "lr": a.lr * 0.1}] if bb else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    total, warm = a.epochs * len(tr), len(tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * s / total)))

    class_w = None
    if a.weighted:
        counts = torch.zeros(len(labels))
        for r in tr.dataset.rows:
            counts[r["label"]] += 1
        counts = counts.clamp(min=1)                          # avoid div-by-zero for any empty class
        class_w = (counts.sum() / (len(labels) * counts)).to(dev)   # inverse-frequency, mean weight ~1
        print(f"class weights: min {class_w.min():.2f} max {class_w.max():.2f} "
              f"(rarest class {int(counts.min())} samples, commonest {int(counts.max())})")
    crit = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_w)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{a.mode}{'_weighted' if a.weighted else ''}"
    path, best = CKPT_DIR / f"{tag}.pt", 0.0
    for ep in range(a.epochs):
        model.train()
        run = 0.0
        for b in tqdm(tr, desc=f"epoch {ep + 1}/{a.epochs}", leave=False):
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
                loss = crit(fwd(model, b, dev), b["label"].to(dev))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        t1, t5 = evaluate(model, va, dev)
        print(f"epoch {ep + 1}: loss {run / len(tr):.3f} | val top1 {t1:.3f} top5 {t5:.3f}")
        if t1 > best:
            best = t1
            torch.save({"model": model.state_dict(), "mode": a.mode, "labels": labels}, path)

    model.load_state_dict(torch.load(path, map_location=dev)["model"])
    t1, t5 = evaluate(model, te, dev)
    print(f"TEST top1 {t1:.3f} top5 {t5:.3f}  (best val {best:.3f}) -> {path}")
    res = {"mode": a.mode, "weighted": a.weighted, "classes": len(labels), "epochs": a.epochs,
           "n_train": len(tr.dataset), "n_val": len(va.dataset), "n_test": len(te.dataset),
           "best_val_top1": round(best, 4), "test_top1": round(t1, 4), "test_top5": round(t5, 4)}
    json.dump(res, open(CKPT_DIR / f"{tag}_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
