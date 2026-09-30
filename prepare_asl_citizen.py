"""Build meta.csv + labels.json for ASL Citizen without downloading the full archive.

By default this reads the official Microsoft ZIP through HTTP range requests and only
fetches the three small CSV files. Video members are fetched later by extract_keypoints.py.

Choosing the vocabulary (ASL Citizen is close to balanced, ~30 videos per sign, so "most frequent"
is mostly a tie between many glosses):
  --pick frequent   (default) top N by training count, ties broken alphabetically -> deterministic
  --pick random     N glosses drawn with --seed -> a vocabulary spread across the alphabet
  --glosses FILE    your own list, one gloss per line (matched case-insensitively); overrides --n/--pick
"""
import argparse, csv, json
from collections import Counter

import numpy as np
from config import *
from asl_citizen import open_archive, load_split_csv

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150, help="number of glosses; this project uses 150; use 2731 for all")
ap.add_argument("--pick", choices=["frequent", "random"], default="frequent")
ap.add_argument("--seed", type=int, default=0, help="seed for --pick random")
ap.add_argument("--glosses", default=None, help="text file with one gloss per line")
args = ap.parse_args()

with open_archive() as z:
    train = load_split_csv(z, "train")
    val = load_split_csv(z, "val")
    test = load_split_csv(z, "test")

train_counts = train["Gloss"].value_counts()
all_glosses = sorted(train_counts.index)

if args.glosses:
    by_upper = {g.upper(): g for g in all_glosses}
    wanted = [w.strip() for w in open(args.glosses, encoding="utf-8") if w.strip()]
    unknown = [w for w in wanted if w.upper() not in by_upper]
    if unknown:
        raise SystemExit(f"{len(unknown)} gloss(es) not found in the training split, e.g. {unknown[:10]}")
    selected = [by_upper[w.upper()] for w in wanted]
elif args.pick == "random":
    rng = np.random.RandomState(args.seed)
    selected = [str(g) for g in rng.choice(all_glosses, size=min(args.n, len(all_glosses)), replace=False)]
else:
    selected = sorted(all_glosses, key=lambda g: (-int(train_counts[g]), g))[:args.n]

selected_set = set(selected)
labels = sorted(selected_set)
label_to_id = {g: i for i, g in enumerate(labels)}

rows = []
for split, df in (("train", train), ("val", val), ("test", test)):
    df = df[df["Gloss"].isin(selected_set)].copy()
    for r in df.to_dict("records"):
        rows.append({
            "video_id": str(r["Video file"]),
            "gloss": str(r["Gloss"]),
            "label": label_to_id[str(r["Gloss"])],
            "split": split,
            "signer_id": str(r["Participant ID"]),
            "video_file": str(r["Video file"]),
            "asl_lex_code": str(r.get("ASL-LEX Code", "")),
        })

DATA.mkdir(parents=True, exist_ok=True)
with open(META_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
json.dump(labels, open(LABELS_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

signers = {s: {r["signer_id"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
overlap = {f"{a}&{b}": len(signers[a] & signers[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}

print(f"ASL Citizen | {len(labels)} glosses | {len(rows)} selected videos")
print("splits:", dict(Counter(r["split"] for r in rows)))
print("train videos per gloss:", min(train_counts[g] for g in selected), "to", max(train_counts[g] for g in selected))
print("signers per split:", {k: len(v) for k, v in signers.items()}, "| signers shared between splits:", overlap,
      "(0 = signer-independent evaluation)")
print("labels:", labels[:10], "..." if len(labels) > 10 else "")
