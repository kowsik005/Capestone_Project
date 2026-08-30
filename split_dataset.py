"""
Turns two flat folders of images (one real, one fake) into the
train/val/{real,fake} layout that train_deepfake_model.py expects.

Before:
    raw_data/
        real/  img1.jpg, img2.jpg, ...
        fake/  img1.jpg, img2.jpg, ...

After (created by this script):
    data/
        train/real/...  train/fake/...
        val/real/...    val/fake/...

Usage:
    python split_dataset.py --real raw_data/real --fake raw_data/fake --out data --val-split 0.2
"""

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_class(src_dir: Path, out_dir: Path, class_name: str, val_split: float, seed: int):
    files = [p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise SystemExit(f"No images found in {src_dir}")

    random.Random(seed).shuffle(files)
    n_val = max(1, int(len(files) * val_split))
    val_files = files[:n_val]
    train_files = files[n_val:]

    for split_name, split_files in [("train", train_files), ("val", val_files)]:
        dest = out_dir / split_name / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for f in split_files:
            shutil.copy2(f, dest / f.name)

    print(f"{class_name}: {len(train_files)} train, {len(val_files)} val")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="Folder containing only real images")
    parser.add_argument("--fake", required=True, help="Folder containing only fake images")
    parser.add_argument("--out", default="data", help="Output folder (default: data)")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    split_class(Path(args.real), out_dir, "real", args.val_split, args.seed)
    split_class(Path(args.fake), out_dir, "fake", args.val_split, args.seed)

    print(f"\nDone. Now run:\n  python train_deepfake_model.py --data-dir {out_dir} --epochs 5")


if __name__ == "__main__":
    main()
