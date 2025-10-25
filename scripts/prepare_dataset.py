import argparse, os, shutil, random, yaml, glob
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

def pair_files(images_dir, labels_dir):
    images = []
    for p in Path(images_dir).iterdir():
        if p.suffix in IMG_EXTS:
            lab = Path(labels_dir) / (p.stem + ".txt")
            if lab.exists():
                images.append((p, lab))
    return images

def split_pairs(pairs, val_split=0.2, seed=42):
    random.Random(seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_split)) if len(pairs) > 1 else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    return train_pairs, val_pairs

def copy_pairs(pairs, out_img_dir, out_lab_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lab_dir, exist_ok=True)
    for img, lab in pairs:
        shutil.copy2(img, Path(out_img_dir) / img.name)
        shutil.copy2(lab, Path(out_lab_dir) / lab.name)

def write_data_yaml(path="data.yaml", nc=1, names=None):
    if names is None:
        names = ["Header"]
    data = {
        "train": "dataset/train/images",
        "val": "dataset/val/images",
        "nc": nc,
        "names": names,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] Wrote {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_raw", default="dataset/images_raw", help="Folder of raw images")
    ap.add_argument("--labels_raw", default="dataset/labels_raw", help="Folder of raw YOLO .txt labels")
    ap.add_argument("--train_out", default="dataset/train", help="Train output base folder")
    ap.add_argument("--val_out", default="dataset/val", help="Val output base folder")
    ap.add_argument("--val_split", type=float, default=0.2, help="Validation split fraction")
    ap.add_argument("--classes", nargs="*", default=["Header"], help="List of class names, order defines class ids")
    args = ap.parse_args()

    pairs = pair_files(args.images_raw, args.labels_raw)
    if not pairs:
        raise SystemExit("No (image,label) pairs found. Check dataset/images_raw and dataset/labels_raw.")

    train_pairs, val_pairs = split_pairs(pairs, args.val_split)
    print(f"Found {len(pairs)} pairs -> train={len(train_pairs)} val={len(val_pairs)}")

    copy_pairs(train_pairs, os.path.join(args.train_out, "images"), os.path.join(args.train_out, "labels"))
    copy_pairs(val_pairs, os.path.join(args.val_out, "images"), os.path.join(args.val_out, "labels"))

    write_data_yaml(path="data.yaml", nc=len(args.classes), names=args.classes)

if __name__ == "__main__":
    main()
