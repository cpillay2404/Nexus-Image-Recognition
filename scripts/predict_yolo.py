import argparse, os, shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained weights (best.pt)")
    ap.add_argument("--source", default="dataset/val/images", help="Image file or folder")
    ap.add_argument("--confidence", type=float, default=0.25)
    ap.add_argument("--save_dir", default="outputs")
    args = ap.parse_args()

    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.confidence, save=True, save_txt=False)

    # Move predicted images to outputs/<session>/
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    for r in results:
        if hasattr(r, "save_dir"):
            # Each result has a session save dir; copy files
            for f in Path(r.save_dir).glob("*.*"):
                shutil.copy2(f, Path(args.save_dir) / f.name)
    print(f"[OK] Saved annotated images to {args.save_dir}")

if __name__ == "__main__":
    main()
