import argparse, os, sys
from pathlib import Path
from PIL import Image
import piexif
import cv2

def strip_exif_inplace(image_path: Path):
    try:
        img = Image.open(image_path)
        data = list(img.getdata())
        img_no_exif = Image.new(img.mode, img.size)
        img_no_exif.putdata(data)
        img_no_exif.save(image_path)
        return True
    except Exception as e:
        print(f"[WARN] EXIF strip failed for {image_path}: {e}")
        return False

def blur_faces_inplace(image_path: Path):
    try:
        # Use OpenCV's built-in cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[WARN] Cannot read {image_path}")
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = img[y:y+h, x:x+w]
            roi = cv2.GaussianBlur(roi, (51, 51), 30)
            img[y:y+h, x:x+w] = roi
        cv2.imwrite(str(image_path), img)
        return True
    except Exception as e:
        print(f"[WARN] Face blur failed for {image_path}: {e}")
        return False

def process_folder(path: Path, strip_exif=False, blur_faces=False):
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    images = [p for p in path.rglob("*") if p.suffix in exts]
    ok = 0
    for p in images:
        if strip_exif:
            strip_exif_inplace(p)
        if blur_faces:
            blur_faces_inplace(p)
        ok += 1
    print(f"[OK] Processed {ok} images in {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Folder of images to process")
    ap.add_argument("--strip-exif", action="store_true", help="Remove EXIF metadata from images")
    ap.add_argument("--blur-faces", action="store_true", help="Blur detected faces in images (in-place)")
    args = ap.parse_args()

    process_folder(Path(args.images), strip_exif=args.strip_exif, blur_faces=args.blur_faces)
