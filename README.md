# Image Recognition Starter (Local, Privacy-Safe)

This is a **no-cloud**, laptop-friendly starter to build an image recognition system for merchandising images using **YOLOv8 (Ultralytics)** and **Streamlit**. Nothing leaves your machine.

## What you'll do (simple)
1. Put a small set of photos into `dataset/images_raw/` and matching labels into `dataset/labels_raw/` (we explain below).
2. Run one script to split your data and generate a `data.yaml`.
3. Train with one command.
4. Test with one command **or** use the local Streamlit app.

## Quick start (Windows/Mac)
> Open a terminal (Command Prompt or PowerShell on Windows).

### 0) Create & activate a virtual environment (recommended)
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 1) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Prepare your dataset
- Use **LabelImg** (or any labeler that exports YOLO format) to annotate bounding boxes.
- Place your images in: `dataset/images_raw/`
- Place your YOLO label `.txt` files in: `dataset/labels_raw/` (each file name must match the image name).

Then run:
```bash
python scripts/prepare_dataset.py --val_split 0.2
```
This will:
- Validate your pairs (image + label)
- Split into `dataset/train/*` and `dataset/val/*`
- Auto-generate `data.yaml` at the project root

### 3) Train your model
```bash
python scripts/train_yolo.py --epochs 50 --imgsz 640 --model yolov8n.pt
```
- We default to **yolov8n.pt** (the small model). You can change to `yolov8s.pt` later.
- Results and best weights will be in `runs/detect/<timestamp>/weights/best.pt`

### 4) Predict on a test image/folder
```bash
python scripts/predict_yolo.py --weights runs/detect/*/weights/best.pt --source dataset/val/images
```
Annotated results will be saved under `outputs/`.

### 5) Optional: Run a local app (Streamlit)
```bash
streamlit run scripts/app.py
```
- Upload an image; it will run locally and show detections.
- **No internet** and **no cloud uploads.**

---

## Privacy helpers
Two optional tools to harden privacy before training or prediction:
- `privacy_tools.py#strip_exif` → removes EXIF metadata (e.g., device, time, sometimes GPS).
- `privacy_tools.py#blur_faces_inplace` → blurs detected faces using OpenCV's built-in cascade.

Examples:
```bash
# Strip EXIF on all raw images
python scripts/privacy_tools.py --strip-exif --images dataset/images_raw

# Blur faces on all raw images (writes in-place)
python scripts/privacy_tools.py --blur-faces --images dataset/images_raw
```

---

## Labeling refresher (YOLO format)
For each image `photo.jpg`, there must be `photo.txt` in YOLO format:
```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```
- Values are **normalized 0..1**.
- `class_id` is **0-based** index into `names` in `data.yaml`.

**Tip:** Start with **1–3 classes** and **20–50 images** total to prove the concept.

---

## Updating classes
Open `data.yaml` (auto-created) and edit:
```yaml
nc: 2
names: ["Header", "POS_Stand"]
```
Make sure `nc` matches the number of names.

---

## Troubleshooting
- If Ultralytics complains about CUDA/GPU, training will still work on CPU (just slower).
- If predictions show no boxes, check:
  - Class names in `data.yaml`
  - Label files align with images (exact same filename base)
  - You didn't split away **all** examples of a class into val set
- If Streamlit can't find weights, point it to `runs/detect/.../weights/best.pt` or put a copy in `/weights`.

---

## Credits
- [Ultralytics/YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)

> Built for a simple, local-first workflow with client privacy in mind.
