import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image, ImageDraw
import glob
import numpy as np

# --- CONFIG ---
st.set_page_config(page_title="Clipstrip Learning Insights", layout="wide")

# --- HEADER ---
st.markdown(
    """
    <h1 style="text-align:center; color:#0078D4;">
        🧠 Carin's Image Recognition Dashboard
    </h1>
    <p style="text-align:center; font-size:16px;">
        Using your trained YOLO model to analyze training and validation datasets.
    </p>
    """,
    unsafe_allow_html=True
)

# --- PATHS ---
model_path = "C:/Users/CarinPillay/OneDrive - Meridian Group/image-recognition-starter/image-recognition-starter/runs/detect/clipstrip_clean_final/weights/best.pt"

train_dir = "C:/Users/CarinPillay/OneDrive - Meridian Group/image-recognition-starter/image-recognition-starter/dataset_clean/images/train"
val_dir = "C:/Users/CarinPillay/OneDrive - Meridian Group/image-recognition-starter/image-recognition-starter/dataset_clean/images/val"

# --- LOAD MODEL ---
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"❌ Could not load YOLO model: {e}")
    st.stop()

# --- GATHER IMAGES ---
train_images = sorted(glob.glob(os.path.join(train_dir, "*.jpg")) + glob.glob(os.path.join(train_dir, "*.png")))
val_images = sorted(glob.glob(os.path.join(val_dir, "*.jpg")) + glob.glob(os.path.join(val_dir, "*.png")))
all_images = train_images + val_images

if not all_images:
    st.warning("⚠️ No images found in dataset folders.")
    st.stop()

# --- SUMMARY CALCULATIONS ---
st.sidebar.header("📊 Summary Statistics")
total_images = len(all_images)
total_detections = 0
confidences = []

# quick scan to compute overall stats
for img_path in all_images:
    results = model.predict(source=img_path, conf=0.25, verbose=False)
    det_count = len(results[0].boxes)
    total_detections += det_count
    if det_count > 0:
        confs = [float(box.conf) for box in results[0].boxes]
        confidences.extend(confs)

# compute averages
avg_conf = np.mean(confidences) if confidences else 0.0
compliant_images = sum(1 for img_path in all_images if len(model.predict(source=img_path, conf=0.25, verbose=False)[0].boxes) > 0)
non_compliant_images = total_images - compliant_images
compliance_rate = (compliant_images / total_images) * 100

# --- DISPLAY SUMMARY PANEL ---
colA, colB, colC, colD, colE = st.columns(5)
colA.metric("🏬 Total Stores", f"{len(set([os.path.basename(os.path.dirname(p)) for p in all_images]))}")
colB.metric("🖼️ Total Images", total_images)
colC.metric("🧩 Clipstrips Detected", total_detections)
colD.metric("📈 Avg Confidence", f"{avg_conf:.2f}")
colE.metric("✅ Compliance", f"{compliance_rate:.1f}%")

st.divider()

# --- STATE ---
if "img_index" not in st.session_state:
    st.session_state.img_index = 0

# --- NAVIGATION ---
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬅️ Previous") and st.session_state.img_index > 0:
        st.session_state.img_index -= 1
with col3:
    if st.button("Next ➡️") and st.session_state.img_index < len(all_images) - 1:
        st.session_state.img_index += 1

current_img = all_images[st.session_state.img_index]

# --- DETECT & DRAW ---
st.subheader(f"📸 Analyzing image {st.session_state.img_index + 1}/{len(all_images)}")
results = model.predict(source=current_img, conf=0.25, verbose=False)

image = Image.open(current_img).convert("RGB")
draw = ImageDraw.Draw(image)

detections = 0
conf_list = []
for box in results[0].boxes:
    coords = box.xyxy[0].tolist()
    label = model.names[int(box.cls)]
    conf = float(box.conf)
    detections += 1
    conf_list.append(conf)
    draw.rectangle(coords, outline="red", width=3)
    draw.text((coords[0], coords[1] - 15), f"{label} {conf:.2f}", fill="red")

# --- COMPLIANCE ---
avg_conf_img = np.mean(conf_list) if conf_list else 0
if detections >= 1:
    compliance = f"✅ Compliant (Clipstrip detected, avg conf {avg_conf_img:.2f})"
    color = "green"
else:
    compliance = "❌ Non-Compliant (No clipstrip found)"
    color = "red"

# --- DISPLAY ---
st.markdown(f"<h3 style='color:{color}; text-align:center;'>{compliance}</h3>", unsafe_allow_html=True)
st.image(image, caption=f"Detections: {detections} | Avg Conf: {avg_conf_img:.2f}", use_container_width=True)

st.divider()
st.markdown(
    f"""
    <div style="text-align:center; font-size:15px;">
        <b>Total Images:</b> {total_images} |
        <b>Current:</b> {st.session_state.img_index + 1} |
        <b>Detections in this image:</b> {detections}
    </div>
    """,
    unsafe_allow_html=True
)
