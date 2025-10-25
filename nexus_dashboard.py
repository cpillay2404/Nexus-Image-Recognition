import streamlit as st
import os
import base64
import glob
from ultralytics import YOLO
from PIL import Image, ImageDraw

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(page_title="Nexus | Vision – Image Recognition", layout="wide")

# --- Paths ---
image_folder = r"C:\Users\CarinPillay\OneDrive - Meridian Group\image-recognition-starter\image-recognition-starter\dataset_clean\images\val"
model_path = r"C:\Users\CarinPillay\OneDrive - Meridian Group\image-recognition-starter\image-recognition-starter\runs\detect\clipstrip_clean_final\weights\best.pt"
logo_path = r"C:\Users\CarinPillay\OneDrive - Meridian Group\image-recognition-starter\Meridian Nexus.png"

# =========================================================
# HEADER
# =========================================================
with open(logo_path, "rb") as f:
    logo_base64 = base64.b64encode(f.read()).decode("utf-8")

st.markdown(
    f"""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        background-color:#023A6F;
        padding:12px 25px;
        border-radius:8px;
    ">
        <h1 style="color:white; font-size:32px; margin:0;">
            <span style='color:#F57C00;'>Nexus</span> | Vision – Image Recognition
        </h1>
        <img src="data:image/png;base64,{logo_base64}" 
             alt="Meridian Nexus Logo" 
             style="height:85px; width:auto; margin-right:10px;">
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("---")

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_local_images(folder):
    """List local (non-cloud) image files."""
    all_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        all_files.extend(glob.glob(os.path.join(folder, ext)))
    local_files = [f for f in all_files if os.path.exists(f) and os.path.getsize(f) > 0]
    return sorted(local_files, key=os.path.getmtime, reverse=True)

def draw_boxes(image, results):
    """Draw YOLO bounding boxes on the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for box in results.boxes:
        coords = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        draw.rectangle(coords, outline="red", width=4)
        draw.text((coords[0], coords[1]-10), f"Clipstrip {conf:.2f}", fill="red")
    return img

# =========================================================
# LOAD MODEL
# =========================================================
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"❌ Could not load YOLO model: {e}")
    st.stop()

# =========================================================
# SCAN IMAGES + RUN DETECTIONS
# =========================================================
image_paths = get_local_images(image_folder)
num_images = len(image_paths)
st.caption(f"🖼️ Found {num_images} available image(s)")

if num_images == 0:
    st.warning("No available images found.")
    st.stop()

detections = []
image_results = []  # (path, has_detection, avg_conf, num_detections, results)

for path in image_paths:
    try:
        results = model.predict(path, verbose=False)
        num_detections = len(results[0].boxes)
        has_detection = num_detections > 0
        confs = [float(box.conf[0]) for box in results[0].boxes] if has_detection else []
        avg_conf = sum(confs) / len(confs) * 100 if confs else 0
        image_results.append((path, has_detection, avg_conf, num_detections, results[0]))
        if has_detection:
            detections.extend(confs)
    except Exception:
        continue

total_detections = len(detections)
avg_confidence = round(sum(detections) / len(detections) * 100, 2) if detections else 0.0
compliance = round(min((total_detections / max(num_images, 1)) * 100, 100), 1)

num_compliant = sum(1 for _, has_det, _, _, _ in image_results if has_det)
num_noncompliant = num_images - num_compliant
total_clipstrips = sum(num_det for _, _, _, num_det, _ in image_results)

# =========================================================
# SUMMARY
# =========================================================
with st.expander("📊 Summary Overview", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Images", num_images)
    col2.metric("Detections", total_detections)
    col3.metric("Avg Confidence (%)", avg_confidence)
    col4.metric("Compliance (%)", compliance)

    # ✅ new summary line
    st.markdown(
        f"<div style='margin-top:10px;font-weight:600;font-size:16px;color:#023A6F;'>"
        f"✅ {num_compliant} Compliant | ❌ {num_noncompliant} Non-compliant | 🧩 {total_clipstrips} Clipstrips total"
        f"</div>",
        unsafe_allow_html=True,
    )

st.write("---")

# =========================================================
# PAGINATION
# =========================================================
images_per_page = 4
page_num = st.session_state.get("page_num", 0)

col_prev, col_next = st.columns([1, 1])
with col_prev:
    if st.button("⬅ Previous") and page_num > 0:
        page_num -= 1
with col_next:
    if st.button("Next ➡") and (page_num + 1) * images_per_page < num_images:
        page_num += 1
st.session_state.page_num = page_num

start_idx = page_num * images_per_page
end_idx = start_idx + images_per_page
page_images = image_results[start_idx:end_idx]

st.markdown(f"📸 **Viewing images {start_idx + 1} to {min(end_idx, num_images)} of {num_images}**")

# =========================================================
# DISPLAY EACH IMAGE WITH LABELS
# =========================================================
cols = st.columns(4)
for i, (path, has_detection, avg_conf, num_detections, results) in enumerate(page_images):
    compliance_label = (
        f"<div style='color:green;font-weight:600;font-size:16px;'>✅ Compliant (Clipstrip detected – {avg_conf:.1f}%)</div>"
        if has_detection
        else "<div style='color:red;font-weight:600;font-size:16px;'>❌ Non-compliant (No Clipstrip detected)</div>"
    )
    count_label = f"<div style='color:#555;font-weight:500;font-size:14px;'>🧩 {num_detections} Clipstrip(s) Detected</div>"

    try:
        img = Image.open(path)
        if has_detection:
            img = draw_boxes(img, results)
        with cols[i % 4]:
            st.markdown(compliance_label, unsafe_allow_html=True)
            st.markdown(count_label, unsafe_allow_html=True)
            st.image(img, caption=os.path.basename(path), use_container_width=True)
    except Exception:
        continue

st.write("---")

# =========================================================
# REFRESH BUTTON
# =========================================================
if st.button("🔁 Refresh Now"):
    st.cache_data.clear()
    st.rerun()

