import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io, os

st.set_page_config(page_title="Local Shelf Recognition", page_icon="🛒", layout="centered")
st.title("🛒 Merchandising Image Recognition (Local)")

st.write("All processing happens on your machine. No uploads leave your device.")

default_weights = None
# Try to auto-find a recent best.pt in runs/detect
import pathlib
runs = sorted(pathlib.Path("runs/detect").glob("*/weights/best.pt"), key=os.path.getmtime, reverse=True)
if runs:
    default_weights = str(runs[0])

weights = st.text_input("Path to weights (.pt)", value=default_weights or "weights/best.pt")
conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.01)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if weights and uploaded:
    try:
        model = YOLO(weights)
    except Exception as e:
        st.error(f"Could not load weights: {e}")
    else:
        img = Image.open(uploaded).convert("RGB")
        res = model(img, conf=conf)
        # Show annotated
        annotated = res[0].plot()  # numpy array
        st.image(annotated, caption="Detections", use_column_width=True)

st.markdown("---")
st.caption("Tip: Put a copy of your trained weights at `weights/best.pt` for convenience.")
