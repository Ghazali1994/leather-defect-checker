# app.py
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from anomalib.models import Fastflow
from anomalib.pre_processing import PreProcessor

st.set_page_config(page_title="🧠 AI Leather Defect Detection")
st.title("AI Leather Defect Detection (FastFlow)")

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model():
    # Use anomalib Fastflow model (anomalib>=2.3.2)
    model = Fastflow(backbone="resnet18")
    model.eval()
    return model

model = load_model()

# -------------------------
# Image preprocessing
# -------------------------
def preprocess_image(image: Image.Image):
    """
    Converts PIL image to tensor compatible with Fastflow.
    """
    image = image.resize((256, 256))
    image_np = np.array(image).astype(np.float32) / 255.0
    # shape (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return image_tensor

# -------------------------
# Detect defects
# -------------------------
def detect_defects(image: Image.Image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        anomaly_map = output["anomaly_map"].squeeze().cpu().numpy()
        # Normalize for display
        heatmap = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        heatmap = (heatmap * 255).astype(np.uint8)

    # Simple threshold for bounding boxes
    thresh = 25
    mask = heatmap > thresh
    coords = np.column_stack(np.where(mask))
    annotated = np.array(image).copy()

    boxes = []
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        boxes.append((x_min, y_min, x_max, y_max))
        # Draw rectangle
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return annotated, boxes, heatmap

# -------------------------
# Streamlit UI
# -------------------------
option = st.radio("Choose Input", ["Upload Image", "Camera"])
image = None

if option == "Upload Image":
    file = st.file_uploader("Upload leather image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")
else:
    cam = st.camera_input("Capture")
    if cam:
        image = Image.open(cam).convert("RGB")

if image is not None:
    annotated, defects, heatmap = detect_defects(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(annotated, caption="Detected Defects", use_column_width=True)
    with col2:
        st.image(heatmap, caption="Anomaly Heatmap", use_column_width=True)

    st.write(f"### 🧪 {len(defects)} defect(s) found")
    for i, (x_min, y_min, x_max, y_max) in enumerate(defects, 1):
        st.write(f"Defect {i} → Location ({x_min},{y_min}) Size {x_max-x_min}x{y_max-y_min}")
