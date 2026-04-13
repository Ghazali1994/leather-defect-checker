import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
from anomalib.models import Fastflow

st.set_page_config(page_title="🧠 AI Leather Defect Detection", layout="wide")
st.title("🧠 AI Leather Defect Detection (FastFlow)")

# Device

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------

# Load model

# -------------------------

@st.cache_resource
def load_model():
    model = Fastflow(backbone="resnet18")
    model.eval()
    model.to(device)
    return model

model = load_model()

# -------------------------

# Preprocess image

# -------------------------

def preprocess_image(image: Image.Image):
    image = image.resize((256, 256))
    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return image_tensor.to(device), image_np

# -------------------------

# Detect defects

# -------------------------

def detect_defects(image: Image.Image, thresh):
    input_tensor, resized_img = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        anomaly_map = output.anomaly_map.squeeze().cpu().numpy()

    heatmap = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-8
    )
    heatmap = (heatmap * 255).astype(np.uint8)

    mask = heatmap > thresh

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    annotated = (resized_img * 255).astype(np.uint8).copy()
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w * h < 50:
            continue

        boxes.append((x, y, x + w, y + h))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(annotated, 0.6, heatmap_color, 0.4, 0)

    return annotated, boxes, heatmap_color, overlay


# -------------------------

# UI

# -------------------------

option = st.radio("Choose Input", ["Upload Image", "Camera"])
thresh = st.slider("Detection Sensitivity", 0, 255, 25)

image = None

if option == "Upload Image":
    file = st.file_uploader("Upload leather image", type=["jpg", "jpeg", "png"])
    if file:
        image = Image.open(file).convert("RGB")
else:
    cam = st.camera_input("Capture")
    if cam:
        image = Image.open(cam).convert("RGB")

# -------------------------

# Run detection

# -------------------------

if image is not None:
    annotated, defects, heatmap, overlay = detect_defects(image, thresh)

    col1, col2 = st.columns(2)

    with col1:
        st.image(annotated, caption="Detected Defects", use_column_width=True)

    with col2:
        st.image(overlay, caption="Heatmap Overlay", use_column_width=True)

    st.write(f"### 🧪 {len(defects)} defect(s) found")

    for i, (x_min, y_min, x_max, y_max) in enumerate(defects, 1):
        st.write(
            f"Defect {i} → Location ({x_min},{y_min}) "
            f"Size {x_max-x_min}x{y_max-y_min}"
        )
