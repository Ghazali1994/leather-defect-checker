import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch

from anomalib.models import Fastflow

st.set_page_config(page_title="AI Leather Defect Detection")
st.title("🧠 AI Leather Defect Detection (FastFlow)")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    model = Fastflow(backbone="resnet18")
    model.eval()
    return model

model = load_model()


# -------------------------
# Detect defects
# -------------------------
def detect_defects(image):

    img = np.array(image)
    resized = cv2.resize(img, (256, 256))

    tensor = (
        torch.from_numpy(resized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float() / 255
    )

    with torch.no_grad():
        output = model(tensor)

    # anomaly heatmap
    heatmap = output["anomaly_map"][0].cpu().numpy()

    # resize back to original
    heatmap = cv2.resize(
        heatmap,
        (image.width, image.height)
    )

    # normalize
    heatmap = (heatmap - heatmap.min()) / (
        heatmap.max() - heatmap.min() + 1e-8
    )

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)

    # threshold
    thresh = cv2.threshold(
        heatmap_uint8,
        180,
        255,
        cv2.THRESH_BINARY
    )[1]

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    annotated = img.copy()
    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # remove tiny noise
        if w * h > 200:
            boxes.append((x, y, w, h))
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    return annotated, heatmap_uint8, boxes


# -------------------------
# UI
# -------------------------
option = st.radio(
    "Choose Input",
    ["Upload Image", "Camera"]
)

image = None

if option == "Upload Image":
    file = st.file_uploader("Upload leather image")
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

    annotated, heatmap, boxes = detect_defects(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            annotated,
            caption="Detected Defects",
            use_container_width=True
        )

    with col2:
        st.image(
            heatmap,
            caption="Anomaly Heatmap",
            clamp=True,
            use_container_width=True
        )

    st.write(f"### 🔎 {len(boxes)} defects found")

    for i, (x, y, w, h) in enumerate(boxes, 1):
        st.write(
            f"Defect {i} → Location ({x},{y}) Size {w}x{h}"
        )
