import streamlit as st
from PIL import Image
import numpy as np
import cv2
import torch
import os
import urllib.request
from anomalib.models import Fastflow

# -------------------------
# Streamlit page settings
# -------------------------
st.set_page_config(
    page_title="🧠 AI Leather Defect Detection",
    layout="wide"
)

st.title("🧠 AI Leather Defect Detection (FastFlow)")

# -------------------------
# Device
# Streamlit Cloud uses CPU
# -------------------------
device = "cpu"

# -------------------------
# Model download
# -------------------------
MODEL_URL = (
    "https://huggingface.co/openvino/"
    "anomalib-fastflow-mvtec-leather/resolve/main/model.ckpt"
)

MODEL_PATH = "model.ckpt"


def download_model():
    """Download model checkpoint if not available."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model... ⏳"):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        st.success("✅ Model downloaded successfully!")


# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model():
    """Load FastFlow model."""
    download_model()

    model = Fastflow.load_from_checkpoint(MODEL_PATH)

    model.eval()
    model.to(device)

    return model


model = load_model()

# -------------------------
# Preprocess image
# -------------------------
def preprocess_image(image: Image.Image):
    """
    Resize and normalize image.
    """

    image = image.resize((256, 256))

    image_np = np.array(image).astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    image_norm = (image_np - mean) / std

    image_tensor = (
        torch.from_numpy(image_norm)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )

    return image_tensor.to(device), (image_np * 255).astype(np.uint8)


# -------------------------
# Detect defects
# -------------------------
def detect_defects(image: Image.Image, thresh: int):
    """
    Run anomaly detection on image.
    """

    input_tensor, resized_img = preprocess_image(image)

    with torch.no_grad():

        predictions = model(input_tensor)

        # Compatible with different anomalib versions
        if isinstance(predictions, dict):
            anomaly_map = predictions["anomaly_map"]
        else:
            anomaly_map = predictions.anomaly_map

    # Convert tensor to numpy
    anomaly_map = anomaly_map.squeeze().cpu().numpy()

    # -------------------------
    # Normalize heatmap
    # -------------------------
    heatmap = (
        anomaly_map - anomaly_map.min()
    ) / (
        anomaly_map.max() - anomaly_map.min() + 1e-8
    )

    heatmap = (heatmap * 255).astype(np.uint8)

    # -------------------------
    # Create binary mask
    # -------------------------
    mask = heatmap > thresh

    # -------------------------
    # Find contours
    # -------------------------
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    annotated = resized_img.copy()

    boxes = []

    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)

        # Remove tiny noisy regions
        if w * h < 100:
            continue

        boxes.append((x, y, x + w, y + h))

        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # -------------------------
    # Create heatmap overlay
    # -------------------------
    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        annotated,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return annotated, boxes, heatmap_color, overlay


# -------------------------
# UI Controls
# -------------------------
option = st.radio(
    "Choose Input",
    ["Upload Image", "Camera"]
)

thresh = st.slider(
    "Detection Sensitivity",
    min_value=0,
    max_value=255,
    value=25
)

image = None

# -------------------------
# Upload image
# -------------------------
if option == "Upload Image":

    file = st.file_uploader(
        "Upload leather image",
        type=["jpg", "jpeg", "png"]
    )

    if file is not None:
        image = Image.open(file).convert("RGB")

# -------------------------
# Camera input
# -------------------------
else:

    cam = st.camera_input("Capture leather image")

    if cam is not None:
        image = Image.open(cam).convert("RGB")


# -------------------------
# Run detection
# -------------------------
if image is not None:

    with st.spinner("Analyzing image... 🔍"):

        annotated, defects, heatmap, overlay = detect_defects(
            image,
            thresh
        )

    # -------------------------
    # Display results
    # -------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.image(
            annotated,
            caption="Detected Defects",
            use_container_width=True
        )

    with col2:
        st.image(
            overlay,
            caption="Heatmap Overlay",
            use_container_width=True
        )

    # -------------------------
    # Defect summary
    # -------------------------
    st.write(f"## 🧪 {len(defects)} defect(s) found")

    if len(defects) == 0:
        st.success("✅ No major defects detected.")

    for i, (x_min, y_min, x_max, y_max) in enumerate(defects, start=1):

        width = x_max - x_min
        height = y_max - y_min

        st.write(
            f"""
            ### Defect {i}
            - Location: ({x_min}, {y_min})
            - Size: {width} × {height}
            """
        )
