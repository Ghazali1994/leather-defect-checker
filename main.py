import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from anomalib.deploy import TorchInferencer

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Leather Defect Detection Tool")
st.title("🧠 AI Leather Defect Detection Tool")
st.write("Upload or capture image to detect leather defects")

# -----------------------------
# Load Pretrained Model
# -----------------------------
@st.cache_resource
def load_model():
    model = TorchInferencer(
        path="https://github.com/openvinotoolkit/anomalib/releases/download/v0.5.0/padim_mvtec_leather.pt"
    )
    return model

model = load_model()

# -----------------------------
# Transform
# -----------------------------
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor()
])

# -----------------------------
# Detection
# -----------------------------
def detect_defects_anomalib(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    input_tensor = transform(pil).unsqueeze(0)

    with torch.no_grad():
        output = model.predict(input_tensor)

    anomaly_map = output.anomaly_map.squeeze().cpu().numpy()

    # Normalize
    anomaly_map = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-8
    )

    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    anomaly_map = cv2.resize(
        anomaly_map,
        (image.shape[1], image.shape[0])
    )

    # threshold
    _, thresh = cv2.threshold(
        anomaly_map,
        200,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    annotated = image.copy()
    defects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 150:   # filter noise
            x, y, w, h = cv2.boundingRect(cnt)
            defects.append((x, y, w, h))

            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    return annotated, defects, anomaly_map


# -----------------------------
# Input
# -----------------------------
option = st.radio(
    "Choose input method:",
    ["Upload Image", "Capture from Camera"]
)

image = None

# Upload
if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# Camera
else:
    camera_image = st.camera_input("Capture")

    if camera_image is not None:
        bytes_data = camera_image.getvalue()

        file_bytes = np.asarray(
            bytearray(bytes_data),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# -----------------------------
# Run Detection
# -----------------------------
if image is not None:

    annotated, defects, anomaly_map = detect_defects_anomalib(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption="Detected Defects",
            use_column_width=True
        )

    with col2:
        st.image(
            anomaly_map,
            caption="Anomaly Heatmap",
            use_column_width=True
        )

    st.markdown(f"### 🧪 {len(defects)} defect(s) found")

    for i, (x, y, w, h) in enumerate(defects, 1):
        st.write(
            f"Defect {i} → "
            f"Location: ({x},{y}) | "
            f"Size: {w}x{h} | "
            f"Area: {w*h}"
        )
