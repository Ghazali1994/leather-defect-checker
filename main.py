import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from anomalib.models import Padim
from anomalib.data.utils import read_image
from streamlit_paste_button import paste_image_button

# --- Streamlit Page Config ---
st.set_page_config(page_title="AI Leather Defect Detection Tool")
st.title("AI Leather Defect Detection Tool")
st.write("Upload or capture an image to detect leather defects using Anomalib.")

# --- Load Model ---
@st.cache_resource
def load_model():
    model = Padim(backbone="resnet18")
    model.to("cpu")
    model.eval()
    return model

model = load_model()

# --- Detection Function ---
def detect_defects_anomalib(image):
    # Convert BGR -> RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    # Preprocess
    input_tensor = read_image(pil_image, resize=(256, 256)).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        anomaly_map = output["anomaly_map"].squeeze().cpu().numpy()

    # Normalize
    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    # Resize to original
    anomaly_map = cv2.resize(
        anomaly_map,
        (image.shape[1], image.shape[0])
    )

    # Threshold
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

    defects = []
    annotated_image = image.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            defects.append((x, y, w, h))

            cv2.rectangle(
                annotated_image,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2
            )

    return annotated_image, defects


# --- Input Options ---
option = st.radio(
    "Choose input method:",
    ["Upload Image", "Capture from Camera"]
)

image = None

# --- Upload Image ---
if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# --- Camera Input ---
elif option == "Capture from Camera":
    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:
        bytes_data = camera_image.getvalue()

        file_bytes = np.asarray(
            bytearray(bytes_data),
            dtype=np.uint8
        )

        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# --- Paste from Clipboard (SAFE) ---
pasted_image = paste_image_button("📋 Paste Image from Clipboard")

if pasted_image is not None:
    pasted_np = np.array(pasted_image)

    # RGBA
    if len(pasted_np.shape) == 3 and pasted_np.shape[2] == 4:
        image = cv2.cvtColor(pasted_np, cv2.COLOR_RGBA2BGR)

    # RGB
    elif len(pasted_np.shape) == 3 and pasted_np.shape[2] == 3:
        image = cv2.cvtColor(pasted_np, cv2.COLOR_RGB2BGR)

    # GRAYSCALE
    elif len(pasted_np.shape) == 2:
        image = cv2.cvtColor(pasted_np, cv2.COLOR_GRAY2BGR)


# --- Run Detection ---
if image is not None:
    annotated_image, defects = detect_defects_anomalib(image)

    st.image(
        cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
        caption="Annotated Image",
        use_column_width=True
    )

    st.markdown(f"### 🧪 {len(defects)} defect(s) found")

    for idx, (x, y, w, h) in enumerate(defects, 1):
        st.write(
            f"**Defect {idx}:** "
            f"Location: (x={x}, y={y}), "
            f"Size: {w}x{h}, "
            f"Area: {w*h}"
        )
