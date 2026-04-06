import streamlit as st
import numpy as np
from PIL import Image
from skimage.filters import gaussian

st.set_page_config(page_title="AI Leather Defect Detection")
st.title("🧠 AI Leather Defect Detection Tool")


def detect_defects(image):

    img = np.array(image)

    # convert to grayscale
    gray = np.mean(img, axis=2)

    # blur background
    blur = gaussian(gray, sigma=5)

    # anomaly map
    diff = np.abs(gray - blur)

    # normalize
    anomaly = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
    anomaly = (anomaly * 255).astype(np.uint8)

    # threshold
    mask = anomaly > 25

    # find bounding boxes
    coords = np.column_stack(np.where(mask))

    boxes = []
    annotated = img.copy()

    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        boxes.append((x_min, y_min, x_max-x_min, y_max-y_min))

        # draw rectangle
        annotated[y_min:y_max, x_min:x_min+2] = [0,255,0]
        annotated[y_min:y_max, x_max-2:x_max] = [0,255,0]
        annotated[y_min:y_min+2, x_min:x_max] = [0,255,0]
        annotated[y_max-2:y_max, x_min:x_max] = [0,255,0]

    return annotated, boxes, anomaly


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


if image is not None:

    annotated, defects, heatmap = detect_defects(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(annotated, caption="Detected Defects")

    with col2:
        st.image(heatmap, caption="Anomaly Heatmap")

    st.write(f"### 🧪 {len(defects)} defects found")

    for i,(x,y,w,h) in enumerate(defects,1):
        st.write(f"Defect {i} → Location ({x},{y}) Size {w}x{h}")
