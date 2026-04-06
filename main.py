import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.filters import gaussian
from skimage.color import rgb2gray

st.set_page_config(page_title="AI Leather Defect Detection")
st.title("🧠 AI Leather Defect Detection Tool")

# -----------------------------
# Detection (lightweight)
# -----------------------------
def detect_defects(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # smooth background
    blur = cv2.GaussianBlur(gray, (21,21), 0)

    # anomaly map
    diff = cv2.absdiff(gray, blur)

    # normalize
    anomaly_map = cv2.normalize(
        diff, None, 0, 255, cv2.NORM_MINMAX
    )

    # threshold
    _, thresh = cv2.threshold(
        anomaly_map,
        25,
        255,
        cv2.THRESH_BINARY
    )

    # remove noise
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel
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

        if area > 300:
            x,y,w,h = cv2.boundingRect(cnt)
            defects.append((x,y,w,h))

            cv2.rectangle(
                annotated,
                (x,y),
                (x+w,y+h),
                (0,255,0),
                2
            )

    return annotated, defects, anomaly_map


# -----------------------------
# Input
# -----------------------------
option = st.radio(
    "Choose Input",
    ["Upload Image", "Camera"]
)

image = None

if option == "Upload Image":
    file = st.file_uploader(
        "Upload leather image",
        type=["jpg","png","jpeg"]
    )

    if file:
        bytes_data = np.asarray(
            bytearray(file.read()),
            dtype=np.uint8
        )
        image = cv2.imdecode(bytes_data,1)

else:
    cam = st.camera_input("Capture")

    if cam:
        bytes_data = np.asarray(
            bytearray(cam.getvalue()),
            dtype=np.uint8
        )
        image = cv2.imdecode(bytes_data,1)


# -----------------------------
# Run
# -----------------------------
if image is not None:

    annotated, defects, heatmap = detect_defects(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            caption="Detected Defects"
        )

    with col2:
        st.image(
            heatmap,
            caption="Anomaly Heatmap"
        )

    st.markdown(f"### 🧪 {len(defects)} defects found")

    for i,(x,y,w,h) in enumerate(defects,1):
        st.write(
            f"Defect {i} → Location ({x},{y}) | Size {w}x{h}"
        )
