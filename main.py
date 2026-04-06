import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from anomalib.deploy import TorchInferencer

st.set_page_config(page_title="AI Leather Defect Detection Tool")
st.title("🧠 AI Leather Defect Detection Tool")

@st.cache_resource
def load_model():
    model = TorchInferencer(
        path="https://github.com/openvinotoolkit/anomalib/releases/download/v0.5.0/padim_mvtec_leather.pt"
    )
    return model

model = load_model()

transform = T.Compose([
    T.Resize((256,256)),
    T.ToTensor()
])

def detect(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    tensor = transform(pil).unsqueeze(0)

    with torch.no_grad():
        output = model.predict(tensor)

    anomaly_map = output.anomaly_map.squeeze().cpu().numpy()

    anomaly_map = (anomaly_map - anomaly_map.min()) / (
        anomaly_map.max() - anomaly_map.min() + 1e-8
    )

    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    anomaly_map = cv2.resize(
        anomaly_map,
        (image.shape[1], image.shape[0])
    )

    _, thresh = cv2.threshold(anomaly_map, 200, 255, cv2.THRESH_BINARY)

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    annotated = image.copy()
    defects = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 150:
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


option = st.radio(
    "Input Method",
    ["Upload","Camera"]
)

image = None

if option == "Upload":
    file = st.file_uploader("Upload image")

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


if image is not None:

    annotated, defects, heatmap = detect(image)

    col1,col2 = st.columns(2)

    with col1:
        st.image(
            cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB),
            caption="Detected Defects"
        )

    with col2:
        st.image(heatmap,caption="Anomaly Heatmap")

    st.write(f"### 🧪 {len(defects)} defects found")

    for i,(x,y,w,h) in enumerate(defects,1):
        st.write(
            f"Defect {i}: Location=({x},{y}) Size={w}x{h}"
        )
