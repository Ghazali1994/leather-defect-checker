import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch

from anomalib.models import Padim
from anomalib.data.utils import read_image

st.set_page_config(page_title="AI Leather Defect Detection")
st.title("🧠 AI Leather Defect Detection (Anomalib)")

# -------------------------
# Load lightweight model
# -------------------------
@st.cache_resource
def load_model():
    model = Padim(backbone="resnet18")
    model.eval()
    return model

model = load_model()

# -------------------------
# Detect defects
# -------------------------
def detect_defects(image):

    img = np.array(image)
    img_resized = cv2.resize(img, (256,256))

    tensor = torch.from_numpy(img_resized).permute(2,0,1).unsqueeze(0).float()/255

    with torch.no_grad():
        output = model(tensor)

    heatmap = output["anomaly_map"][0].cpu().numpy()

    heatmap = cv2.resize(heatmap, (image.width, image.height))
    heatmap = (heatmap - heatmap.min())/(heatmap.max()-heatmap.min()+1e-8)
    heatmap_uint8 = (heatmap*255).astype(np.uint8)

    # threshold
    thresh = cv2.threshold(heatmap_uint8, 180, 255, cv2.THRESH_BINARY)[1]

    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = np.array(image).copy()
    boxes = []

    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h > 200:   # remove noise
            boxes.append((x,y,w,h))
            cv2.rectangle(annotated,(x,y),(x+w,y+h),(0,255,0),2)

    return annotated, heatmap_uint8, boxes

# -------------------------
# UI
# -------------------------
option = st.radio("Choose Input",["Upload","Camera"])

image = None

if option == "Upload":
    file = st.file_uploader("Upload leather image")
    if file:
        image = Image.open(file).convert("RGB")
else:
    cam = st.camera_input("Capture")
    if cam:
        image = Image.open(cam).convert("RGB")

if image:

    annotated, heatmap, boxes = detect_defects(image)

    col1,col2 = st.columns(2)

    with col1:
        st.image(annotated, caption="Detected Defects")

    with col2:
        st.image(heatmap, caption="Anomaly Heatmap")

    st.write(f"### 🔎 {len(boxes)} defects found")

    for i,(x,y,w,h) in enumerate(boxes,1):
        st.write(f"Defect {i}: ({x},{y}) size {w}x{h}")
