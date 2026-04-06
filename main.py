import streamlit as st
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision import models
import cv2

st.set_page_config(page_title="AI Leather Defect Detection")
st.title("🧠 AI Leather Defect Detection Tool")

# --- Load pretrained CNN (ResNet18) ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()  # evaluation mode

# --- Transform for CNN input ---
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])

# --- Function to generate Grad-CAM heatmap ---
def get_heatmap(image, model):
    """Return heatmap highlighting regions CNN finds most important."""
    img_tensor = transform(image).unsqueeze(0)
    img_tensor.requires_grad_()
    
    # Forward pass
    output = model(img_tensor)
    pred_class = output.argmax(dim=1)
    
    # Backward pass to get gradients
    model.zero_grad()
    output[0, pred_class].backward()
    
    # Extract gradients from last conv layer
    gradients = model.layer4[1].conv2.weight.grad
    pooled_gradients = torch.mean(gradients, dim=[0,2,3])
    
    # Extract activations
    activations = model.layer4[1].conv2.weight.detach()
    
    # Compute weighted sum
    for i in range(activations.shape[0]):
        activations[i] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=0).cpu().numpy()
    
    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    
    return heatmap_uint8

# --- Function to detect defects ---
def detect_defects_cnn(image):
    img_np = np.array(image).copy()
    
    # Generate heatmap from CNN
    heatmap = get_heatmap(image, model)
    
    # Threshold heatmap to get defect regions
    thresh = cv2.threshold(heatmap, 128, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours for bounding boxes
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))
        # Draw rectangle on annotated image
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return img_np, boxes, heatmap

# --- Streamlit UI ---
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
    annotated, defects, heatmap = detect_defects_cnn(image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(annotated, caption="Detected Defects")
    with col2:
        st.image(heatmap, caption="Defect Heatmap", clamp=True)

    st.write(f"### 🧪 {len(defects)} defects found")
    for i, (x, y, w, h) in enumerate(defects, 1):
        st.write(f"Defect {i} → Location ({x},{y}) Size {w}x{h}")
