import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import gdown
import os

# -------------------------------
# Page config (optional but good)
# -------------------------------
st.set_page_config(page_title="Sugarcane Disease Detection")

# -------------------------------
# Download model from Google Drive
# -------------------------------
MODEL_PATH = "best_vit_cnn.pth"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1pwUoLixTrTrees-VvRwevocXZlMKX6BG"
        gdown.download(url, MODEL_PATH, quiet=False)

   model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()
    return model

model = load_model()

# -------------------------------
# Image transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["Healthy", "Red Rot", "Rust", "Leaf Spot", "Other"]

# -------------------------------
# UI
# -------------------------------
st.title("🌿 Sugarcane Disease Detection")

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded Image")

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    st.success(f"Prediction: {classes[pred]}")
