import os
import streamlit as st
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn as nn

st.title(" Deepfake Detector")

# -------- Model --------
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "deepfake_detector.pt")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: real/fake

class_to_idx = {"real": 0, "fake": 1}
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    class_to_idx = checkpoint.get("class_to_idx", class_to_idx)
else:
    st.warning(
        "No trained weights found (deepfake_detector.pt). Predictions are from an "
        "untrained head and are not meaningful. Run train_deepfake_model.py first."
    )

model.eval()
idx_to_class = {v: k for k, v in class_to_idx.items()}

# -------- Transform --------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -------- Upload Image --------
uploaded = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
    
    predicted_class = idx_to_class.get(pred, "unknown")
    label = " Real" if predicted_class == "real" else " Deepfake"
    st.subheader(f"Prediction: {label}")
