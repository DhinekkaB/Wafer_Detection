import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------- CONFIG --------
MODEL_PATH = "mobilenetv3_wafer_best.pth"
IMAGE_DIR = "C:/Users/DHAKSHNAMOORTHY/Downloads/wafer_detection/test"   # folder path
IMG_SIZE = 160
CLASS_NAMES = [
    "bridge", "clean", "crack", "ler",
    "missing via", "open", "other",
    "particle", "scratch"
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Load model
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Predict images
for img_name in os.listdir(IMAGE_DIR):
    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(IMAGE_DIR, img_name)

        img = Image.open(img_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        print(
            f"{img_name} → {CLASS_NAMES[pred_idx]} "
            f"({round(probs[0][pred_idx].item()*100, 2)}%)"
        )
