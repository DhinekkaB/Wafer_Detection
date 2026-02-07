import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os

# ---------- CONFIG ----------
DATASET_PATH = "dataset_final"
MODEL_PATH = "mobilenetv3_wafer_best.pth"
IMG_SIZE = 160
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------

# Transforms (NO augmentation)
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Dataset & Loader
test_dataset = datasets.ImageFolder(
    os.path.join(DATASET_PATH, "test"),
    transform=test_transform
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes
num_classes = len(class_names)

# Load model
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Inference
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Metrics
print("\n📊 Classification Report (Test Set):\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("\n🧩 Confusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))
