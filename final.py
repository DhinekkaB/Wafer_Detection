import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ================= CONFIG =================
DATASET_PATH = "dataset_final"
IMG_SIZE = 160                 # Edge-friendly, better than 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4                      # Lower LR for stability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "mobilenetv3_wafer_best.pth"
# =========================================

print(f"Using device: {DEVICE}")

# --------- DATA TRANSFORMS ----------
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # MobileNet expects 3 channels
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# --------- DATASETS ----------
train_dataset = datasets.ImageFolder(
    os.path.join(DATASET_PATH, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATASET_PATH, "valid"),
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

# --------- MODEL ----------
model = models.mobilenet_v3_small(weights="DEFAULT")

# Replace classifier head
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    num_classes
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --------- TRAINING LOOP ----------
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # -------- VALIDATION --------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Loss: {avg_loss:.3f} | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # -------- SAVE BEST MODEL --------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_NAME)
        print(f"✅ Best model saved (Val Acc = {best_val_acc:.2f}%)")

print("\n🎉 Training completed")
print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"💾 Model saved as: {MODEL_NAME}")
