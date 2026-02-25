
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ======================
# CONFIG
# ======================
DATASET_PATH = "balanced_dataset"
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 80
VAL_SPLIT = 0.2
SEED = 42

# Learning rates: Phase 1 (frozen backbone), Phase 2 (fine-tune)
LR_PHASE1 = 5e-4
LR_PHASE2 = 1e-4

WEIGHT_DECAY = 1e-3
EARLY_STOP_PATIENCE = 15
FREEZE_EPOCHS = 12  # Unfreeze after this

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}\n")

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# ======================
# TRANSFORMS
# ======================
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),  # B&W to 3-ch for MobileNet
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================
# LOAD DATASET
# ======================
print("="*70)
print("LOADING DATASET")
print("="*70)

full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)

print(f"Classes ({num_classes}): {', '.join(class_names)}")
print(f"Total images: {len(full_dataset)}")

# Stratified 80:20 split
targets = [s[1] for s in full_dataset.samples]
indices = np.arange(len(full_dataset))
train_idx, val_idx = train_test_split(
    indices,
    test_size=VAL_SPLIT,
    stratify=targets,
    random_state=SEED
)

print(f"Train: {len(train_idx)} | Val: {len(val_idx)}\n")

# Create datasets
train_dataset = Subset(full_dataset, train_idx)
val_dataset_obj = datasets.ImageFolder(DATASET_PATH, transform=val_transform)
val_dataset = Subset(val_dataset_obj, val_idx)

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Class weights for imbalance
train_targets = np.array([targets[i] for i in train_idx])
class_counts = np.bincount(train_targets, minlength=num_classes)
class_weights = 1.0 / (class_counts + 1e-8)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

print(f"Class weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]\n")

# ======================
# MODEL
# ======================
print("="*70)
print("LOADING MOBILENETV3 SMALL")
print("="*70)

model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
model = model.to(DEVICE)

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable params (Phase 1): {trainable_params:,}\n")

# ======================
# LOSS + OPTIMIZER + SCHEDULER
# ======================
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_PHASE1,
    weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
)

# ======================
# TRAINING
# ======================
print("="*70)
print("TRAINING STARTED")
print("="*70 + "\n")

best_val_acc = 0
early_stop_count = 0
best_epoch = 0

for epoch in range(EPOCHS):
    
    # Unfreeze backbone and fine-tune
    if epoch == FREEZE_EPOCHS:
        print(f"\n*** PHASE 2: Unfreezing backbone at epoch {epoch+1} ***\n")
        for param in model.features.parameters():
            param.requires_grad = True
        
        optimizer = optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7
        )
    
    # TRAIN
    model.train()
    train_loss, train_correct = 0.0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
    
    train_acc = train_correct / len(train_dataset)
    train_loss /= len(train_dataset)
    
    # VAL
    model.eval()
    val_loss, val_correct = 0.0, 0
    val_preds_all, val_labels_all = [], []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            
            val_preds_all.extend(preds.cpu().numpy())
            val_labels_all.extend(labels.cpu().numpy())
    
    val_acc = val_correct / len(val_dataset)
    val_loss /= len(val_dataset)
    gap = train_acc - val_acc
    
    scheduler.step(val_acc)
    
    print(f"Epoch {epoch+1:2d}/{EPOCHS} | Train: {train_acc*100:5.2f}% | Val: {val_acc*100:5.2f}% | Gap: {gap*100:5.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        best_preds = val_preds_all
        best_labels = val_labels_all
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  ✓ Saved (Best: {best_val_acc*100:.2f}%)")
        early_stop_count = 0
    else:
        early_stop_count += 1
    
    if early_stop_count >= EARLY_STOP_PATIENCE:
        print(f"\nEarly stop at epoch {epoch+1}\n")
        break

# ======================
# RESULTS
# ======================
print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
print(f"Best Validation: {best_val_acc*100:.2f}% (Epoch {best_epoch})\n")

model.load_state_dict(torch.load("best_model.pth"))

print("="*70)
print("CLASSIFICATION REPORT (VALIDATION SET)")
print("="*70)
print(classification_report(best_labels, best_preds, target_names=class_names))

pth_size = os.path.getsize("best_model.pth") / (1024*1024)

metadata = {
    "model_architecture": "MobileNetV3 Small",
    "validation_accuracy_percent": round(best_val_acc * 100, 2),
    "best_epoch": best_epoch,
    "num_classes": num_classes,
    "class_names": class_names,
    "image_size": IMAGE_SIZE,
    "input_format": "Grayscale (128x128) converted to 3-channel RGB",
    "dataset": {
        "total_images": len(full_dataset),
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
        "split": "80:20 stratified"
    },
    "training": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "loss": "CrossEntropyLoss with label smoothing (0.1)",
        "phase1_epochs": FREEZE_EPOCHS,
        "phase1_lr": LR_PHASE1,
        "phase2_lr": LR_PHASE2,
        "weight_decay": WEIGHT_DECAY,
        "early_stopping_patience": EARLY_STOP_PATIENCE
    },
    "augmentation": "Flip (0.5), Rotation (20°), Affine (translate+scale), ColorJitter",
    "pytorch_model_size_mb": round(pth_size, 2),
    "next_steps": "Convert to TFLite (no quantization) → Quantize for deployment → Convert to C array"
}

with open("model_metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n" + "="*70)
print("✅ PIPELINE COMPLETE")
print("="*70)
print(f"📊 Accuracy: {best_val_acc*100:.2f}%")
print(f"📦 Model: best_model.pth ({pth_size:.2f} MB)")
print(f"📝 Metadata: model_metadata.json")
print(f"⏭️  Ready for: TFLite conversion → Quantization → C array conversion")
print("="*70)
