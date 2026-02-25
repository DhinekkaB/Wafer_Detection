<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Phase%203%20Results&fontSize=60&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Semiconductor%20Defect%20Classification%20%7C%20MobileNetV3%20Small&descAlignY=60&descSize=18" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Accuracy-82.69%25-brightgreen?style=for-the-badge&logo=checkmarx&logoColor=white&labelColor=1a1a2e"/>
  <img src="https://img.shields.io/badge/Model-MobileNetV3_Small-blue?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=16213e"/>
  <img src="https://img.shields.io/badge/Classes-11-orange?style=for-the-badge&logo=buffer&logoColor=white&labelColor=0f3460"/>
  <img src="https://img.shields.io/badge/Model_Size-5.96_MB-purple?style=for-the-badge&logo=databricks&logoColor=white&labelColor=533483"/>
</p>
<p>
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/TFLite-Ready-00B16A?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Best_Epoch-54-F7DC6F?style=for-the-badge&logo=lightning&logoColor=black"/>
  <img src="https://img.shields.io/badge/Input-128x128_Grayscale-E74C3C?style=for-the-badge&logo=opencv&logoColor=white"/>
</p>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=00D9FF&center=true&vCenter=true&multiline=false&repeat=true&width=750&height=45&lines=82.69%25+Validation+Accuracy+%F0%9F%8E%AF;11-Class+Wafer+Defect+Classification+%F0%9F%94%AC;MobileNetV3+Small+%7C+TFLite+Deployment+Ready+%F0%9F%9A%80" alt="Typing SVG"/>
</a>

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🏗️ Architecture](#%EF%B8%8F-architecture)
- [⚙️ Training Configuration](#%EF%B8%8F-training-configuration)
- [📊 Results](#-results)
- [📈 Performance Charts](#-performance-charts)
- [🗂️ Classification Report](#%EF%B8%8F-classification-report)
- [🚀 Deployment Pipeline](#-deployment-pipeline)
- [📁 File Structure](#-file-structure)
- [💻 Quick Inference](#-quick-inference)

---

## 🎯 Overview

<div align="center">

> **Phase 3** trains a lightweight, deployment-ready deep learning model for **semiconductor wafer defect classification** using two-phase transfer learning on MobileNetV3 Small. Goal: maximum accuracy in a minimal footprint for edge deployment.

</div>

```mermaid
flowchart LR
    A[Balanced Dataset - 11 Classes] --> B[Augmentation Pipeline]
    B --> C[Phase 1 - Frozen Backbone]
    C --> D[Phase 2 - Full Fine-tune]
    D --> E[best_model.pth - 82.69%]
    E --> F[model.tflite - Edge Ready]
```

---

## 🏗️ Architecture

```mermaid
flowchart TD
    A[Input - 128x128 Grayscale Image]
    B[Grayscale to 3-Channel Replication]
    C[MobileNetV3 Small Backbone - ImageNet Pretrained]
    D[Phase 1 - Backbone Frozen - Classifier Only - LR 5e-4]
    E[Phase 2 - All Layers Train - LR 1e-4]
    F[Classifier Head - Linear - HardSwish - Dropout - Linear 11]
    G[Softmax Output - 11 Defect Classes]

    A --> B --> C --> D --> E --> F --> G

    style A fill:#1a1a2e,color:#fff,stroke:#00D9FF
    style D fill:#1a3a2e,color:#F7DC6F,stroke:#F7DC6F
    style E fill:#3a1a1a,color:#E74C3C,stroke:#E74C3C
    style G fill:#0f3460,color:#fff,stroke:#00D9FF
```

**Model Size:** `5.96 MB (.pth)` &nbsp;|&nbsp; **TFLite:** `model.tflite` ✅ &nbsp;|&nbsp; **Total Params:** ~1.52M

---

## ⚙️ Training Configuration

<div align="center">

| Parameter | Value |
|-----------|-------|
| 🖼️ Image Size | `128 × 128` |
| 📦 Batch Size | `32` |
| 🔄 Max Epochs | `80` |
| 🏆 Best Epoch | **54** |
| ✂️ Val Split | `20% stratified` |
| 🧊 Phase 1 LR | `5e-4` — frozen backbone, epochs 1–12 |
| 🔥 Phase 2 LR | `1e-4` — full fine-tune, epochs 13–54 |
| ⚖️ Weight Decay | `1e-3` |
| ⏹️ Early Stop | `patience = 15` |
| 📉 Loss | `CrossEntropyLoss + Label Smoothing (0.1)` |
| 🔧 Optimizer | `AdamW` |
| 📅 Scheduler | `ReduceLROnPlateau (factor=0.5, patience=3)` |
| 🎨 Augmentation | `Flip (p=0.5) · Rotation ±20° · Affine · ColorJitter` |

</div>

### Two-Phase Training Strategy

```mermaid
gantt
    title Training Timeline — Best Epoch 54 of 80
    dateFormat X
    axisFormat Ep %s

    section Phase 1 Frozen Backbone
    Classifier only  LR 5e-4    :done, 0, 12

    section Phase 2 Full Fine-tune
    All layers  LR 1e-4         :active, 12, 54

    section Early Stopping
    Patience window triggered   :crit, 54, 69
```

---

## 📊 Results

<div align="center">

### 🏆 Final Metrics

| Metric | Score |
|--------|-------|
| ✅ **Validation Accuracy** | **82.69%** |
| 📊 Macro Avg Precision | **0.84** |
| 📊 Macro Avg Recall | **0.83** |
| 📊 Macro Avg F1-Score | **0.83** |
| 🗓️ Best Epoch | **54 / 80** |
| 💾 Model Size | **5.96 MB** |
| 🧪 Val Samples | **439** (20% stratified) |

</div>

---

## 📈 Performance Charts

### Per-Class F1 Score

```mermaid
xychart-beta horizontal
    title "Per-Class F1 Score"
    x-axis ["CLEAN_CRACK", "VIA", "CLEAN_VIA", "CRACK", "LER", "OPEN", "CMP", "BRIDGE", "OTHERS", "PARTICLE", "CLEAN_LAYER"]
    y-axis "F1 Score" 0 --> 1
    bar [0.94, 0.89, 0.89, 0.88, 0.81, 0.80, 0.80, 0.79, 0.79, 0.76, 0.76]
```

### Precision vs Recall Per Class

```mermaid
xychart-beta horizontal
    title "Precision vs Recall per Class"
    x-axis ["CC", "VIA", "CVA", "CRK", "LER", "OPEN", "CMP", "BRG", "OTH", "PAR", "CL"]
    y-axis "Score" 0 --> 1
    bar  [0.91, 0.94, 0.84, 0.97, 0.91, 0.86, 0.76, 0.75, 0.82, 0.77, 0.68]
    line [0.97, 0.85, 0.95, 0.80, 0.72, 0.75, 0.85, 0.82, 0.78, 0.75, 0.85]
```

> **CC** = CLEAN\_CRACK &nbsp;·&nbsp; **CVA** = CLEAN\_VIA &nbsp;·&nbsp; **CRK** = CRACK &nbsp;·&nbsp; **BRG** = BRIDGE &nbsp;·&nbsp; **OTH** = OTHERS &nbsp;·&nbsp; **PAR** = PARTICLE &nbsp;·&nbsp; **CL** = CLEAN\_LAYER

---

## 🗂️ Classification Report

<div align="center">

| Class | Precision | Recall | F1-Score | Support | Status |
|-------|:---------:|:------:|:--------:|:-------:|:------:|
| 🟢 **CLEAN_CRACK** | 0.91 | 0.97 | **0.94** | 40 | ✅ Excellent |
| 🟢 **VIA** | 0.94 | 0.85 | **0.89** | 40 | ✅ Excellent |
| 🟢 **CLEAN_VIA** | 0.84 | 0.95 | **0.89** | 39 | ✅ Excellent |
| 🟢 **CRACK** | 0.97 | 0.80 | **0.88** | 40 | ✅ Excellent |
| 🟡 **LER** | 0.91 | 0.72 | **0.81** | 40 | ⚡ Good |
| 🟡 **OPEN** | 0.86 | 0.75 | **0.80** | 40 | ⚡ Good |
| 🟡 **CMP** | 0.76 | 0.85 | **0.80** | 40 | ⚡ Good |
| 🟡 **BRIDGE** | 0.75 | 0.82 | **0.79** | 40 | ⚡ Good |
| 🟡 **OTHERS** | 0.82 | 0.78 | **0.79** | 40 | ⚡ Good |
| 🟠 **PARTICLE** | 0.77 | 0.75 | **0.76** | 40 | 🔧 Fair |
| 🟠 **CLEAN_LAYER** | 0.68 | 0.85 | **0.76** | 40 | 🔧 Fair |
| | | | | | |
| **Accuracy** | | | **0.83** | **439** | |
| **Macro Avg** | **0.84** | **0.83** | **0.83** | 439 | |
| **Weighted Avg** | **0.84** | **0.83** | **0.83** | 439 | |

</div>

> 🔑 **Strongest:** `CRACK (P=0.97)`, `VIA (P=0.94)`, `LER (P=0.91)` — critical defects caught with high precision  
> ⚠️ **Weakest:** `CLEAN_LAYER (P=0.68)` — visually similar to other clean surfaces; candidate for targeted augmentation in Phase 4

---

## 🚀 Deployment Pipeline

```mermaid
flowchart TD
    A[Balanced Dataset - 11 Classes - 80/20 Split]
    B[Augmentation - Flip Rotate Affine ColorJitter]
    C[Phase 1 - Frozen Backbone - 12 Epochs - LR 5e-4]
    D[Phase 2 - Full Fine-tune - Epoch 13 to 54 - LR 1e-4]
    E{Validation - 82.69% - Best at Epoch 54}
    F[best_model.pth - 5.96 MB]
    G[model_metadata.json]
    H[model.tflite - TFLite Export]
    I[INT8 Quantization]
    J[C Array - MCU Edge Deployment]

    A --> B --> C --> D --> E
    E --> F
    E --> G
    F --> H --> I --> J

    style A fill:#1a1a2e,color:#fff,stroke:#00D9FF
    style E fill:#0f3460,color:#fff,stroke:#2ECC71
    style H fill:#16213e,color:#fff,stroke:#00B16A
    style J fill:#533483,color:#fff,stroke:#9B59B6
```

### Inference Sequence

```mermaid
sequenceDiagram
    participant IMG as Input Image
    participant PRE as Preprocessor
    participant MDL as MobileNetV3 Small
    participant OUT as Output

    IMG->>PRE: Raw grayscale image
    PRE->>PRE: Resize to 128x128
    PRE->>PRE: Grayscale to 3-channel
    PRE->>PRE: Normalize with ImageNet stats
    PRE->>MDL: Tensor 1 x 3 x 128 x 128
    MDL->>MDL: Backbone feature extraction
    MDL->>MDL: Classifier head forward pass
    MDL->>OUT: Logits 1 x 11
    OUT->>OUT: Softmax to probabilities
    OUT-->>IMG: Predicted class and confidence
```

---
## Deployment result
On porting we got
<p align="center">
  <img src="images/output.png" width="500"/>
</p>
---


## 📁 File Structure

```
phase3_results/
├── 📂 logs/
│   └── 📄 train_log.txt              ← Full epoch-by-epoch training log
├── 📂 code/ 
│    └── 📄 TRAIN_FINAL.py            ← Full training pipeline   
│   └── 📄 TEST_PREDICTIONS.py        ← Inference and prediction script    
│ 
├── 📦 best_model.pth                 ← PyTorch checkpoint (5.96 MB)
├── 📱 model.tflite                   ← TFLite deployment model ✅
├── 📋 model_metadata.json            ← Architecture config and metrics
└── 📖 README.md                      ← This file
```

---

## 💻 Quick Inference

```python
from torchvision import transforms, models
import torch, torch.nn as nn
from PIL import Image

CLASS_NAMES = [
    'BRIDGE', 'CLEAN_CRACK', 'CLEAN_LAYER', 'CLEAN_VIA',
    'CMP', 'CRACK', 'LER', 'OPEN', 'OTHERS', 'PARTICLE', 'VIA'
]

# Load model
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 11)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# Preprocessing — must match training exactly
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict
img = transform(Image.open("your_image.png").convert("RGB")).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(img), dim=1)
    pred  = torch.argmax(probs).item()

print(f"Prediction : {CLASS_NAMES[pred]}")
print(f"Confidence : {probs[0, pred]:.2%}")
```
---
<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=twinkling" width="100%"/>

**Phase 3 Complete** &nbsp;·&nbsp; MobileNetV3 Small &nbsp;·&nbsp; **82.69% Accuracy** &nbsp;·&nbsp; TFLite ✅

![visitors](https://visitor-badge.laobi.icu/badge?page_id=phase3.wafer.defect&left_color=1a1a2e&right_color=0099ff)

</div>

---
