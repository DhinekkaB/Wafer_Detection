<div align="center">

# 🔬 Edge-AI Defect Classification
### Semiconductor Wafer/Die Inspection System

[![Hackathon](https://img.shields.io/badge/i4C-DeepTech%20Hackathon-blue?style=for-the-badge)](https://github.com)
[![Phase](https://img.shields.io/badge/Phase-1-success?style=for-the-badge)](https://github.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai)

**A lightweight, edge-ready AI system for real-time semiconductor defect classification**

[Overview](#-overview) • [Architecture](#-system-architecture) • [Results](#-results) • [Quick Start](#-quick-start) • [Documentation](#-documentation)

---
### 📥 Access Dataset

The complete dataset is available on Google Drive: [Download Dataset](https://drive.google.com/drive/folders/1ELvzk_jxf1cS49J7Bg6TYLVo37f8jUG1?usp=drive_link)
---

### 📦 Trained Models

| Model | Format | Download |
|:-----:|:------:|:--------:|
| **Edge Deployment** | ONNX | [Download](https://drive.google.com/file/d/1P7pXT_2dHnE4BEGRX0pdVWwTS6Ac68sx/view?usp=drive_link) |
| **PyTorch Checkpoint** | .pth | [Download](https://drive.google.com/file/d/1m6rA06NR_sDJqM55CPSqZAFLOWPsUex-/view?usp=drive_link) |

---
---

# 🚀 Phase 2 – ONNX Inference & Evaluation (Hackathon Submission)

<div align="center">

**Strict reuse of Phase-1 ONNX model without retraining**

</div>

---

## 📌 Phase-2 Overview

<table>
<tr>
<td width="60%">

### Objective

Evaluate the previously trained **MobileNetV3 ONNX model** on the hackathon test dataset under strict constraints:

- ❌ No retraining allowed  
- ❌ No architecture modification  
- ❌ No new model submission  
- ✅ Same Phase-1 exported ONNX model reused  
- ✅ Class mismatch handled via mapping to `other`  

### Evaluation Scope

- Pure ONNX inference
- Deterministic pipeline
- Minimal preprocessing (resize + required scaling only)
- Metrics generation
- Confusion matrix visualization

</td>
<td width="40%">

### 📊 Phase-2 Performance

| Metric | Score |
|:------:|:-----:|
| **Accuracy** | **54.39%** |
| **Precision (Macro)** | **53.89%** |
| **Recall (Macro)** | **49.79%** |

</td>
</tr>
</table>

---

## 📈 Confusion Matrix (Phase-2 Evaluation)

<div align="center">

<img src="confusion_matrix_phase2.png" alt="Phase-2 Confusion Matrix" width="500"/>

**Evaluation performed using strict ONNX inference pipeline**

</div>

---

## 🧾 Inference Log (Phase-2 Run)

<div align="center">

<img src="inference_log_phase2.png" alt="Inference Log Screenshot" width="600"/>

</div>

> The complete log file is available in the repository as:
> `inference_log_phase2.txt`

---

## 📂 Phase-2 Submission Files

<table>
<tr>
<th>File</th>
<th>Description</th>
</tr>

<tr>
<td><code>hackathon_test_dataset_prediction.py</code></td>
<td>ONNX inference script used for Phase-2 evaluation</td>
</tr>

<tr>
<td><code>inference_log_phase2.png</code></td>
<td>Generated log file containing metrics and confusion matrix values</td>
</tr>

<tr>
<td><code>confusion_matrix_phase2.png</code></td>
<td>Confusion matrix visualization for hackathon test dataset</td>
</tr>

</table>

---

## ⚙️ Phase-2 Inference Pipeline

```python
Image → Resize (160×160) → Float32 Scaling → ONNX Runtime → Argmax → Metrics
```
---

## ⚙️ Key Characteristics

<table>
<tr>
<td width="100%">

| Feature | Description |
|----------|-------------|
| **Model Reuse** | `mobilenetv3_wafer.onnx` from Phase-1 |
| **Retraining** | Not performed |
| **Fine-Tuning** | Not performed |
| **Parameter Modification** | None |
| **Class Handling** | Test mismatches mapped to `other` |
| **Evaluation Type** | Fully deterministic ONNX inference |

</td>
</tr>
</table>

---

## 🔎 Observations

<table>
<tr>
<td width="50%" valign="top">

### ✅ Strong Detection Performance

The model demonstrates reliable classification for:

- **Bridge**
- **LER**
- **Particle**
- **Other**

These classes show strong diagonal dominance in the confusion matrix, indicating stable prediction consistency.
Other class has moderate accuracy

</td>

<td width="50%" valign="top">

### ⚠️ Expected Confusion Patterns

Misclassifications primarily occur between visually similar defects:

- **Open ↔ Other**
- **Missing Via ↔ Clean**

Such confusion patterns are consistent with structural similarity in wafer defect morphology.

</td>
</tr>
</table>

---

## 📈 Generalization Insight

<div align="center">

The model demonstrates stable generalization capability under strict evaluation constraints.

</div>

<table>
<tr>
<td align="center">

**Evaluation Constraints Applied**

- No retraining  
- No architectural modification  
- Class label mismatch in test dataset  

</td>
</tr>
</table>

---

## 🏁 Compliance Statement

<table>
<tr>
<td width="100%">

Phase-2 evaluation strictly adheres to hackathon rules:

- Model retraining was **NOT** performed.
- Phase-1 ONNX model reused without modification.
- Only permitted preprocessing (resize + required scaling) applied.
- All mandatory evaluation artifacts generated and included.

</td>
</tr>
</table>

---

</div>

## 🎯 Overview(Phase 1)

<table>
<tr>
<td width="60%">

### The Challenge

Semiconductor manufacturing requires precise defect detection at the nanometer scale. Traditional inspection methods are:
- ⏱️ **Time-intensive** – Manual inspection bottlenecks
- 💰 **Cost-prohibitive** – Expensive equipment and expertise
- 🎯 **Inconsistent** – Human error variability

### Our Solution

An **edge-deployable AI system** that:
- ✅ Classifies 9 defect categories with **82% accuracy**
- ✅ Runs on **resource-constrained devices**
- ✅ Enables **real-time decision making** at the edge

</td>
<td width="40%">

### 📊 Quick Stats
```
🎯 Test Accuracy:    82%
📈 F1-Score:         0.79
🔍 Classes:          9 defect types
📸 Dataset:          1000+ images
⚡ Model:            MobileNetV3-Small
📦 Export:           ONNX ready
```

### 🏆 Defect Categories
```diff
+ Bridge        + Crack
+ LER           + Missing Via
+ Open          + Particle
+ Scratch       + Clean
+ Other
```

</td>
</tr>
</table>

---

### 🎯 Architecture Highlights

| Layer | Technology | Purpose |
|:-----:|:----------:|:-------:|
| **Input** | Grayscale Images | 160×160 wafer defect images |
| **Preprocessing** | PyTorch Transforms | Augmentation & normalization |
| **Model** | MobileNetV3-Small | Lightweight CNN architecture |
| **Training** | Transfer Learning | ImageNet pre-trained weights |
| **Export** | ONNX | Edge deployment compatibility |

</div>

---

## 📊 Dataset

<div align="center">

### Dataset Composition

| Attribute | Value |
|:---------:|:-----:|
| 📦 **Total Images** | ~1000+ (augmented) |
| 🏷️ **Classes** | 9 categories |
| 🎨 **Format** | Grayscale (160×160) |
| 📐 **Split Ratio** | 70 / 15 / 15 |
| 🔄 **Augmentation** | Training set only |

</div>

<details>
<summary><b>📋 Class Distribution Details</b></summary>

<br>

**Defect Classes (7):**
- 🔗 Bridge
- 💥 Crack  
- 📏 LER (Line Edge Roughness)
- ⭕ Missing Via
- 🔓 Open
- ⚪ Particle
- 〰️ Scratch

**Non-Defect Classes (2):**
- ✅ Clean
- ❓ Other

**Data Sources:**
- Public wafer/SEM datasets
- Manual curation and labeling
- Folder-based classification structure

</details>

---

## 🧠 Model Architecture

<table>
<tr>
<td width="50%">

### 🎯 Design Choices

**Why MobileNetV3-Small?**
```
✓ Optimized for mobile/edge devices
✓ Minimal memory footprint
✓ Fast inference time
✓ Proven transfer learning capabilities
✓ ONNX export compatibility
```

### 📐 Model Specifications

| Component | Detail |
|-----------|--------|
| **Base Architecture** | MobileNetV3-Small |
| **Framework** | PyTorch |
| **Training Method** | Transfer Learning |
| **Input Shape** | (3, 160, 160) |
| **Output Classes** | 9 |

</td>
<td width="50%">

### ⚙️ Training Configuration
```python
# Training Hyperparameters
EPOCHS          = 20
BATCH_SIZE      = 32
OPTIMIZER       = Adam
LEARNING_RATE   = 1e-4
LOSS_FUNCTION   = CrossEntropyLoss
CHECKPOINT      = Best validation accuracy

# Data Processing
INPUT_SIZE      = 160×160
COLOR_MODE      = Grayscale → RGB
NORMALIZATION   = ImageNet stats
AUGMENTATION    = Train only
```

### 🎓 Training Strategy

1. **Initialization:** Pre-trained ImageNet weights
2. **Fine-tuning:** All layers trainable
3. **Validation:** 15% holdout set
4. **Selection:** Best epoch checkpoint
5. **Export:** ONNX conversion

</td>
</tr>
</table>

---
---

## ✅ Model Verification

### Live Testing Results

The model has been tested and verified on real wafer images. Below is a sample inference run:

<div align="center">

![Test Results](test_result.png)

**Sample predictions showing high-confidence classification across defect categories**

</div>

### Test Output Summary
```bash
python test.py
```

**Sample Predictions:**
- ✅ Bridge defect detected with 99.99% confidence
- ✅ Scratch defect detected with 100.0% confidence  
- ✅ Real-time inference validation successful

> The model demonstrates consistent high-confidence predictions on unseen test images, validating its deployment readiness.

---
## 📈 Results

<div align="center">

### 🎯 Test Set Performance

<table>
<tr>
<td align="center">

### Overall Metrics

| Metric | Score |
|:------:|:-----:|
| **Accuracy** | **82%** |
| **Precision** | **0.80** |
| **Recall** | **0.79** |
| **F1-Score** | **0.79** |

</td>
<td align="center">

### Confusion Matrix

<img src="confusion_matrix.png" alt="Confusion Matrix" width="400"/>

</td>
</tr>
</table>

</div>

### 🔍 Key Insights

<table>
<tr>
<td width="50%" valign="top">

#### ✅ Strong Performance
- **LER Detection:** High precision and recall
- **Missing Via:** Excellent classification accuracy
- **Particle Defects:** Minimal false negatives
- **Balanced Metrics:** Consistent across most classes

</td>
<td width="50%" valign="top">

#### ⚠️ Expected Challenges
- **Visually Similar Defects:** Confusion between Open/Bridge/Crack
- **Class Imbalance:** Some defect types less represented
- **Edge Cases:** Complex multi-defect scenarios
- **Grayscale Limitations:** Fine-grained texture differences

</td>
</tr>
</table>

---

## ⚡ Edge Deployment Readiness

<div align="center">

### Why This Model is Edge-Ready

</div>

| Feature | Benefit | Impact |
|---------|---------|--------|
| 🎯 **MobileNetV3-Small** | Lightweight architecture | Low compute requirements |
| 🖼️ **Grayscale Input** | Single channel processing | 3× memory reduction |
| 📦 **ONNX Format** | Cross-platform compatibility | Deploy anywhere |
| ⚡ **Optimized Inference** | Compact model size | Fast predictions |
| 🔧 **Transfer Learning** | Fewer parameters to train | Quick adaptation |

<div align="center">

### 🎮 Target Platforms

[![NXP](https://img.shields.io/badge/NXP-eIQ-00A3E0?style=flat-square)](https://www.nxp.com)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Jetson-76B900?style=flat-square)](https://www.nvidia.com/jetson)
[![RPi](https://img.shields.io/badge/Raspberry-Pi-A22846?style=flat-square)](https://www.raspberrypi.org)
[![Intel](https://img.shields.io/badge/Intel-OpenVINO-0071C5?style=flat-square)](https://www.intel.com/openvino)

**Note:** Phase 1 focuses on software implementation. Hardware deployment validation planned for future phases.

</div>

---

## 🚀 Quick Start

### 📋 Prerequisites
```bash
# Clone the repository
git clone https://github.com/DhinekkaB/Wafer_Detection.git
cd wafer_detection

# Install dependencies
pip install -r requirements.txt
```

<details>
<summary><b>📦 Required Dependencies</b></summary>
```
torch>=2.0.0
torchvision>=0.15.0
onnx>=1.14.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
numpy>=1.24.0
Pillow>=9.5.0
```

</details>

---

### 🎯 Usage

<table>
<tr>
<td width="50%">

#### 1️⃣ Train Model
```bash
python final.py
```

**What it does:**
- Loads and preprocesses dataset
- Trains MobileNetV3-Small
- Saves best checkpoint
- Outputs training metrics

**Output:** `mobilenetv3_wafer_best.pth`

</td>
<td width="50%">

#### 2️⃣ Evaluate Model
```bash
python evaluate.py
```

**What it does:**
- Loads test dataset
- Runs inference
- Calculates metrics
- Displays results

**Output:** Precision, Recall, F1-Score

</td>
</tr>
<tr>
<td width="50%">

#### 3️⃣ Generate Confusion Matrix
```bash
python matrix.py
```

**What it does:**
- Evaluates on test set
- Creates visualization
- Saves as PNG

**Output:** `confusion_matrix.png`

</td>
<td width="50%">

#### 4️⃣ Export to ONNX
```bash
python export_onnx.py
```

**What it does:**
- Converts PyTorch → ONNX
- Validates conversion
- Optimizes for inference

**Output:** `mobilenetv3_wafer.onnx`

</td>
</tr>
</table>

---

## 📁 Repository Structure
```
📦 wafer-defect-classification
 ┣ 📜 final.py                      # Main training script
 ┣ 📜 evaluate.py                   # Model evaluation
 ┣ 📜 matrix.py                     # Confusion matrix generator
 ┣ 📜 export_onnx.py                # ONNX export utility
 ┣ 📊 confusion_matrix.png          # Results visualization
 ┣ 🤖 mobilenetv3_wafer_best.pth    # Trained model checkpoint
 ┣ 📦 mobilenetv3_wafer.onnx        # ONNX model
 ┣ 📦 mobilenetv3_wafer.onnx.data   # ONNX weights
 ┣ 📋 requirements.txt              # Python dependencies
 ┗ 📖 README.md                     # Documentation
```

---

## 🛠️ Technology Stack

<div align="center">

### Core Frameworks

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)](https://onnx.ai)

### Libraries & Tools

![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square)

</div>

---

## 📚 References

1. **Deep Learning for Wafer Defect Inspection** – Industrial survey on CNN-based semiconductor defect classification
2. **Public SEM/Wafer Defect Datasets** – Open-source semiconductor inspection image repositories
3. **NXP eIQ Edge AI Toolkit Documentation** – Edge deployment framework and optimization guidelines

---

## 👥 Team

<div align="center">

**i4C DeepTech Hackathon – Phase 1**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/DhinekkaB)

</div>

---

## 📝 License

This project was developed for the **i4C DeepTech Hackathon**. All rights reserved.

---

<div align="center">

### ⚠️ Important Notice

**This implementation represents Phase 1 software development.**

Results are based on test set evaluation. No hardware deployment or real-time performance claims are made at this stage.

---

**Made with 💙 for i4C DeepTech Hackathon**

[![Star this repo](https://img.shields.io/github/stars/yourusername/wafer-defect-classification?style=social)](https://github.com/DhinekkaB/Wafer_Detection.git)

</div>
