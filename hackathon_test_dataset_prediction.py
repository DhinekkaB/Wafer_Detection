import os
import warnings
import numpy as np
import onnxruntime as ort
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from datetime import datetime

# ==========================
# SUPPRESS WARNINGS
# ==========================
warnings.filterwarnings("ignore")

# ==========================
# CONFIG
# ==========================
MODEL_PATH = "mobilenetv3_wafer.onnx"
TEST_DIR   = "hackathon_test_dataset"
INPUT_SIZE = 160

LOG_FILE = "inference_log.txt"
CM_IMAGE = "confusion_matrix.png"

# ---- TRAINING CLASSES (DO NOT CHANGE ORDER) ----
TRAIN_CLASSES = [
    "bridge",
    "clean",
    "crack",
    "ler",
    "missing via",
    "open",
    "other",
    "particle",
    "scratch"
]

# ---- HACKATHON TEST → TRAIN CLASS MAPPING ----
CLASS_MAPPING = {
    "Bridge": "bridge",
    "Clean": "clean",
    "Crack": "crack",
    "LER": "ler",
    "Open": "open",
    "Other": "other",
    "Particle": "particle",
    "VIA": "missing via",
    "CMP": "other"
}

print("\n========== ONNX INFERENCE ==========\n")

# ==========================
# LOAD MODEL
# ==========================
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# ==========================
# PREPROCESS
# ==========================
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32)
    img *= (1.0 / 255.0)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# ==========================
# INFERENCE
# ==========================
all_preds = []
all_labels = []

for folder_name in os.listdir(TEST_DIR):

    folder_path = os.path.join(TEST_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue

    if folder_name not in CLASS_MAPPING:
        continue

    mapped_class = CLASS_MAPPING[folder_name]
    true_label_index = TRAIN_CLASSES.index(mapped_class)

    for img_name in os.listdir(folder_path):

        img_path = os.path.join(folder_path, img_name)
        input_tensor = preprocess(img_path)

        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0]

        pred_index = int(np.argmax(logits, axis=1)[0])

        all_preds.append(pred_index)
        all_labels.append(true_label_index)

# ==========================
# METRICS
# ==========================
accuracy  = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
recall    = recall_score(all_labels, all_preds, average="macro", zero_division=0)

print("📊 FINAL EVALUATION")
print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")

# ==========================
# CONFUSION MATRIX
# ==========================
cm = confusion_matrix(all_labels, all_preds, labels=range(len(TRAIN_CLASSES)))

plt.figure(figsize=(10,8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix (ONNX)")
plt.colorbar()

tick_marks = np.arange(len(TRAIN_CLASSES))
plt.xticks(tick_marks, TRAIN_CLASSES, rotation=45)
plt.yticks(tick_marks, TRAIN_CLASSES)

for i in range(len(TRAIN_CLASSES)):
    for j in range(len(TRAIN_CLASSES)):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black")

plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()

# Save confusion matrix image
plt.savefig(CM_IMAGE, dpi=300)
plt.close()

# ==========================
# SAVE LOG FILE
# ==========================
with open(LOG_FILE, "w") as f:
    f.write("========== ONNX INFERENCE LOG ==========\n")
    f.write(f"Date: {datetime.now()}\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test Dataset: {TEST_DIR}\n\n")

    f.write("FINAL METRICS\n")
    f.write(f"Accuracy  : {accuracy*100:.2f}%\n")
    f.write(f"Precision : {precision*100:.2f}%\n")
    f.write(f"Recall    : {recall*100:.2f}%\n\n")

    f.write("Confusion Matrix:\n")
    f.write(str(cm))

print(f"\n✅ Confusion matrix saved as: {CM_IMAGE}")
print(f"✅ Log file saved as: {LOG_FILE}")
