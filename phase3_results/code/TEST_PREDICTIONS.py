import os
import json
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ======================
# SETUP LOGGING
# ======================
log_filename = f"test_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================
# CONFIG
# ======================
TEST_DATASET_PATH = "dataset2"   # Folder with only images
IMAGE_SIZE = 128
MODEL_PATH = "best_mobilenetv3_small.pth"
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {DEVICE}\n")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ======================
# TRANSFORMS (Same as validation)
# ======================
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================
# LOAD MODEL
# ======================
logger.info("="*70)
logger.info("LOADING MODEL")
logger.info("="*70)

if not os.path.exists(MODEL_PATH):
    logger.error(f"❌ Model not found: {MODEL_PATH}")
    exit(1)

# ⚠️ IMPORTANT:
# Make sure class names match your training dataset order
CLASS_NAMES = [
    'BRIDGE', 'CLEAN_CRACK', 'CLEAN_LAYER', 'CLEAN_VIA',
    'CMP', 'CRACK', 'LER', 'OPEN', 'OTHERS', 'PARTICLE', 'VIA'
]

num_classes = len(CLASS_NAMES)

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    num_classes
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

logger.info("✓ Model loaded successfully\n")

# ======================
# LOAD IMAGES (NO LABELS)
# ======================
logger.info("="*70)
logger.info("LOADING TEST IMAGES (No Labels)")
logger.info("="*70)

if not os.path.exists(TEST_DATASET_PATH):
    logger.error(f"❌ Test dataset path not found: {TEST_DATASET_PATH}")
    exit(1)

image_extensions = (".jpg", ".jpeg", ".png", ".bmp")

image_paths = [
    os.path.join(TEST_DATASET_PATH, f)
    for f in os.listdir(TEST_DATASET_PATH)
    if f.lower().endswith(image_extensions)
]

if len(image_paths) == 0:
    logger.error("❌ No images found in test folder!")
    exit(1)

logger.info(f"Total images found: {len(image_paths)}\n")

# ======================
# RUN PREDICTIONS
# ======================
logger.info("="*70)
logger.info("RUNNING PREDICTIONS")
logger.info("="*70 + "\n")

detailed_results = []

with torch.no_grad():
    for img_path in image_paths:

        try:
            img = Image.open(img_path).convert("RGB")
            img = test_transform(img)
            img = img.unsqueeze(0).to(DEVICE)

            outputs = model(img)
            probabilities = torch.softmax(outputs, dim=1)

            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

            result = {
                "image_name": os.path.basename(img_path),
                "predicted_class": CLASS_NAMES[predicted_class],
                "confidence": round(confidence, 4)
            }

            detailed_results.append(result)

            logger.info(
                f"{os.path.basename(img_path):25s} | "
                f"Predicted: {CLASS_NAMES[predicted_class]:15s} | "
                f"Confidence: {confidence:.4f}"
            )

        except Exception as e:
            logger.warning(f"⚠️ Failed to process {img_path}: {e}")

# ======================
# SAVE RESULTS
# ======================
results_json = {
    "test_summary": {
        "total_images": len(detailed_results),
        "timestamp": datetime.now().isoformat(),
        "model_path": MODEL_PATH,
        "num_classes": num_classes,
        "class_names": CLASS_NAMES
    },
    "predictions": detailed_results
}

results_filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

with open(results_filename, 'w') as f:
    json.dump(results_json, f, indent=2)

logger.info("\n" + "="*70)
logger.info("TEST COMPLETE")
logger.info("="*70)
logger.info(f"📊 Total Images Processed: {len(detailed_results)}")
logger.info(f"📋 Log file: {log_filename}")
logger.info(f"📁 Results JSON: {results_filename}")
logger.info("="*70)