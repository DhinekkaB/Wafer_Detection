import torch
import torch.nn as nn
from torchvision import models

IMG_SIZE = 160
NUM_CLASSES = 9   # update if needed
MODEL_PATH = "mobilenetv3_wafer_best.pth"
ONNX_PATH = "mobilenetv3_wafer.onnx"

# Load model
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

# Export
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("✅ ONNX export completed:", ONNX_PATH)
