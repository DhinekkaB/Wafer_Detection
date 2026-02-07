import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Class names (same order as your report)
class_names = [
    "bridge", "clean", "crack", "ler",
    "missing via", "open", "other",
    "particle", "scratch"
]

# Confusion matrix values (paste from your output)
cm = np.array([
    [9, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 4, 0, 0, 0, 0, 2, 0, 0],
    [0, 0, 7, 0, 0, 1, 0, 1, 2],
    [0, 0, 0, 9, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 0, 0, 0],
    [3, 0, 3, 0, 0, 10, 0, 0, 3],
    [1, 1, 0, 0, 0, 1, 5, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 14, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 15]
])

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Wafer Defect Classification")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

print("✅ Confusion matrix image saved as confusion_matrix.png")
