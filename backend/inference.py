import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from Project.src.model import SpatialVisionFusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Points to the best trained SpatialVisionFusion checkpoint.
# Will gracefully fall back to random weights until the model is trained.
CHECKPOINT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "models",
    "SpatialVisionFusion_experiment_1", "best_model.pt"
))

model = SpatialVisionFusion().to(device)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[inference] Loaded checkpoint: {CHECKPOINT_PATH}")
else:
    print(f"[inference] Warning: no checkpoint at {CHECKPOINT_PATH} — using random weights until trained.")

model.eval()

CLASS_NAMES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]


def predict(image_tensor: torch.Tensor, metadata_tensor: torch.Tensor) -> list:
    """
    Run inference on preprocessed tensors.

    Args:
        image_tensor:    (1, 3, H, W) float tensor, already normalized
        metadata_tensor: (1, 22) float tensor of encoded metadata features

    Returns:
        Top-3 predictions as list of {"label": str, "confidence": float}
    """
    with torch.no_grad():
        outputs = model(image_tensor.to(device), metadata_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_idx = torch.topk(probs, 3)

    return [
        {
            "label": CLASS_NAMES[top_idx[0][i].item()],
            "confidence": round(float(top_probs[0][i]), 4),
        }
        for i in range(3)
    ]
