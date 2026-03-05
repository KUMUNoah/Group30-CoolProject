import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from Project.src.model import SpatialVisionFusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model
model = SpatialVisionFusion().to(device)

checkpoint = torch.load(
    "../Project/models/SpatialVisionFusion_experiment_2/best_model.pt",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

class_names = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]

def predict(image_tensor, metadata_tensor):
    with torch.no_grad():
        outputs = model(image_tensor.to(device), metadata_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)

        top_probs, top_idx = torch.topk(probs, 3)

    results = []
    for i in range(3):
        results.append({
            "label": class_names[top_idx[0][i].item()],
            "prob": float(top_probs[0][i])
        })

    return results