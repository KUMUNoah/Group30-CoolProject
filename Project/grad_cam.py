"""
Grad-CAM visualization for SpatialVisionFusion.

Targets the last convolutional layer of the ResNet-50 backbone
(resnet_feature_extractor, i.e. layer4) and overlays the activation
map on the input image.

Run from the Project/ directory:
    python grad_cam.py

Output is saved to ../docs/gradcam_sample.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from data.dataloader import get_dataloaders
from src.model import SpatialVisionFusion

# ── config ────────────────────────────────────────────────────────────────────
import kagglehub as _kagglehub
DATA_ROOT = _kagglehub.dataset_download("mahdavi1202/skin-cancer")

CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..",
    "models", "SpatialVisionFusion_experiment_1", "best_model.pt",
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "gradcam_sample.png")

CLASS_NAMES = ["BCC", "SCC", "ACK", "SEK", "MEL", "NEV"]

# ── Grad-CAM helpers ──────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for a target layer.  Works by registering forward / backward
    hooks on the chosen module.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,   # (1, 3, H, W)
        metadata_tensor: torch.Tensor,  # (1, 22)
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Returns:
            cam      – (H, W) float32 array in [0, 1]
            pred_cls – predicted class index
            pred_conf – predicted class confidence
        """
        self.model.eval()
        image_tensor = image_tensor.requires_grad_(False)

        # Forward
        output = self.model(image_tensor, metadata_tensor)
        probs = torch.softmax(output, dim=1)
        pred_cls = int(probs.argmax(dim=1).item())
        pred_conf = float(probs[0, pred_cls].item())

        if target_class is None:
            target_class = pred_cls

        # Backward on the target class score
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward()

        # Global-average-pool the gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam).squeeze()  # (h, w)

        # Normalise to [0, 1]
        cam = cam.cpu().numpy().astype(np.float32)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_cls, pred_conf


def cam_to_heatmap(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Upsample CAM to `size` (H, W) and convert to an RGB heatmap array."""
    pil_cam = Image.fromarray(np.uint8(cam * 255)).resize(
        (size[1], size[0]), Image.BILINEAR
    )
    cam_resized = np.array(pil_cam) / 255.0
    cmap = plt.get_cmap("jet")
    heatmap = np.delete(cmap(cam_resized), 3, axis=2)  # drop alpha → (H, W, 3)
    return heatmap.astype(np.float32)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor to a uint8 numpy array."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load data ──
    print("[grad_cam] Loading data …")
    _, _, test_loader = get_dataloaders(DATA_ROOT, batch_size=1, num_workers=0)

    # Grab 6 samples – one per class if possible
    samples_per_class: dict[int, dict] = {}
    for batch in test_loader:
        label = int(batch["label"].item())
        if label not in samples_per_class:
            samples_per_class[label] = {
                "image":    batch["image"],
                "metadata": batch["metadata"],
                "label":    label,
                "img_id":   batch["img_id"][0],
                "true_cls": CLASS_NAMES[label],
            }
        if len(samples_per_class) == len(CLASS_NAMES):
            break

    print(f"[grad_cam] Collected {len(samples_per_class)} sample(s).")

    # ── load model ──
    model = SpatialVisionFusion().to(device)
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[grad_cam] Loaded checkpoint: {CHECKPOINT}")
    else:
        print("[grad_cam] Warning: checkpoint not found – using random weights.")

    # ── attach Grad-CAM to the last block of layer4 ──
    target_layer = model.resnet_feature_extractor[-1]  # BasicBlock / Bottleneck
    grad_cam = GradCAM(model, target_layer)

    # ── generate & plot ──
    n = len(samples_per_class)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle("Grad-CAM – SpatialVisionFusion (ResNet-50 layer4)", fontsize=13)

    for row, (label_idx, sample) in enumerate(sorted(samples_per_class.items())):
        img_t = sample["image"].to(device)
        meta_t = sample["metadata"].to(device)

        cam, pred_cls, pred_conf = grad_cam.generate(img_t, meta_t)

        orig_np = denormalize(img_t)
        h, w = orig_np.shape[:2]
        heatmap = cam_to_heatmap(cam, (h, w))
        overlay = np.clip(np.float32(orig_np) / 255.0 * 0.55 + heatmap * 0.45, 0, 1)

        true_name = sample["true_cls"]
        pred_name = CLASS_NAMES[pred_cls]

        axes[row][0].imshow(orig_np)
        axes[row][0].set_title(f"Original\n(true: {true_name})", fontsize=8)
        axes[row][0].axis("off")

        axes[row][1].imshow(heatmap)
        axes[row][1].set_title("Grad-CAM heatmap", fontsize=8)
        axes[row][1].axis("off")

        axes[row][2].imshow(overlay)
        axes[row][2].set_title(
            f"Overlay\npred: {pred_name} ({pred_conf:.1%})", fontsize=8
        )
        axes[row][2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"[grad_cam] Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
