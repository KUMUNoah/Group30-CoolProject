import sys
import os
import io
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from torchvision import transforms
import torch
from dotenv import load_dotenv

load_dotenv()

from inference import predict
from rag import run_pipeline

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Image preprocessing — matches eval transform in dataloader (300x300)
# ---------------------------------------------------------------------------
eval_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Metadata encoding
# Feature order must match CATEGORICAL_COLS + NUMERICAL_COLS in dataloader.py:
#   Categorical (19): smoke, drink, background_father, background_mother,
#     pesticide, gender, skin_cancer_history, cancer_history, has_piped_water,
#     has_sewage_system, fitspatrick, region, itch, grew, hurt, changed,
#     bleed, elevation, biopsed
#   Numerical (3): age, diameter_1, diameter_2
# ---------------------------------------------------------------------------

def encode_metadata(data: dict) -> torch.Tensor:
    """
    Encode a metadata dict from the request into a (1, 22) float tensor.
    Binary fields accept 0/1. Numerical fields are normalized by typical ranges.
    Complex categoricals (background, region) default to 0 since we don't
    have the training cat_maps at inference time.
    """
    features = [
        float(data.get("smoke", 0)),
        float(data.get("drink", 0)),
        0.0,  # background_father — complex categorical, default unknown
        0.0,  # background_mother — complex categorical, default unknown
        float(data.get("pesticide", 0)),
        float(data.get("gender", 0)),          # 0 = female, 1 = male
        float(data.get("skin_cancer_history", 0)),
        float(data.get("cancer_history", 0)),
        float(data.get("has_piped_water", 0)),
        float(data.get("has_sewage_system", 0)),
        float(data.get("fitspatrick", 3)) / 6.0,  # normalize 1-6 → 0-1
        0.0,  # region — complex categorical, default unknown
        float(data.get("itch", 0)),
        float(data.get("grew", 0)),
        float(data.get("hurt", 0)),
        float(data.get("changed", 0)),
        float(data.get("bleed", 0)),
        float(data.get("elevation", 0)),
        float(data.get("biopsed", 0)),
        # Numerical: normalize by typical ranges
        float(data.get("age", 50)) / 100.0,
        float(data.get("diameter_1", 10)) / 30.0,
        float(data.get("diameter_2", 10)) / 30.0,
    ]
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, 22)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict_route():
    """Basic prediction endpoint using mock tensors."""
    image_tensor = torch.randn(1, 3, 300, 300)
    metadata_tensor = torch.randn(1, 22)
    predictions = predict(image_tensor, metadata_tensor)
    return jsonify({"predictions": predictions})


@app.route("/diagnose", methods=["POST"])
def diagnose_route():
    """
    Full pipeline: image + metadata → CNN prediction → RAG explanation.

    Expects multipart/form-data with:
        image:       image file (jpg/png)
        metadata:    JSON string of symptom checkboxes and patient info
        description: free-text patient description of symptoms
    """
    # --- Image ---
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
    image_tensor = eval_transform(image).unsqueeze(0)  # (1, 3, 300, 300)

    # --- Metadata ---
    metadata_dict = json.loads(request.form.get("metadata", "{}"))
    metadata_tensor = encode_metadata(metadata_dict)

    # --- Description ---
    description = request.form.get("description", "")

    # --- CNN Inference ---
    predictions = predict(image_tensor, metadata_tensor)
    top_prediction = predictions[0]  # {"label": "BCC", "confidence": 0.87}

    # --- RAG Pipeline ---
    try:
        rag_result = run_pipeline(metadata_dict, description, top_prediction)
    except Exception as e:
        rag_result = {"error": str(e)}

    return jsonify({
        "predictions": predictions,
        "rag": rag_result,
    })


@app.route("/health")
def health():
    return jsonify({"status": "SkinAI running"})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
