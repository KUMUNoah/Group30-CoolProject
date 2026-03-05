from flask import Flask, request, jsonify
import torch
from inference import predict

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict_route():

    # Normally you would process image + metadata here
    # For now we mock tensors

    image_tensor = torch.randn(1,3,224,224)
    metadata_tensor = torch.randn(1,22)

    predictions = predict(image_tensor, metadata_tensor)

    return jsonify({
        "predictions": predictions
    })


@app.route("/health")
def health():
    return {"status": "SkinAI running"}

if __name__ == "__main__":
    app.run(port=5000, debug=True)