import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from Project.src.retriever import retrieve
from Project.src.generator import generate_response


def run_pipeline(metadata: dict, description: str, cnn_prediction: dict) -> dict:
    """
    Full RAG pipeline: retrieve relevant chunks then generate an explanation.

    Args:
        metadata:       patient symptom dict (checkboxes + numerical fields)
        description:    free-text patient description of their symptoms
        cnn_prediction: top CNN prediction {"label": str, "confidence": float}

    Returns:
        dict with keys: cnn_prediction, retrieved_diseases, explanation, sources
    """
    chunks = retrieve(metadata, description)

    explanation = generate_response(description, metadata, cnn_prediction, chunks)

    retrieved_diseases = list({c["disease"] for c in chunks})
    sources = list({c["url"] for c in chunks})

    return {
        "cnn_prediction": cnn_prediction,
        "retrieved_diseases": retrieved_diseases,
        "explanation": explanation,
        "sources": sources,
    }
