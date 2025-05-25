import os
import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

# Global classifier
classifier = None
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "final_roberta_persona"))


def load_classifier():
    """
    Loads or reloads the text classification pipeline from local model directory.
    """
    global classifier
    try:
        classifier = pipeline(
            "text-classification",
            model=MODEL_DIR,
            tokenizer=MODEL_DIR,
            top_k=None
        )
        logger.info("Model loaded successfully from: %s", MODEL_DIR)
    except Exception as e:
        logger.error("Failed to load model pipeline: %s", e)
        raise RuntimeError("Could not initialize model") from e


# Load classifier once on import
load_classifier()


def predict_persona(text: str):
    """
    Predict persona label for a given text.

    Args:
        text (str): Combined bio and post input.

    Returns:
        top_label (str): Highest scoring label
        scores (dict): Score for each label
    """
    if classifier is None:
        raise RuntimeError("Classifier not initialized")
    try:
        results = classifier(text)[0]
        sorted_scores = sorted(results, key=lambda x: x["score"], reverse=True)
        top_label = sorted_scores[0]["label"]
        scores = {item["label"]: item["score"] for item in sorted_scores}
        logger.info("Prediction complete. Top label: %s", top_label)
        return top_label, scores
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise