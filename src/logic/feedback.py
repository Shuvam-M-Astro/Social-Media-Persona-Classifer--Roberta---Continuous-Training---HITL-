import csv
import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def save_feedback(bio: str, posts: str, corrected_label: str, path="result.csv"):
    row = {"bio": bio, "posts": posts, "label": corrected_label}
    file_exists = os.path.isfile(path)
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        logger.info("Feedback saved.")
        return True
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        return False


def retrain_model(script_path=os.path.join("src", "logic", "train_model.py")):
    try:
        result = subprocess.run([
            "python", script_path
        ], capture_output=True, text=True, check=True)
        logger.info("Model retrained successfully.")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining failed: {e.stderr}")
        return False, e.stderr