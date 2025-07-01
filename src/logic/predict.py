import os
import logging
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from transformers import pipeline
import torch

logger = logging.getLogger(__name__)

# Global classifier
classifier = None
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "final_roberta_persona"))

# Fault tolerance configuration for prediction
PREDICTION_CONFIG = {
    'max_retries': 3,
    'retry_delay': 1,
    'timeout_seconds': 30,
    'enable_fallback': True,
    'fallback_model': 'roberta-base',  # Use base model as fallback
    'cache_predictions': True,
    'cache_size': 1000,
    'model_health_check_interval': 300  # 5 minutes
}

class PredictionCache:
    """Simple in-memory cache for predictions."""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get prediction from cache."""
        with self.lock:
            return self.cache.get(key)
    
    def set(self, key: str, value: Dict[str, Any]):
        """Set prediction in cache."""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[key] = value
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()

# Global prediction cache
prediction_cache = PredictionCache(PREDICTION_CONFIG['cache_size'])

class ModelHealthMonitor:
    """Monitor model health and trigger reloads if needed."""
    def __init__(self):
        self.last_health_check = 0
        self.health_check_lock = threading.Lock()
        self.is_healthy = True
    
    def check_model_health(self) -> bool:
        """Check if the model is healthy."""
        with self.health_check_lock:
            current_time = time.time()
            
            # Only check periodically
            if current_time - self.last_health_check < PREDICTION_CONFIG['model_health_check_interval']:
                return self.is_healthy
            
            self.last_health_check = current_time
            
            try:
                # Test prediction with a simple input
                test_input = "This is a test input for health check"
                if classifier is not None:
                    result = classifier(test_input)
                    self.is_healthy = True
                    logger.debug("Model health check passed")
                else:
                    self.is_healthy = False
                    logger.warning("Model health check failed: classifier is None")
                    
            except Exception as e:
                self.is_healthy = False
                logger.error(f"Model health check failed: {e}")
            
            return self.is_healthy
    
    def mark_unhealthy(self):
        """Mark model as unhealthy to trigger reload."""
        with self.health_check_lock:
            self.is_healthy = False
            self.last_health_check = 0  # Force immediate recheck

# Global health monitor
health_monitor = ModelHealthMonitor()


def load_classifier(use_fallback: bool = False):
    """
    Loads or reloads the text classification pipeline with fault tolerance.
    """
    global classifier
    try:
        if use_fallback:
            logger.info("Loading fallback model: %s", PREDICTION_CONFIG['fallback_model'])
            classifier = pipeline(
                "text-classification",
                model=PREDICTION_CONFIG['fallback_model'],
                top_k=None
            )
        else:
            classifier = pipeline(
                "text-classification",
                model=MODEL_DIR,
                tokenizer=MODEL_DIR,
                top_k=None
            )
            logger.info("Model loaded successfully from: %s", MODEL_DIR)
        
        # Reset health monitor
        health_monitor.is_healthy = True
        return True
        
    except Exception as e:
        logger.error("Failed to load model pipeline: %s", e)
        
        if not use_fallback and PREDICTION_CONFIG['enable_fallback']:
            logger.info("Attempting to load fallback model...")
            return load_classifier(use_fallback=True)
        else:
            raise RuntimeError("Could not initialize model") from e

def predict_persona_with_fault_tolerance(text: str, max_retries: int = None) -> Tuple[str, Dict[str, float]]:
    """Predict persona with fault tolerance mechanisms."""
    if max_retries is None:
        max_retries = PREDICTION_CONFIG['max_retries']
    
    # Check cache first
    if PREDICTION_CONFIG['cache_predictions']:
        cache_key = f"prediction_{hash(text)}"
        cached_result = prediction_cache.get(cache_key)
        if cached_result:
            logger.debug("Returning cached prediction")
            return cached_result['label'], cached_result['scores']
    
    # Check model health
    if not health_monitor.check_model_health():
        logger.warning("Model health check failed, attempting reload")
        try:
            load_classifier()
        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            if PREDICTION_CONFIG['enable_fallback']:
                logger.info("Attempting fallback model")
                load_classifier(use_fallback=True)
    
    # Attempt prediction with retries
    for attempt in range(max_retries):
        try:
            if classifier is None:
                load_classifier()
            
            result = classifier(text)
            
            # Extract label and scores
            if isinstance(result, list) and len(result) > 0:
                predictions = result[0] if isinstance(result[0], list) else result
                label = predictions[0]['label']
                scores = {pred['label']: pred['score'] for pred in predictions}
                
                # Cache the result
                if PREDICTION_CONFIG['cache_predictions']:
                    cache_key = f"prediction_{hash(text)}"
                    prediction_cache.set(cache_key, {
                        'label': label,
                        'scores': scores
                    })
                
                return label, scores
            else:
                raise ValueError("Invalid prediction result format")
                
        except Exception as e:
            logger.error(f"Prediction attempt {attempt + 1} failed: {e}")
            
            # Mark model as unhealthy
            health_monitor.mark_unhealthy()
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying prediction in {PREDICTION_CONFIG['retry_delay']} seconds...")
                time.sleep(PREDICTION_CONFIG['retry_delay'])
                
                # Try to reload model
                try:
                    load_classifier()
                except Exception as reload_error:
                    logger.error(f"Failed to reload model: {reload_error}")
                    if PREDICTION_CONFIG['enable_fallback']:
                        load_classifier(use_fallback=True)
            else:
                # Final failure - return fallback prediction
                logger.error("All prediction attempts failed, returning fallback")
                return "Unknown", {"Unknown": 1.0}
    
    return "Unknown", {"Unknown": 1.0}

def predict_persona(text: str) -> Tuple[str, Dict[str, float]]:
    """Main prediction function with fault tolerance."""
    return predict_persona_with_fault_tolerance(text)

def get_prediction_stats() -> Dict[str, Any]:
    """Get prediction system statistics."""
    stats = {
        'cache_size': len(prediction_cache.cache),
        'cache_max_size': prediction_cache.max_size,
        'model_healthy': health_monitor.is_healthy,
        'model_dir': MODEL_DIR,
        'fallback_enabled': PREDICTION_CONFIG['enable_fallback']
    }
    
    # Check if model files exist
    model_path = Path(MODEL_DIR)
    stats['model_files_exist'] = all([
        (model_path / 'config.json').exists(),
        (model_path / 'pytorch_model.bin').exists(),
        (model_path / 'tokenizer.json').exists()
    ])
    
    return stats

def clear_prediction_cache():
    """Clear the prediction cache."""
    prediction_cache.clear()
    logger.info("Prediction cache cleared")


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
        
    Raises:
        ValueError: If text is empty, None, or too short
        RuntimeError: If classifier is not initialized
    """
    # Input validation
    if text is None:
        raise ValueError("Text input cannot be None")
    
    if not isinstance(text, str):
        raise ValueError("Text input must be a string")
    
    text = text.strip()
    if not text:
        raise ValueError("Text input cannot be empty or whitespace-only")
    
    if len(text.split()) < 5:
        raise ValueError("Text input must contain at least 5 words for meaningful prediction")
    
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