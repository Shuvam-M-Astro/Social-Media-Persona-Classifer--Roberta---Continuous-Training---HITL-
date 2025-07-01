import pandas as pd
import numpy as np
import os
import json
import time
import shutil
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import logging
import torch
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fault tolerance configuration
FAULT_TOLERANCE_CONFIG = {
    'max_retries': 3,
    'retry_delay': 30,
    'backup_interval': 300,
    'checkpoint_dir': 'training_checkpoints',
    'enable_backup': True,
    'enable_checkpointing': True,
    'timeout_minutes': 30
}

class TrainingState:
    """Thread-safe training state management."""
    def __init__(self):
        self.is_training = False
        self.lock = threading.Lock()
        self.current_checkpoint = None
        self.start_time = None
    
    def start_training(self, checkpoint_id: str):
        with self.lock:
            if self.is_training:
                return False
            self.is_training = True
            self.current_checkpoint = checkpoint_id
            self.start_time = time.time()
            return True
    
    def stop_training(self):
        with self.lock:
            self.is_training = False
            self.current_checkpoint = None
            self.start_time = None
    
    def get_status(self):
        with self.lock:
            return {
                'is_training': self.is_training,
                'checkpoint': self.current_checkpoint,
                'elapsed_time': time.time() - self.start_time if self.start_time else 0
            }

# Global training state
training_state = TrainingState()

def create_checkpoint(data_hash: str, config: Dict[str, Any]) -> str:
    """Create a training checkpoint with metadata."""
    checkpoint_dir = Path(FAULT_TOLERANCE_CONFIG['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_path = checkpoint_dir / checkpoint_id
    checkpoint_path.mkdir(exist_ok=True)
    
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'data_hash': data_hash,
        'config': config,
        'status': 'in_progress',
        'model_path': str(Path('final_roberta_persona').absolute())
    }
    
    with open(checkpoint_path / 'metadata.json', 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    return checkpoint_id

def calculate_data_hash(data_paths: list) -> str:
    """Calculate hash of training data for change detection."""
    combined_hash = hashlib.md5()
    
    for path in data_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                combined_hash.update(f.read())
    
    return combined_hash.hexdigest()

def backup_model(source_path: str, backup_name: str = None):
    """Create a backup of the current model."""
    if not FAULT_TOLERANCE_CONFIG['enable_backup']:
        return
    
    try:
        if backup_name is None:
            backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_dir = Path(FAULT_TOLERANCE_CONFIG['checkpoint_dir']) / 'backups'
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / backup_name
        
        if os.path.exists(source_path):
            shutil.copytree(source_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            
            # Clean old backups (keep last 5)
            cleanup_old_backups(backup_dir, keep_count=5)
            
    except Exception as e:
        logger.error(f"Backup creation failed: {e}")

def cleanup_old_backups(backup_dir: Path, keep_count: int = 5):
    """Remove old backups, keeping only the most recent ones."""
    try:
        backups = sorted(backup_dir.glob("backup_*"), 
                        key=lambda x: x.stat().st_mtime, reverse=True)
        
        for backup in backups[keep_count:]:
            shutil.rmtree(backup)
            logger.info(f"Removed old backup: {backup}")
            
    except Exception as e:
        logger.error(f"Backup cleanup failed: {e}")

def verify_model_integrity(model_path: str) -> bool:
    """Verify that a model directory contains all required files."""
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
    
    try:
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                return False
        return True
    except Exception:
        return False

def train_model_with_fault_tolerance(enable_fault_tolerance: bool = True):
    """Main training function with fault tolerance."""
    if enable_fault_tolerance:
        return train_model_fault_tolerant()
    else:
        return train_model_original()

def train_model_fault_tolerant():
    """Fault-tolerant version of train_model."""
    # Check if training is already in progress
    if not training_state.start_training("fault_tolerant"):
        logger.warning("Training already in progress")
        return
    
    try:
        # Calculate data hash
        data_paths = ["persona_dataset.csv", "result.csv"]
        data_hash = calculate_data_hash(data_paths)
        
        # Create checkpoint
        checkpoint_id = None
        if FAULT_TOLERANCE_CONFIG['enable_checkpointing']:
            config = {
                'max_length': 256,
                'batch_size': 16,
                'epochs': 5,
                'learning_rate': 2e-5
            }
            checkpoint_id = create_checkpoint(data_hash, config)
            training_state.current_checkpoint = checkpoint_id
        
        # Create backup before training
        backup_model("final_roberta_persona")
        
        # Attempt training with retries
        for attempt in range(FAULT_TOLERANCE_CONFIG['max_retries']):
            try:
                logger.info(f"Training attempt {attempt + 1}/{FAULT_TOLERANCE_CONFIG['max_retries']}")
                
                # Run original training logic
                success = train_model_original()
                
                if success:
                    # Update checkpoint status
                    if checkpoint_id:
                        update_checkpoint_status(checkpoint_id, 'completed')
                    logger.info("Training completed successfully")
                    return True
                else:
                    logger.warning(f"Training attempt {attempt + 1} failed")
                    
                    if attempt < FAULT_TOLERANCE_CONFIG['max_retries'] - 1:
                        logger.info(f"Retrying in {FAULT_TOLERANCE_CONFIG['retry_delay']} seconds...")
                        time.sleep(FAULT_TOLERANCE_CONFIG['retry_delay'])
                    else:
                        if checkpoint_id:
                            update_checkpoint_status(checkpoint_id, 'failed')
                        logger.error("Training failed after all retry attempts")
                        return False
            
            except Exception as e:
                logger.error(f"Training attempt {attempt + 1} exception: {e}")
                
                if attempt < FAULT_TOLERANCE_CONFIG['max_retries'] - 1:
                    time.sleep(FAULT_TOLERANCE_CONFIG['retry_delay'])
                else:
                    if checkpoint_id:
                        update_checkpoint_status(checkpoint_id, 'failed')
                    logger.error(f"Training failed with exception: {e}")
                    return False
    
    finally:
        training_state.stop_training()

def update_checkpoint_status(checkpoint_id: str, status: str):
    """Update checkpoint status."""
    try:
        checkpoint_dir = Path(FAULT_TOLERANCE_CONFIG['checkpoint_dir']) / checkpoint_id
        metadata_file = checkpoint_dir / 'metadata.json'
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            metadata['status'] = status
            metadata['completion_time'] = datetime.now().isoformat()
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
    except Exception as e:
        logger.error(f"Failed to update checkpoint status: {e}")

def train_model_original():
    """Original training logic without fault tolerance."""
    if torch.cuda.is_available():
        logger.info("GPU in use: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("GPU not available. Running on CPU.")

    # Load datasets
    try:
        df_main = pd.read_csv("persona_dataset.csv")
        df_main["text"] = df_main["bio"] + " " + df_main["posts"]
        df_main = df_main[["text", "label"]]
        logger.info("Loaded main dataset with %d records.", len(df_main))
    except Exception as e:
        logger.error("Failed to load persona_dataset.csv: %s", e)
        return

    if os.path.isfile("result.csv"):
        try:
            df_feedback = pd.read_csv("result.csv")
            df_feedback["text"] = df_feedback["bio"] + " " + df_feedback["posts"]
            df_feedback = df_feedback[["text", "label"]]
            logger.info("Loaded %d feedback samples.", len(df_feedback))
        except Exception as e:
            logger.warning("Failed to load result.csv: %s", e)
            df_feedback = pd.DataFrame(columns=["text", "label"])
    else:
        logger.info("No feedback data found.")
        df_feedback = pd.DataFrame(columns=["text", "label"])

    # Combine and deduplicate
    df = pd.concat([df_main, df_feedback], ignore_index=True)
    df.drop_duplicates(subset=["text", "label"], inplace=True)
    logger.info("Total samples after merge and deduplication: %d", len(df))

    # Oversample underrepresented classes
    min_count = 1
    counts = df["label"].value_counts()
    for label, count in counts.items():
        if count < min_count:
            to_add = df[df["label"] == label]
            repeats = (min_count - count) // count + 1
            df = pd.concat([df, to_add] * repeats, ignore_index=True)
    logger.info("Total samples after oversampling: %d", len(df))

    # Save processed dataset
    df.to_csv("processed_personas.csv", index=False)
    df["label"].value_counts().to_csv("label_distribution.csv")

    # Label mappings
    label2id_path = "label2id.json"
    id2label_path = "id2label.json"
    if os.path.isfile(label2id_path) and os.path.isfile(id2label_path):
        with open(label2id_path, "r") as f:
            label2id = json.load(f)
        with open(id2label_path, "r") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
        logger.info("Loaded existing label mappings.")
    else:
        label2id = {}
        id2label = {}
        logger.info("No existing label mappings found. Starting fresh.")

    next_id = max(id2label.keys(), default=-1) + 1
    for label in sorted(df["label"].unique()):
        if label not in label2id:
            label2id[label] = next_id
            id2label[next_id] = label
            next_id += 1

    with open(label2id_path, "w") as f:
        json.dump(label2id, f)
    with open(id2label_path, "w") as f:
        json.dump(id2label, f)
    logger.info("Updated and saved label mappings.")

    # Encode labels
    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    # Tokenize
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(encode_labels)
    tokenized_dataset = dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True)

    # Load model
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"models/roberta_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    os.makedirs("final_roberta_persona", exist_ok=True)
    model.save_pretrained("final_roberta_persona")
    tokenizer.save_pretrained("final_roberta_persona")
    logger.info("Final model saved to final_roberta_persona")

    with open(f"{output_dir}/training_summary.txt", "w") as f:
        f.write(f"Trained on {len(df)} samples across {len(label2id)} classes\n")
        for i, label in id2label.items():
            f.write(f"{i}: {label}\n")
    
    return True

def get_training_status() -> Dict[str, Any]:
    """Get current training status and statistics."""
    status = training_state.get_status()
    
    # Get checkpoint statistics
    checkpoint_dir = Path(FAULT_TOLERANCE_CONFIG['checkpoint_dir'])
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*"))
        total_checkpoints = len(checkpoint_files)
        
        completed_checkpoints = 0
        failed_checkpoints = 0
        
        for checkpoint_path in checkpoint_files:
            metadata_file = checkpoint_path / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        if metadata.get('status') == 'completed':
                            completed_checkpoints += 1
                        elif metadata.get('status') == 'failed':
                            failed_checkpoints += 1
                except Exception:
                    pass
        
        status.update({
            'total_checkpoints': total_checkpoints,
            'completed_checkpoints': completed_checkpoints,
            'failed_checkpoints': failed_checkpoints,
            'success_rate': completed_checkpoints / max(total_checkpoints, 1)
        })
    
    return status

def cleanup_failed_checkpoints() -> int:
    """Remove failed checkpoints to free up space."""
    removed_count = 0
    checkpoint_dir = Path(FAULT_TOLERANCE_CONFIG['checkpoint_dir'])
    
    if checkpoint_dir.exists():
        for checkpoint_path in checkpoint_dir.glob("checkpoint_*"):
            metadata_file = checkpoint_path / 'metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata.get('status') == 'failed':
                        shutil.rmtree(checkpoint_path)
                        removed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process checkpoint {checkpoint_path}: {e}")
    
    logger.info(f"Removed {removed_count} failed checkpoints")
    return removed_count

# Backward compatibility
def train_model():
    """Main training function - uses fault tolerance by default."""
    return train_model_with_fault_tolerance(enable_fault_tolerance=True)

if __name__ == "__main__":
    # Allow command line arguments for fault tolerance
    import sys
    enable_fault_tolerance = True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--no-fault-tolerance":
            enable_fault_tolerance = False
        elif sys.argv[1] == "--status":
            status = get_training_status()
            print(json.dumps(status, indent=2))
            sys.exit(0)
        elif sys.argv[1] == "--cleanup":
            removed = cleanup_failed_checkpoints()
            print(f"Removed {removed} failed checkpoints")
            sys.exit(0)
    
    train_model_with_fault_tolerance(enable_fault_tolerance)
