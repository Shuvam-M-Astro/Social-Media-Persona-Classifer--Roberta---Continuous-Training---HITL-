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
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic
import torch.nn.functional as F

# Import data validation
from .data_validation import DataValidator, ValidationResult, validate_training_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance optimization configuration
PERFORMANCE_CONFIG = {
    'enable_mixed_precision': True,
    'enable_gradient_accumulation': True,
    'gradient_accumulation_steps': 4,
    'enable_pruning': False,
    'pruning_amount': 0.3,  # 30% of weights to prune
    'pruning_type': 'l1_unstructured',  # 'l1_unstructured', 'random_unstructured', 'ln_structured'
    'enable_quantization': False,
    'quantization_type': 'dynamic',  # 'dynamic', 'static'
    'target_model_size_mb': 100,  # Target model size in MB
    'enable_amp': True,  # Automatic Mixed Precision
    'fp16': True,
    'bf16': False,  # Use bfloat16 if available
}

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

class PerformanceOptimizer:
    """Handles performance optimizations for training and inference."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = None
        self.original_model_size = None
        
    def setup_mixed_precision(self):
        """Setup mixed precision training."""
        if not self.config['enable_mixed_precision']:
            return None
            
        try:
            if self.config['bf16'] and torch.cuda.is_available():
                # Use bfloat16 if available (better numerical stability)
                logger.info("Using bfloat16 mixed precision")
                return torch.cuda.amp.GradScaler()
            elif self.config['fp16']:
                # Use float16 mixed precision
                logger.info("Using float16 mixed precision")
                return torch.cuda.amp.GradScaler()
            else:
                logger.info("Mixed precision disabled")
                return None
        except Exception as e:
            logger.warning(f"Failed to setup mixed precision: {e}")
            return None
    
    def get_training_arguments_with_optimizations(self, base_args: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance training arguments with performance optimizations."""
        optimized_args = base_args.copy()
        
        # Mixed precision settings
        if self.config['enable_mixed_precision']:
            optimized_args.update({
                'fp16': self.config['fp16'],
                'bf16': self.config['bf16'],
                'dataloader_pin_memory': True,
                'dataloader_num_workers': 4,
            })
        
        # Gradient accumulation
        if self.config['enable_gradient_accumulation']:
            optimized_args['gradient_accumulation_steps'] = self.config['gradient_accumulation_steps']
            # Adjust batch size for effective batch size
            effective_batch_size = optimized_args.get('per_device_train_batch_size', 16) * self.config['gradient_accumulation_steps']
            logger.info(f"Effective batch size: {effective_batch_size}")
        
        return optimized_args
    
    def apply_pruning(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply model pruning to reduce model size."""
        if not self.config['enable_pruning']:
            return model
            
        try:
            logger.info(f"Applying {self.config['pruning_type']} pruning...")
            
            # Store original model size
            self.original_model_size = self.get_model_size_mb(model)
            
            # Apply pruning based on type
            if self.config['pruning_type'] == 'l1_unstructured':
                # L1 unstructured pruning
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(
                            module, 
                            name='weight', 
                            amount=self.config['pruning_amount']
                        )
            
            elif self.config['pruning_type'] == 'random_unstructured':
                # Random unstructured pruning
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        prune.random_unstructured(
                            module, 
                            name='weight', 
                            amount=self.config['pruning_amount']
                        )
            
            elif self.config['pruning_type'] == 'ln_structured':
                # Ln structured pruning
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        prune.ln_structured(
                            module, 
                            name='weight', 
                            amount=self.config['pruning_amount'],
                            n=2,  # L2 norm
                            dim=0
                        )
            
            # Calculate pruning statistics
            total_params = sum(p.numel() for p in model.parameters())
            zero_params = sum(p.numel() for p in model.parameters() if p.data.eq(0).all())
            sparsity = zero_params / total_params
            
            logger.info(f"Pruning applied. Sparsity: {sparsity:.2%}")
            logger.info(f"Zero parameters: {zero_params:,} / {total_params:,}")
            
            return model
            
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model
    
    def apply_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply post-training quantization."""
        if not self.config['enable_quantization']:
            return model
            
        try:
            logger.info(f"Applying {self.config['quantization_type']} quantization...")
            
            if self.config['quantization_type'] == 'dynamic':
                # Dynamic quantization (no calibration data needed)
                quantized_model = quantize_dynamic(
                    model, 
                    {torch.nn.Linear, torch.nn.LSTM, torch.nn.LSTMCell, torch.nn.RNNCell, torch.nn.GRUCell}, 
                    dtype=torch.qint8
                )
                
                # Calculate size reduction
                original_size = self.get_model_size_mb(model)
                quantized_size = self.get_model_size_mb(quantized_model)
                size_reduction = (original_size - quantized_size) / original_size
                
                logger.info(f"Dynamic quantization applied. Size reduction: {size_reduction:.2%}")
                logger.info(f"Model size: {original_size:.1f}MB -> {quantized_size:.1f}MB")
                
                return quantized_model
            
            elif self.config['quantization_type'] == 'static':
                # Static quantization requires calibration data
                logger.warning("Static quantization requires calibration data. Using dynamic quantization instead.")
                return self.apply_quantization(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model
    
    def get_model_size_mb(self, model: torch.nn.Module) -> float:
        """Calculate model size in MB."""
        try:
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            size_mb = (param_size + buffer_size) / 1024 / 1024
            return size_mb
            
        except Exception as e:
            logger.error(f"Failed to calculate model size: {e}")
            return 0.0
    
    def optimize_model_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply all optimizations for inference."""
        logger.info("Optimizing model for inference...")
        
        # Apply pruning if enabled
        if self.config['enable_pruning']:
            model = self.apply_pruning(model)
        
        # Apply quantization if enabled
        if self.config['enable_quantization']:
            model = self.apply_quantization(model)
        
        # Optimize for inference
        model.eval()
        
        # Use TorchScript for faster inference if possible
        try:
            if torch.cuda.is_available():
                model = model.half()  # Use FP16 for inference
                logger.info("Model converted to FP16 for inference")
        except Exception as e:
            logger.warning(f"Failed to convert model to FP16: {e}")
        
        return model
    
    def get_optimization_stats(self, model: torch.nn.Module) -> Dict[str, Any]:
        """Get statistics about applied optimizations."""
        stats = {
            'model_size_mb': self.get_model_size_mb(model),
            'pruning_enabled': self.config['enable_pruning'],
            'quantization_enabled': self.config['enable_quantization'],
            'mixed_precision_enabled': self.config['enable_mixed_precision'],
            'gradient_accumulation_enabled': self.config['enable_gradient_accumulation'],
        }
        
        if self.original_model_size:
            stats['original_size_mb'] = self.original_model_size
            stats['size_reduction_mb'] = self.original_model_size - stats['model_size_mb']
            stats['size_reduction_percent'] = (stats['size_reduction_mb'] / self.original_model_size) * 100
        
        return stats

class OptimizedTrainer(Trainer):
    """Enhanced Trainer with performance optimizations."""
    
    def __init__(self, *args, performance_optimizer: PerformanceOptimizer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.performance_optimizer = performance_optimizer
        self.scaler = None
        
        if performance_optimizer:
            self.scaler = performance_optimizer.setup_mixed_precision()
    
    def training_step(self, model, inputs):
        """Enhanced training step with mixed precision."""
        if self.scaler is not None:
            # Use mixed precision training
            with torch.cuda.amp.autocast():
                loss = super().training_step(model, inputs)
            
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
            return loss
        else:
            return super().training_step(model, inputs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Enhanced loss computation with mixed precision."""
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                return super().compute_loss(model, inputs, return_outputs)
        else:
            return super().compute_loss(model, inputs, return_outputs)

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

def validate_training_data_before_training(enable_validation: bool = True, 
                                         strict_mode: bool = False) -> ValidationResult:
    """Validate training data before starting the training process."""
    if not enable_validation:
        logger.info("Data validation disabled")
        return ValidationResult(True, [], [], {}, [], [])
    
    logger.info("Starting data validation before training...")
    
    try:
        # Run comprehensive validation
        validation_result = validate_training_data(
            main_data_path="persona_dataset.csv",
            feedback_data_path="result.csv",
            output_dir="validation_results"
        )
        
        # Log validation results
        if validation_result.is_valid:
            logger.info("✅ Data validation passed")
        else:
            logger.error("❌ Data validation failed")
            for error in validation_result.errors:
                logger.error(f"  - {error}")
        
        # Log warnings
        for warning in validation_result.warnings:
            logger.warning(f"⚠️  {warning}")
        
        # Log recommendations
        for rec in validation_result.recommendations:
            logger.info(f"💡 {rec}")
        
        # In strict mode, fail training if validation fails
        if strict_mode and not validation_result.is_valid:
            logger.error("Training aborted due to data validation failures in strict mode")
            return validation_result
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Data validation failed with exception: {e}")
        if strict_mode:
            return ValidationResult(False, [f"Validation exception: {e}"], [], {}, [], [])
        else:
            logger.warning("Continuing with training despite validation failure")
            return ValidationResult(True, [], [f"Validation failed: {e}"], {}, [], [])

def train_model_with_fault_tolerance(enable_fault_tolerance: bool = True, 
                                   enable_validation: bool = True,
                                   strict_validation: bool = False,
                                   performance_config: Dict[str, Any] = None):
    """Main training function with fault tolerance, data validation, and performance optimizations."""
    
    # Run data validation first
    validation_result = validate_training_data_before_training(
        enable_validation=enable_validation,
        strict_mode=strict_validation
    )
    
    # If validation failed in strict mode, abort training
    if strict_validation and not validation_result.is_valid:
        logger.error("Training aborted due to data validation failures")
        return False
    
    # Continue with training
    if enable_fault_tolerance:
        return train_model_fault_tolerant(performance_config)
    else:
        return train_model_original(performance_config)

def train_model_fault_tolerant(performance_config: Dict[str, Any] = None):
    """Fault-tolerant version of train_model with performance optimizations."""
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
                'learning_rate': 2e-5,
                'performance_config': performance_config or PERFORMANCE_CONFIG
            }
            checkpoint_id = create_checkpoint(data_hash, config)
            training_state.current_checkpoint = checkpoint_id
        
        # Create backup before training
        backup_model("final_roberta_persona")
        
        # Attempt training with retries
        for attempt in range(FAULT_TOLERANCE_CONFIG['max_retries']):
            try:
                logger.info(f"Training attempt {attempt + 1}/{FAULT_TOLERANCE_CONFIG['max_retries']}")
                
                # Run enhanced training logic with performance optimizations
                success = train_model_original(performance_config)
                
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

def train_model_original(performance_config: Dict[str, Any] = None):
    """Enhanced training logic with performance optimizations."""
    if performance_config is None:
        performance_config = PERFORMANCE_CONFIG
    
    # Initialize performance optimizer
    optimizer = PerformanceOptimizer(performance_config)
    
    if torch.cuda.is_available():
        logger.info("GPU in use: %s", torch.cuda.get_device_name(0))
        logger.info("GPU Memory: %s", torch.cuda.get_device_properties(0))
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
        return False

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

    # Log original model size
    original_size = optimizer.get_model_size_mb(model)
    logger.info(f"Original model size: {original_size:.1f}MB")

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"models/roberta_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Base training arguments
    base_training_args = {
        "output_dir": output_dir,
        "evaluation_strategy": "epoch",
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "num_train_epochs": 5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "disable_tqdm": True,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "logging_dir": f"{output_dir}/logs",
        "save_total_limit": 3,
    }

    # Apply performance optimizations to training arguments
    training_args_dict = optimizer.get_training_arguments_with_optimizations(base_training_args)
    training_args = TrainingArguments(**training_args_dict)

    # Use optimized trainer if performance optimizations are enabled
    if (performance_config['enable_mixed_precision'] or 
        performance_config['enable_gradient_accumulation']):
        trainer = OptimizedTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            performance_optimizer=optimizer
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

    # Train the model
    logger.info("Starting training with performance optimizations...")
    trainer.train()

    # Apply post-training optimizations
    logger.info("Applying post-training optimizations...")
    optimized_model = optimizer.optimize_model_for_inference(model)

    # Save optimized model
    os.makedirs("final_roberta_persona", exist_ok=True)
    optimized_model.save_pretrained("final_roberta_persona")
    tokenizer.save_pretrained("final_roberta_persona")
    
    # Save optimization statistics
    optimization_stats = optimizer.get_optimization_stats(optimized_model)
    with open("final_roberta_persona/optimization_stats.json", "w") as f:
        json.dump(optimization_stats, f, indent=2)
    
    logger.info("Final optimized model saved to final_roberta_persona")
    logger.info(f"Model size: {optimization_stats['model_size_mb']:.1f}MB")
    if 'size_reduction_percent' in optimization_stats:
        logger.info(f"Size reduction: {optimization_stats['size_reduction_percent']:.1f}%")

    # Save training summary with optimization info
    with open(f"{output_dir}/training_summary.txt", "w") as f:
        f.write(f"Trained on {len(df)} samples across {len(label2id)} classes\n")
        f.write(f"Original model size: {original_size:.1f}MB\n")
        f.write(f"Final model size: {optimization_stats['model_size_mb']:.1f}MB\n")
        if 'size_reduction_percent' in optimization_stats:
            f.write(f"Size reduction: {optimization_stats['size_reduction_percent']:.1f}%\n")
        f.write(f"Mixed precision: {optimization_stats['mixed_precision_enabled']}\n")
        f.write(f"Gradient accumulation: {optimization_stats['gradient_accumulation_enabled']}\n")
        f.write(f"Pruning: {optimization_stats['pruning_enabled']}\n")
        f.write(f"Quantization: {optimization_stats['quantization_enabled']}\n")
        f.write("\nLabel mappings:\n")
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
def train_model(enable_validation: bool = True, strict_validation: bool = False, 
               performance_config: Dict[str, Any] = None):
    """Main training function - uses fault tolerance by default with optional validation and performance optimizations."""
    return train_model_with_fault_tolerance(
        enable_fault_tolerance=True,
        enable_validation=enable_validation,
        strict_validation=strict_validation,
        performance_config=performance_config
    )

def get_performance_config() -> Dict[str, Any]:
    """Get current performance optimization configuration."""
    return PERFORMANCE_CONFIG.copy()

def update_performance_config(new_config: Dict[str, Any]):
    """Update performance optimization configuration."""
    global PERFORMANCE_CONFIG
    PERFORMANCE_CONFIG.update(new_config)
    logger.info("Performance configuration updated")

def create_optimized_config(
    enable_mixed_precision: bool = True,
    enable_gradient_accumulation: bool = True,
    gradient_accumulation_steps: int = 4,
    enable_pruning: bool = False,
    pruning_amount: float = 0.3,
    pruning_type: str = 'l1_unstructured',
    enable_quantization: bool = False,
    quantization_type: str = 'dynamic'
) -> Dict[str, Any]:
    """Create a custom performance optimization configuration."""
    config = PERFORMANCE_CONFIG.copy()
    config.update({
        'enable_mixed_precision': enable_mixed_precision,
        'enable_gradient_accumulation': enable_gradient_accumulation,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'enable_pruning': enable_pruning,
        'pruning_amount': pruning_amount,
        'pruning_type': pruning_type,
        'enable_quantization': enable_quantization,
        'quantization_type': quantization_type,
    })
    return config

def benchmark_model_performance(model_path: str = "final_roberta_persona") -> Dict[str, Any]:
    """Benchmark model performance and size."""
    try:
        from transformers import RobertaForSequenceClassification, RobertaTokenizer
        import time
        
        # Load model
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        
        # Calculate model size
        optimizer = PerformanceOptimizer(PERFORMANCE_CONFIG)
        model_size = optimizer.get_model_size_mb(model)
        
        # Benchmark inference speed
        model.eval()
        test_texts = [
            "I love technology and coding!",
            "Fitness is my passion, I work out daily.",
            "Food is life, I'm always cooking new recipes.",
            "Fashion is everything to me.",
            "I love sharing memes and jokes!"
        ]
        
        # Warm up
        for _ in range(3):
            inputs = tokenizer(test_texts[0], return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                _ = model(**inputs)
        
        # Benchmark
        total_time = 0
        num_runs = 100
        
        for _ in range(num_runs):
            for text in test_texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                start_time = time.time()
                with torch.no_grad():
                    _ = model(**inputs)
                total_time += time.time() - start_time
        
        avg_inference_time = total_time / (num_runs * len(test_texts))
        
        # Check if optimization stats exist
        optimization_stats = {}
        stats_file = os.path.join(model_path, "optimization_stats.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                optimization_stats = json.load(f)
        
        benchmark_results = {
            'model_size_mb': model_size,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'throughput_samples_per_sec': 1.0 / avg_inference_time,
            'optimization_stats': optimization_stats,
            'device': str(optimizer.device),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Model benchmark completed:")
        logger.info(f"  Size: {model_size:.1f}MB")
        logger.info(f"  Avg inference time: {avg_inference_time*1000:.2f}ms")
        logger.info(f"  Throughput: {1.0/avg_inference_time:.1f} samples/sec")
        
        return benchmark_results
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {'error': str(e)}

def optimize_existing_model(
    model_path: str = "final_roberta_persona",
    enable_pruning: bool = True,
    enable_quantization: bool = True,
    pruning_amount: float = 0.3,
    pruning_type: str = 'l1_unstructured'
) -> bool:
    """Apply optimizations to an existing trained model."""
    try:
        from transformers import RobertaForSequenceClassification, RobertaTokenizer
        
        logger.info(f"Optimizing existing model at {model_path}")
        
        # Load model
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        
        # Create optimization config
        config = create_optimized_config(
            enable_pruning=enable_pruning,
            enable_quantization=enable_quantization,
            pruning_amount=pruning_amount,
            pruning_type=pruning_type
        )
        
        # Apply optimizations
        optimizer = PerformanceOptimizer(config)
        optimized_model = optimizer.optimize_model_for_inference(model)
        
        # Save optimized model
        optimized_path = f"{model_path}_optimized"
        optimized_model.save_pretrained(optimized_path)
        tokenizer.save_pretrained(optimized_path)
        
        # Save optimization statistics
        optimization_stats = optimizer.get_optimization_stats(optimized_model)
        with open(f"{optimized_path}/optimization_stats.json", "w") as f:
            json.dump(optimization_stats, f, indent=2)
        
        logger.info(f"Model optimized and saved to {optimized_path}")
        logger.info(f"Size reduction: {optimization_stats.get('size_reduction_percent', 0):.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"Model optimization failed: {e}")
        return False

if __name__ == "__main__":
    # Allow command line arguments for fault tolerance, validation, and performance optimizations
    import sys
    enable_fault_tolerance = True
    enable_validation = True
    strict_validation = False
    performance_config = None
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--no-fault-tolerance":
            enable_fault_tolerance = False
        elif sys.argv[1] == "--no-validation":
            enable_validation = False
        elif sys.argv[1] == "--strict-validation":
            strict_validation = True
        elif sys.argv[1] == "--validate-only":
            # Run only validation without training
            result = validate_training_data_before_training(enable_validation=True, strict_mode=False)
            print(f"Validation completed. Valid: {result.is_valid}")
            if result.errors:
                print("Errors:", result.errors)
            if result.warnings:
                print("Warnings:", result.warnings)
            if result.recommendations:
                print("Recommendations:", result.recommendations)
            sys.exit(0)
        elif sys.argv[1] == "--status":
            status = get_training_status()
            print(json.dumps(status, indent=2))
            sys.exit(0)
        elif sys.argv[1] == "--cleanup":
            removed = cleanup_failed_checkpoints()
            print(f"Removed {removed} failed checkpoints")
            sys.exit(0)
        elif sys.argv[1] == "--benchmark":
            # Benchmark model performance
            results = benchmark_model_performance()
            print(json.dumps(results, indent=2))
            sys.exit(0)
        elif sys.argv[1] == "--optimize-model":
            # Optimize existing model
            success = optimize_existing_model()
            print(f"Model optimization: {'Success' if success else 'Failed'}")
            sys.exit(0)
        elif sys.argv[1] == "--performance-config":
            # Show current performance configuration
            config = get_performance_config()
            print(json.dumps(config, indent=2))
            sys.exit(0)
        elif sys.argv[1] == "--enable-pruning":
            # Enable pruning for training
            performance_config = create_optimized_config(enable_pruning=True)
        elif sys.argv[1] == "--enable-quantization":
            # Enable quantization for training
            performance_config = create_optimized_config(enable_quantization=True)
        elif sys.argv[1] == "--enable-all-optimizations":
            # Enable all optimizations
            performance_config = create_optimized_config(
                enable_pruning=True,
                enable_quantization=True
            )
        elif sys.argv[1] == "--help":
            print("Usage: python train_model.py [options]")
            print("Options:")
            print("  --no-fault-tolerance        Disable fault tolerance")
            print("  --no-validation             Disable data validation")
            print("  --strict-validation         Abort training if validation fails")
            print("  --validate-only             Run validation only, no training")
            print("  --status                    Show training status")
            print("  --cleanup                   Clean up failed checkpoints")
            print("  --benchmark                 Benchmark model performance")
            print("  --optimize-model            Optimize existing model")
            print("  --performance-config        Show performance configuration")
            print("  --enable-pruning            Enable model pruning during training")
            print("  --enable-quantization       Enable quantization during training")
            print("  --enable-all-optimizations  Enable all performance optimizations")
            print("  --help                      Show this help message")
            sys.exit(0)
    
    train_model_with_fault_tolerance(
        enable_fault_tolerance=enable_fault_tolerance,
        enable_validation=enable_validation,
        strict_validation=strict_validation,
        performance_config=performance_config
    )
