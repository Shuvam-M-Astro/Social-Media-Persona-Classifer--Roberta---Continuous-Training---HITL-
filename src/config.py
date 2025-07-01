"""
Configuration management for the Persona Classifier application.

This module centralizes all configuration settings, environment variables,
and provides validation and default values for the application.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class FaultToleranceConfig:
    """Fault tolerance configuration."""
    # Training fault tolerance
    training_max_retries: int = 3
    training_retry_delay: int = 30
    training_timeout_minutes: int = 30
    enable_training_checkpoints: bool = True
    enable_training_backups: bool = True
    checkpoint_dir: str = "training_checkpoints"
    
    # Feedback fault tolerance
    feedback_queue_size: int = 1000
    feedback_batch_size: int = 50
    feedback_flush_interval: int = 300
    feedback_max_retries: int = 3
    enable_feedback_queue: bool = True
    enable_feedback_backup: bool = True
    
    # Prediction fault tolerance
    prediction_max_retries: int = 3
    prediction_retry_delay: int = 1
    prediction_timeout_seconds: int = 30
    enable_prediction_fallback: bool = True
    enable_prediction_cache: bool = True
    prediction_cache_size: int = 1000
    model_health_check_interval: int = 300

@dataclass
class ModelConfig:
    """Model-related configuration."""
    model_dir: str = "final_roberta_persona"
    base_model: str = "roberta-base"
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    test_size: float = 0.2
    min_samples_per_class: int = 1


@dataclass
class DataConfig:
    """Data-related configuration."""
    data_dir: str = "data"
    main_dataset: str = "persona_dataset.csv"
    feedback_dataset: str = "result.csv"
    processed_dataset: str = "processed_personas.csv"
    label_distribution: str = "label_distribution.csv"
    label2id_file: str = "label2id.json"
    id2label_file: str = "id2label.json"
    encoding: str = "utf-8"


@dataclass
class UIConfig:
    """UI-related configuration."""
    page_title: str = "Decode Your Digital Persona"
    layout: str = "centered"
    initial_page: str = "form"
    sidebar_state: str = "collapsed"
    debug_mode: bool = False
    show_debug_info: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "logs"
    log_file: str = "app.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_level: str = "WARNING"


@dataclass
class AppConfig:
    """Main application configuration."""
    environment: Environment = Environment.DEVELOPMENT
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)
    
    # Application paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    src_dir: Path = field(init=False)
    assets_dir: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize computed paths after object creation."""
        self.src_dir = self.base_dir / "src"
        self.assets_dir = self.base_dir / "assets"
        
        # Override with environment variables if present
        self._load_from_env()
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Environment
        env_str = os.getenv("APP_ENVIRONMENT", "development").lower()
        try:
            self.environment = Environment(env_str)
        except ValueError:
            logger.warning(f"Invalid environment '{env_str}', using development")
            self.environment = Environment.DEVELOPMENT
        
        # Model configuration
        self.model.model_dir = os.getenv("MODEL_DIR", self.model.model_dir)
        self.model.base_model = os.getenv("BASE_MODEL", self.model.base_model)
        self.model.max_length = int(os.getenv("MAX_LENGTH", self.model.max_length))
        self.model.batch_size = int(os.getenv("BATCH_SIZE", self.model.batch_size))
        self.model.epochs = int(os.getenv("EPOCHS", self.model.epochs))
        self.model.learning_rate = float(os.getenv("LEARNING_RATE", self.model.learning_rate))
        
        # Data configuration
        self.data.data_dir = os.getenv("DATA_DIR", self.data.data_dir)
        self.data.main_dataset = os.getenv("MAIN_DATASET", self.data.main_dataset)
        self.data.feedback_dataset = os.getenv("FEEDBACK_DATASET", self.data.feedback_dataset)
        
        # UI configuration
        self.ui.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.ui.show_debug_info = os.getenv("SHOW_DEBUG_INFO", "false").lower() == "true"
        
        # Logging configuration
        self.logging.level = os.getenv("LOG_LEVEL", self.logging.level)
        self.logging.log_dir = os.getenv("LOG_DIR", self.logging.log_dir)
        
        # Fault tolerance configuration
        self.fault_tolerance.training_max_retries = int(os.getenv("TRAINING_MAX_RETRIES", self.fault_tolerance.training_max_retries))
        self.fault_tolerance.training_retry_delay = int(os.getenv("TRAINING_RETRY_DELAY", self.fault_tolerance.training_retry_delay))
        self.fault_tolerance.enable_training_checkpoints = os.getenv("ENABLE_TRAINING_CHECKPOINTS", "true").lower() == "true"
        self.fault_tolerance.enable_training_backups = os.getenv("ENABLE_TRAINING_BACKUPS", "true").lower() == "true"
        
        self.fault_tolerance.feedback_queue_size = int(os.getenv("FEEDBACK_QUEUE_SIZE", self.fault_tolerance.feedback_queue_size))
        self.fault_tolerance.enable_feedback_queue = os.getenv("ENABLE_FEEDBACK_QUEUE", "true").lower() == "true"
        
        self.fault_tolerance.prediction_max_retries = int(os.getenv("PREDICTION_MAX_RETRIES", self.fault_tolerance.prediction_max_retries))
        self.fault_tolerance.enable_prediction_fallback = os.getenv("ENABLE_PREDICTION_FALLBACK", "true").lower() == "true"
        self.fault_tolerance.enable_prediction_cache = os.getenv("ENABLE_PREDICTION_CACHE", "true").lower() == "true"
    
    def get_model_path(self) -> Path:
        """Get the full path to the model directory."""
        return self.base_dir / self.model.model_dir
    
    def get_data_path(self) -> Path:
        """Get the full path to the data directory."""
        return self.base_dir / self.data.data_dir
    
    def get_log_path(self) -> Path:
        """Get the full path to the log directory."""
        return self.base_dir / self.logging.log_dir
    
    def get_dataset_path(self, dataset_name: str) -> Path:
        """Get the full path to a dataset file."""
        return self.base_dir / dataset_name
    
    def get_label_mappings(self) -> Dict[str, Any]:
        """Load and return label mappings."""
        label2id_path = self.base_dir / self.data.label2id_file
        id2label_path = self.base_dir / self.data.id2label_file
        
        mappings = {"label2id": {}, "id2label": {}}
        
        if label2id_path.exists():
            try:
                with open(label2id_path, 'r', encoding=self.data.encoding) as f:
                    mappings["label2id"] = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load label2id mapping: {e}")
        
        if id2label_path.exists():
            try:
                with open(id2label_path, 'r', encoding=self.data.encoding) as f:
                    raw_id2label = json.load(f)
                    mappings["id2label"] = {int(k): v for k, v in raw_id2label.items()}
            except Exception as e:
                logger.error(f"Failed to load id2label mapping: {e}")
        
        return mappings
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required directories
        required_dirs = [
            (self.get_data_path(), "Data directory"),
            (self.get_log_path(), "Log directory"),
            (self.assets_dir, "Assets directory")
        ]
        
        for path, name in required_dirs:
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created {name}: {path}")
                except Exception as e:
                    issues.append(f"Cannot create {name}: {e}")
        
        # Check model configuration
        if self.model.max_length <= 0:
            issues.append("max_length must be positive")
        
        if self.model.batch_size <= 0:
            issues.append("batch_size must be positive")
        
        if self.model.epochs <= 0:
            issues.append("epochs must be positive")
        
        if not 0 < self.model.test_size < 1:
            issues.append("test_size must be between 0 and 1")
        
        # Check data files
        main_dataset_path = self.get_dataset_path(self.data.main_dataset)
        if not main_dataset_path.exists():
            issues.append(f"Main dataset not found: {main_dataset_path}")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "environment": self.environment.value,
            "model": {
                "model_dir": self.model.model_dir,
                "base_model": self.model.base_model,
                "max_length": self.model.max_length,
                "batch_size": self.model.batch_size,
                "epochs": self.model.epochs,
                "learning_rate": self.model.learning_rate,
                "test_size": self.model.test_size
            },
            "data": {
                "data_dir": self.data.data_dir,
                "main_dataset": self.data.main_dataset,
                "feedback_dataset": self.data.feedback_dataset
            },
            "ui": {
                "page_title": self.ui.page_title,
                "debug_mode": self.ui.debug_mode,
                "show_debug_info": self.ui.show_debug_info
            },
            "logging": {
                "level": self.logging.level,
                "log_dir": self.logging.log_dir
            }
        }
    
    def save_config(self, path: Optional[Path] = None) -> bool:
        """Save current configuration to file."""
        if path is None:
            path = self.base_dir / "config.json"
        
        try:
            with open(path, 'w', encoding=self.data.encoding) as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    @classmethod
    def load_config(cls, path: Optional[Path] = None) -> 'AppConfig':
        """Load configuration from file."""
        if path is None:
            path = Path(__file__).parent.parent / "config.json"
        
        config = cls()
        
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Update configuration from file
                if "model" in data:
                    for key, value in data["model"].items():
                        if hasattr(config.model, key):
                            setattr(config.model, key, value)
                
                if "data" in data:
                    for key, value in data["data"].items():
                        if hasattr(config.data, key):
                            setattr(config.data, key, value)
                
                if "ui" in data:
                    for key, value in data["ui"].items():
                        if hasattr(config.ui, key):
                            setattr(config.ui, key, value)
                
                if "logging" in data:
                    for key, value in data["logging"].items():
                        if hasattr(config.logging, key):
                            setattr(config.logging, key, value)
                
                logger.info(f"Configuration loaded from {path}")
            except Exception as e:
                logger.error(f"Failed to load configuration from {path}: {e}")
        
        return config


# Global configuration instance
config = AppConfig.load_config()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> AppConfig:
    """Reload configuration from file and environment variables."""
    global config
    config = AppConfig.load_config()
    return config


def validate_config() -> bool:
    """Validate the current configuration and log any issues."""
    issues = config.validate()
    
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("Configuration validation passed")
    return True 