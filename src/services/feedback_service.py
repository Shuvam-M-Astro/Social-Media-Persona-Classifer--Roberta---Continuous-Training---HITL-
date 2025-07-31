import csv
import os
import subprocess
import logging
import json
import threading
import time
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Fault tolerance configuration for feedback
FEEDBACK_CONFIG = {
    'queue_size': 1000,
    'batch_size': 50,
    'flush_interval': 300,  # 5 minutes
    'max_retries': 3,
    'retry_delay': 5,
    'enable_queue': True,
    'backup_feedback': True,
    'feedback_backup_dir': 'feedback_backups'
}

@dataclass
class FeedbackItem:
    """Represents a feedback item with metadata."""
    bio: str
    posts: str
    corrected_label: str
    timestamp: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    retry_count: int = 0

class FeedbackQueue:
    """Thread-safe feedback queue with persistence."""
    def __init__(self, queue_size: int = 1000):
        self.queue = queue.Queue(maxsize=queue_size)
        self.lock = threading.Lock()
        self.is_running = True
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
    
    def add_feedback(self, feedback: FeedbackItem) -> bool:
        """Add feedback to queue."""
        try:
            self.queue.put(feedback, timeout=1)
            return True
        except queue.Full:
            logger.warning("Feedback queue is full, dropping feedback")
            return False
    
    def _flush_worker(self):
        """Background worker to flush feedback to disk."""
        while self.is_running:
            try:
                time.sleep(FEEDBACK_CONFIG['flush_interval'])
                self.flush_queue()
            except Exception as e:
                logger.error(f"Flush worker error: {e}")
    
    def flush_queue(self) -> int:
        """Flush all queued feedback to disk."""
        flushed_count = 0
        feedback_items = []
        
        # Collect all items from queue
        while not self.queue.empty():
            try:
                feedback_items.append(self.queue.get_nowait())
            except queue.Empty:
                break
        
        if feedback_items:
            # Save to feedback file
            success = save_feedback_batch(feedback_items)
            if success:
                flushed_count = len(feedback_items)
                logger.info(f"Flushed {flushed_count} feedback items")
            else:
                # Re-queue failed items
                for item in feedback_items:
                    item.retry_count += 1
                    if item.retry_count < FEEDBACK_CONFIG['max_retries']:
                        self.queue.put(item)
                    else:
                        logger.error(f"Feedback item exceeded max retries: {item}")
        
        return flushed_count
    
    def shutdown(self):
        """Shutdown the queue and flush remaining items."""
        self.is_running = False
        self.flush_queue()

# Global feedback queue
feedback_queue = FeedbackQueue(FEEDBACK_CONFIG['queue_size'])


def save_feedback_batch(feedback_items: List[FeedbackItem], path="result.csv") -> bool:
    """Save multiple feedback items in batch."""
    if not feedback_items:
        return True
    
    # Backup existing file if it exists
    if FEEDBACK_CONFIG['backup_feedback'] and os.path.exists(path):
        backup_feedback_file(path)
    
    try:
        file_exists = os.path.isfile(path)
        
        with open(path, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["bio", "posts", "label", "timestamp", "user_id", "session_id"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for item in feedback_items:
                row = {
                    "bio": item.bio,
                    "posts": item.posts,
                    "label": item.corrected_label,
                    "timestamp": item.timestamp,
                    "user_id": item.user_id or "",
                    "session_id": item.session_id or ""
                }
                writer.writerow(row)
        
        logger.info(f"Saved {len(feedback_items)} feedback items in batch.")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save feedback batch: {e}")
        return False

def save_feedback(bio: str, posts: str, corrected_label: str, path="result.csv", 
                 use_queue: bool = True, user_id: str = None, session_id: str = None):
    """Save feedback with fault tolerance options."""
    if use_queue and FEEDBACK_CONFIG['enable_queue']:
        # Add to queue for batch processing
        feedback_item = FeedbackItem(
            bio=bio,
            posts=posts,
            corrected_label=corrected_label,
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            session_id=session_id
        )
        
        success = feedback_queue.add_feedback(feedback_item)
        if success:
            logger.info("Feedback added to queue.")
            return True
        else:
            logger.warning("Queue full, falling back to direct save.")
    
    # Direct save (fallback or when queue is disabled)
    feedback_item = FeedbackItem(
        bio=bio,
        posts=posts,
        corrected_label=corrected_label,
        timestamp=datetime.now().isoformat(),
        user_id=user_id,
        session_id=session_id
    )
    
    return save_feedback_batch([feedback_item], path)

def backup_feedback_file(path: str):
    """Create backup of feedback file."""
    try:
        backup_dir = Path(FEEDBACK_CONFIG['feedback_backup_dir'])
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"feedback_backup_{timestamp}.csv"
        
        shutil.copy2(path, backup_path)
        logger.info(f"Created feedback backup: {backup_path}")
        
        # Clean old backups (keep last 10)
        cleanup_old_feedback_backups(backup_dir, keep_count=10)
        
    except Exception as e:
        logger.error(f"Failed to backup feedback file: {e}")

def cleanup_old_feedback_backups(backup_dir: Path, keep_count: int = 10):
    """Remove old feedback backups."""
    try:
        backups = sorted(backup_dir.glob("feedback_backup_*.csv"), 
                        key=lambda x: x.stat().st_mtime, reverse=True)
        
        for backup in backups[keep_count:]:
            os.remove(backup)
            logger.info(f"Removed old feedback backup: {backup}")
            
    except Exception as e:
        logger.error(f"Failed to cleanup feedback backups: {e}")

def get_feedback_stats() -> Dict[str, Any]:
    """Get feedback statistics."""
    stats = {
        'queue_size': feedback_queue.queue.qsize(),
        'queue_max_size': feedback_queue.queue.maxsize,
        'is_queue_running': feedback_queue.is_running
    }
    
    # Count feedback items in file
    try:
        if os.path.exists("result.csv"):
            with open("result.csv", 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                stats['total_feedback_items'] = sum(1 for _ in reader)
        else:
            stats['total_feedback_items'] = 0
    except Exception as e:
        logger.error(f"Failed to count feedback items: {e}")
        stats['total_feedback_items'] = 0
    
    return stats


def retrain_model(script_path=os.path.join("src", "logic", "train_model.py"), 
                 enable_fault_tolerance: bool = True,
                 enable_validation: bool = True,
                 strict_validation: bool = False):
    """Retrain model with fault tolerance and validation options."""
    try:
        # Flush any pending feedback before retraining
        if FEEDBACK_CONFIG['enable_queue']:
            flushed_count = feedback_queue.flush_queue()
            logger.info(f"Flushed {flushed_count} feedback items before retraining")
        
        # Build command with options
        cmd = ["python", script_path]
        
        # Add fault tolerance flag
        if enable_fault_tolerance:
            cmd.append("--fault-tolerance")
        else:
            cmd.append("--no-fault-tolerance")
        
        # Add validation flags
        if not enable_validation:
            cmd.append("--no-validation")
        elif strict_validation:
            cmd.append("--strict-validation")
        
        result = subprocess.run(
            cmd,
            capture_output=True, 
            text=True, 
            check=True,
            timeout=1800  # 30 minute timeout
        )
        logger.info("Model retrained successfully.")
        return True, result.stdout
        
    except subprocess.TimeoutExpired:
        error_msg = "Retraining timed out after 30 minutes"
        logger.error(error_msg)
        return False, error_msg
    except subprocess.CalledProcessError as e:
        logger.error(f"Retraining failed: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        logger.error(f"Retraining failed with exception: {e}")
        return False, str(e)

def shutdown_feedback_system():
    """Gracefully shutdown the feedback system."""
    if FEEDBACK_CONFIG['enable_queue']:
        feedback_queue.shutdown()
        logger.info("Feedback system shutdown complete")

# Add missing import
import shutil