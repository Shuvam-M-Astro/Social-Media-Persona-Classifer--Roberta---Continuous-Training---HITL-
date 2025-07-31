import streamlit as st
import sys
import logging
import os
from pathlib import Path
import traceback
from typing import Optional

# Allow imports from src/
sys.path.append(os.path.abspath("src"))

# Patch torch for Streamlit on Windows if needed
try:
    sys.modules['torch.classes'] = None
except Exception as e:
    pass

# Import configuration
from src.config import get_config, validate_config, reload_config

# Get configuration
config = get_config()

# Configure structured logging
def setup_logging() -> None:
    """Configure logging with proper formatting and file output."""
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(config.logging.format)
    
    # File handler
    file_handler = logging.FileHandler(log_dir / config.logging.log_file)
    file_handler.setLevel(getattr(logging, config.logging.level))
    file_handler.setFormatter(formatter)
    
    # Stream handler for console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, config.logging.console_level))
    stream_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

# Setup logging
logger = setup_logging()

def initialize_session_state() -> None:
    """Initialize Streamlit session state with default values."""
    defaults = {
        "page": config.ui.initial_page,
        "result": {},
        "show_retrain": False,
        "error_message": None,
        "app_initialized": False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def setup_fault_tolerance() -> None:
    """Setup fault tolerance systems."""
    try:
        # Initialize fault tolerance configurations
        from src.logic.train_model import FAULT_TOLERANCE_CONFIG
        from src.logic.feedback import FEEDBACK_CONFIG
        from src.logic.predict import PREDICTION_CONFIG
        
        # Update configurations from app config
        FAULT_TOLERANCE_CONFIG.update({
            'max_retries': config.fault_tolerance.training_max_retries,
            'retry_delay': config.fault_tolerance.training_retry_delay,
            'enable_checkpointing': config.fault_tolerance.enable_training_checkpoints,
            'enable_backup': config.fault_tolerance.enable_training_backups,
            'checkpoint_dir': config.fault_tolerance.checkpoint_dir
        })
        
        FEEDBACK_CONFIG.update({
            'queue_size': config.fault_tolerance.feedback_queue_size,
            'enable_queue': config.fault_tolerance.enable_feedback_queue,
            'backup_feedback': config.fault_tolerance.enable_feedback_backup
        })
        
        PREDICTION_CONFIG.update({
            'max_retries': config.fault_tolerance.prediction_max_retries,
            'enable_fallback': config.fault_tolerance.enable_prediction_fallback,
            'cache_predictions': config.fault_tolerance.enable_prediction_cache,
            'cache_size': config.fault_tolerance.prediction_cache_size
        })
        
        logger.info("Fault tolerance systems initialized")
        
    except Exception as e:
        logger.error(f"Failed to setup fault tolerance: {e}")

def cleanup_on_shutdown() -> None:
    """Cleanup resources on application shutdown."""
    try:
        from src.logic.feedback import shutdown_feedback_system
        shutdown_feedback_system()
        logger.info("Application shutdown cleanup completed")
    except Exception as e:
        logger.error(f"Error during shutdown cleanup: {e}")

# Register cleanup handler
import atexit
atexit.register(cleanup_on_shutdown)

def handle_error(error: Exception, context: str = "Unknown") -> None:
    """Handle and log errors gracefully."""
    error_msg = f"Error in {context}: {str(error)}"
    logger.error(error_msg, exc_info=True)
    
    # Store error in session state for user display
    st.session_state.error_message = {
        "message": "An unexpected error occurred. Please try again.",
        "details": error_msg if config.ui.debug_mode else None
    }

def display_error_page() -> None:
    """Display error page when something goes wrong."""
    st.error("🚨 Something went wrong!")
    
    if st.session_state.error_message:
        st.error(st.session_state.error_message["message"])
        
        if config.ui.debug_mode and st.session_state.error_message["details"]:
            with st.expander("Debug Information"):
                st.code(st.session_state.error_message["details"])
    
    if st.button("🔄 Restart Application"):
        # Clear session state and restart
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

def load_page_modules() -> Optional[dict]:
    """Safely load page modules with error handling."""
    try:
        from src.ui.pages import (
            page_form,
            page_confirm,
            page_result,
            page_feedback
        )
        return {
            "form": page_form,
            "confirm": page_confirm,
            "result": page_result,
            "feedback": page_feedback
        }
    except ImportError as e:
        handle_error(e, "Loading page modules")
        return None
    except Exception as e:
        handle_error(e, "Unexpected error loading modules")
        return None

def main() -> None:
    """Main application entry point with comprehensive error handling."""
    try:
        # Validate configuration
        if not validate_config():
            st.error("❌ Application configuration is invalid. Please check your setup.")
            return
        
        # Set page configuration
        st.set_page_config(
            page_title=config.ui.page_title,
            layout=config.ui.layout,
            initial_sidebar_state=config.ui.sidebar_state
        )
        
        # Initialize session state
        initialize_session_state()
        setup_fault_tolerance()
        
        # Load page modules
        pages = load_page_modules()
        if not pages:
            display_error_page()
            return
        
        # Log application startup
        logger.info(f"Application started successfully in {config.environment.value} environment")
        
        # Page routing with error handling
        current_page = st.session_state.page
        
        if current_page not in pages:
            logger.error(f"Unknown page state: {current_page}")
            st.error("Unknown page state. Please restart the app.")
            return
        
        try:
            # Execute the current page
            pages[current_page]()
            
            # Clear any previous error messages on successful page load
            if st.session_state.error_message:
                st.session_state.error_message = None
                
        except Exception as e:
            handle_error(e, f"Page execution: {current_page}")
            display_error_page()
            
    except Exception as e:
        # Catch any unexpected errors in the main function
        handle_error(e, "Main application")
        st.error("🚨 Critical application error. Please restart the app.")
        
        if config.ui.debug_mode:
            with st.expander("Debug Information"):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()