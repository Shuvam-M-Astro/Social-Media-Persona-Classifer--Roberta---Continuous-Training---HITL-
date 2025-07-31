# Project Structure Documentation

## Overview

This document provides a detailed explanation of the Social Media Persona Classifier project structure, explaining the purpose and organization of each directory and file.

## Root Directory Structure

### Core Application Files

- **`app.py`**: Main application entry point that initializes and runs the Streamlit application
- **`requirements.txt`**: Python package dependencies with version specifications
- **`Dockerfile`**: Container configuration for Docker deployment
- **`pytest.ini`**: Pytest configuration for test execution
- **`.gitattributes`**: Git attributes for consistent line endings and file handling

### Configuration Management

```
config/
├── config.example.json    # Example configuration file
└── config.json           # Actual configuration (created from example)
```

The `config/` directory contains all application configuration files. The example file provides a template that users can copy and customize for their specific environment.

### Data Management

```
data/
├── raw/                  # Raw, unprocessed data files
│   └── persona_dataset.csv
└── processed/            # Processed and transformed data
    ├── label_distribution.csv
    ├── label2id.json
    └── id2label.json
```

- **`raw/`**: Contains original, unmodified data files
- **`processed/`**: Contains data that has been cleaned, transformed, or processed for model training

### Documentation

```
docs/
├── PERFORMANCE_OPTIMIZATION_README.md
├── VALIDATION_README.md
└── PROJECT_STRUCTURE.md
```

Comprehensive documentation covering performance optimization strategies, validation procedures, and project structure.

### Models

```
models/
└── final_roberta_persona/  # Trained model files
```

Directory for storing trained models and model artifacts. The application expects the trained RoBERTa model to be placed in the `final_roberta_persona/` subdirectory.

### Scripts

```
scripts/
├── performance_example.py   # Performance testing examples
├── test_validation.py      # Validation testing utilities
└── test_runner.py          # Test execution script
```

Utility scripts for testing, validation, and performance analysis.

### Assets

```
assets/
├── images/                 # Static image assets
│   ├── background.jpg
│   ├── default.jpg
│   ├── demo_image.png
│   ├── fashion.jpg
│   ├── fitness.jpg
│   ├── food.jpg
│   ├── meme.jpg
│   └── tech.jpg
└── videos/                 # Video assets
    └── background.mp4
```

Static assets used by the application UI, organized by type.

## Source Code Structure (`src/`)

### Core Module (`src/core/`)

```
src/core/
├── __init__.py
└── settings.py             # Application settings and configuration
```

Contains the core application logic and configuration management.

### Services Module (`src/services/`)

```
src/services/
├── __init__.py
├── prediction_service.py   # Text classification and prediction logic
├── model_training_service.py # Model training and retraining
└── feedback_service.py     # User feedback processing
```

Business logic services that handle the main application functionality.

### Utils Module (`src/utils/`)

```
src/utils/
├── __init__.py
└── data_validator.py       # Data validation and preprocessing utilities
```

Utility functions and helper classes used across the application.

### Models Module (`src/models/`)

```
src/models/
└── __init__.py
```

Data models and model-related utilities (currently empty, ready for future expansion).

### UI Module (`src/ui/`)

```
src/ui/
├── app_pages.py            # Streamlit page definitions and UI logic
└── theme_manager.py        # UI theme and styling management
```

User interface components and styling.

## Test Structure (`tests/`)

```
tests/
├── __init__.py
├── conftest.py             # Pytest configuration and shared fixtures
├── README.md               # Test documentation
├── e2e/                    # End-to-end tests
│   └── test_app_workflow.py
├── integration/            # Integration tests
│   └── test_model_training.py
└── unit/                   # Unit tests
    ├── test_feedback.py
    ├── test_predict.py
    └── test_ui_components.py
```

Comprehensive test suite with different levels of testing:
- **Unit Tests**: Test individual functions and components
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete user workflows

## Key Design Principles

### 1. Separation of Concerns
- **Services**: Business logic separated from UI
- **Utils**: Reusable utility functions
- **Core**: Application configuration and settings
- **UI**: Presentation layer isolated from business logic

### 2. Modularity
- Each module has a specific responsibility
- Clear interfaces between modules
- Easy to test individual components

### 3. Configuration Management
- Centralized configuration in `config/` directory
- Environment-specific settings
- Easy deployment across different environments

### 4. Data Organization
- Clear separation between raw and processed data
- Consistent naming conventions
- Version control for data files

### 5. Testing Strategy
- Multiple levels of testing (unit, integration, e2e)
- Comprehensive test coverage
- Automated test execution

## File Naming Conventions

- **Python files**: Use snake_case (e.g., `prediction_service.py`)
- **Directories**: Use snake_case (e.g., `model_training_service.py`)
- **Configuration files**: Use descriptive names (e.g., `config.example.json`)
- **Test files**: Prefix with `test_` (e.g., `test_predict.py`)

## Import Structure

The application uses relative imports within the `src/` package:

```python
# Example import from services
from src.services.prediction_service import predict_persona

# Example import from utils
from src.utils.data_validator import validate_input

# Example import from core
from src.core.settings import get_config
```

## Deployment Considerations

### Docker Support
- `Dockerfile` provides containerized deployment
- Environment variables for configuration
- Optimized for production deployment

### Configuration Management
- Example configuration file provided
- Environment-specific settings
- Secure credential management

### Monitoring and Logging
- Comprehensive logging throughout the application
- Performance monitoring capabilities
- Error tracking and reporting

## Future Enhancements

The current structure supports future enhancements:

1. **API Layer**: Easy to add REST API endpoints
2. **Database Integration**: Ready for database models
3. **Microservices**: Services can be easily separated
4. **Cloud Deployment**: Structure supports cloud-native deployment
5. **ML Pipeline**: Extensible for advanced ML workflows

This structure provides a solid foundation for a professional-grade machine learning application with clear separation of concerns, comprehensive testing, and production-ready features. 