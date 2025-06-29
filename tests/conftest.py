import pytest
import tempfile
import os
import shutil
import pandas as pd
import json
from unittest.mock import Mock, patch
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return pd.DataFrame({
        'bio': ['I love technology and coding', 'Food is life', 'Fitness is my passion'],
        'posts': ['Check out this new AI tool', 'Amazing restaurant experience', 'Great workout today'],
        'label': ['Tech Enthusiast', 'Foodie Explorer', 'Fitness Buff']
    })

@pytest.fixture
def sample_feedback_data():
    """Create sample feedback data for testing."""
    return pd.DataFrame({
        'bio': ['I love fashion and style'],
        'posts': ['New outfit looks amazing'],
        'label': ['Fashion Aficionado']
    })

@pytest.fixture
def mock_label_mappings():
    """Create mock label mappings for testing."""
    return {
        'label2id': {
            'Tech Enthusiast': 0,
            'Foodie Explorer': 1,
            'Fitness Buff': 2,
            'Fashion Aficionado': 3,
            'Meme Lord': 4
        },
        'id2label': {
            0: 'Tech Enthusiast',
            1: 'Foodie Explorer',
            2: 'Fitness Buff',
            3: 'Fashion Aficionado',
            4: 'Meme Lord'
        }
    }

@pytest.fixture
def mock_classifier():
    """Create a mock classifier for testing."""
    mock_clf = Mock()
    mock_clf.return_value = [{
        'label': 'Tech Enthusiast',
        'score': 0.8
    }, {
        'label': 'Foodie Explorer',
        'score': 0.15
    }, {
        'label': 'Fitness Buff',
        'score': 0.05
    }]
    return mock_clf

@pytest.fixture
def test_config():
    """Test configuration."""
    return {
        'model_dir': 'test_models',
        'data_dir': 'test_data',
        'max_length': 256,
        'batch_size': 16,
        'epochs': 2
    }

@pytest.fixture(autouse=True)
def setup_test_environment(temp_dir, monkeypatch):
    """Setup test environment variables and paths."""
    monkeypatch.setenv('TESTING', 'true')
    monkeypatch.setenv('MODEL_DIR', temp_dir)
    monkeypatch.setenv('DATA_DIR', temp_dir)
    
    # Create necessary directories
    os.makedirs(os.path.join(temp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'data'), exist_ok=True)
    
    yield temp_dir 