# Testing Documentation

This directory contains comprehensive tests for the Persona Classifier project.

## 📁 Test Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests
│   ├── test_predict.py      # Tests for prediction logic
│   ├── test_feedback.py     # Tests for feedback handling
│   └── test_ui_components.py # Tests for UI components
├── integration/             # Integration tests
│   └── test_model_training.py # Tests for model training workflow
├── e2e/                     # End-to-end tests
│   └── test_app_workflow.py # Tests for complete app workflow
└── README.md               # This file
```

## 🏷️ Test Categories

### Unit Tests (`@pytest.mark.unit`)
- **Purpose**: Test individual functions and components in isolation
- **Scope**: Single function or class
- **Speed**: Fast (< 1 second each)
- **Dependencies**: Mocked external dependencies

### Integration Tests (`@pytest.mark.integration`)
- **Purpose**: Test interactions between components
- **Scope**: Multiple functions or modules working together
- **Speed**: Medium (1-10 seconds each)
- **Dependencies**: Some real dependencies, some mocked

### End-to-End Tests (`@pytest.mark.e2e`)
- **Purpose**: Test complete user workflows
- **Scope**: Entire application flow
- **Speed**: Slow (10+ seconds each)
- **Dependencies**: Mostly mocked for UI components

### Slow Tests (`@pytest.mark.slow`)
- **Purpose**: Tests that take significant time
- **Examples**: Model training, large data processing
- **Speed**: Very slow (30+ seconds each)

## 🚀 Running Tests

### Quick Start
```bash
# Run all tests
python run_tests.py

# Run only unit tests
python run_tests.py --type unit

# Run tests without coverage
python run_tests.py --no-coverage

# Run all checks (tests + linting + type checking)
python run_tests.py --all-checks
```

### Using pytest directly
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/ -m unit
pytest tests/ -m integration
pytest tests/ -m e2e

# Run fast tests only (exclude slow tests)
pytest tests/ -m "not slow"

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_predict.py

# Run specific test function
pytest tests/unit/test_predict.py::TestPredict::test_predict_persona_success
```

## 📊 Test Coverage

The test suite aims for comprehensive coverage:

- **Unit Tests**: 90%+ coverage of core logic
- **Integration Tests**: Key workflow paths
- **E2E Tests**: Critical user journeys

### Coverage Reports
After running tests with coverage, you can find:
- **Terminal output**: Shows missing lines
- **HTML report**: `htmlcov/index.html` - Interactive coverage report
- **XML report**: `coverage.xml` - For CI/CD integration

## 🧪 Test Fixtures

### Common Fixtures (in `conftest.py`)

#### `temp_dir`
- **Purpose**: Creates temporary directory for test files
- **Cleanup**: Automatically removed after test
- **Usage**: File operations, model saving, etc.

#### `sample_dataset`
- **Purpose**: Provides sample training data
- **Content**: 3 sample records with bio, posts, and labels
- **Usage**: Model training tests, data processing tests

#### `mock_classifier`
- **Purpose**: Mock classifier for prediction tests
- **Returns**: Fixed prediction results
- **Usage**: Unit tests for prediction logic

#### `mock_label_mappings`
- **Purpose**: Sample label mappings
- **Content**: 5 persona types with ID mappings
- **Usage**: Label processing tests

## 🔧 Test Configuration

### pytest.ini
- **Test discovery**: `tests/` directory
- **Markers**: Defined for test categories
- **Coverage**: Enabled by default
- **Output**: Verbose with short tracebacks

### Environment Variables
Tests automatically set:
- `TESTING=true`
- `MODEL_DIR=<temp_dir>`
- `DATA_DIR=<temp_dir>`

## 📝 Writing New Tests

### Unit Test Template
```python
import pytest
from unittest.mock import Mock, patch

class TestYourFunction:
    """Test cases for your_function."""
    
    @pytest.mark.unit
    def test_your_function_success(self, mock_dependency):
        """Test successful execution."""
        with patch('module.dependency', mock_dependency):
            result = your_function("test_input")
            assert result == expected_output
    
    @pytest.mark.unit
    def test_your_function_error(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="expected error message"):
            your_function("invalid_input")
```

### Integration Test Template
```python
import pytest

class TestYourWorkflow:
    """Integration tests for your workflow."""
    
    @pytest.mark.integration
    def test_workflow_end_to_end(self, temp_dir, sample_data):
        """Test complete workflow."""
        # Setup
        setup_test_environment(temp_dir, sample_data)
        
        # Execute
        result = run_workflow()
        
        # Verify
        assert result.success
        assert result.output == expected_output
```

### E2E Test Template
```python
import pytest

class TestUserJourney:
    """End-to-end tests for user journeys."""
    
    @pytest.mark.e2e
    def test_complete_user_journey(self, mock_ui_components):
        """Test complete user journey."""
        # Setup mocks
        with patch('ui.components', mock_ui_components):
            # Simulate user actions
            user_input = "test input"
            result = simulate_user_journey(user_input)
            
            # Verify outcomes
            assert result.final_state == expected_state
```

## 🐛 Debugging Tests

### Common Issues

#### Import Errors
```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

#### Missing Dependencies
```bash
# Install test dependencies
pip install -r requirements.txt
```

#### Test Isolation
- Each test runs in isolation
- Use `temp_dir` fixture for file operations
- Mock external dependencies

### Debug Mode
```bash
# Run single test with debug output
pytest tests/unit/test_predict.py::TestPredict::test_predict_persona_success -v -s

# Run with pdb on failure
pytest tests/ --pdb

# Run with detailed output
pytest tests/ -vvv
```

## 🔄 Continuous Integration

### GitHub Actions (Recommended)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python run_tests.py --all-checks
```

### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: python run_tests.py --type unit
        language: system
        pass_filenames: false
```

## 📈 Test Metrics

### Coverage Goals
- **Overall Coverage**: > 85%
- **Critical Paths**: > 95%
- **New Features**: > 90%

### Performance Goals
- **Unit Tests**: < 30 seconds total
- **Integration Tests**: < 2 minutes total
- **E2E Tests**: < 5 minutes total

## 🤝 Contributing

### Adding New Tests
1. **Identify test category** (unit/integration/e2e)
2. **Create test file** in appropriate directory
3. **Write test cases** following templates
4. **Add fixtures** if needed
5. **Update documentation** if adding new test types

### Test Naming Conventions
- **Files**: `test_<module_name>.py`
- **Classes**: `Test<ClassName>`
- **Methods**: `test_<function_name>_<scenario>`

### Best Practices
- **One assertion per test** when possible
- **Descriptive test names** that explain the scenario
- **Proper mocking** of external dependencies
- **Clean setup/teardown** using fixtures
- **Test isolation** - no shared state between tests 