# Data Validation Features

This document explains how to use the new data validation features in the Social Media Persona Classifier.

## Overview

The data validation system provides comprehensive validation of training data before model training, including:

- **Schema validation**: Ensures data has correct structure and types
- **Data quality checks**: Detects missing values, duplicates, and data issues
- **Outlier detection**: Identifies unusual data points using statistical methods
- **Content analysis**: Analyzes text patterns and similarity
- **Label distribution analysis**: Checks for class imbalance and data diversity

## Quick Start

### 1. Test Validation Only

Run validation without training to check your data:

```bash
python test_validation.py
```

### 2. Run Training with Validation

Enable validation during training (default):

```bash
python src/logic/train_model.py
```

### 3. Run Training Without Validation

Disable validation if needed:

```bash
python src/logic/train_model.py --no-validation
```

### 4. Strict Validation Mode

Abort training if validation fails:

```bash
python src/logic/train_model.py --strict-validation
```

### 5. Validation Only Mode

Run validation and see results without training:

```bash
python src/logic/train_model.py --validate-only
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--no-validation` | Disable data validation |
| `--strict-validation` | Abort training if validation fails |
| `--validate-only` | Run validation only, no training |
| `--no-fault-tolerance` | Disable fault tolerance |
| `--status` | Show training status |
| `--cleanup` | Clean up failed checkpoints |
| `--help` | Show help message |

## Validation Features

### Schema Validation
- Checks for required columns (`bio`, `posts`, `label`)
- Validates data types
- Ensures proper structure

### Data Quality Checks
- **Missing values**: Detects and reports missing data
- **Text length**: Validates minimum/maximum text lengths
- **Duplicates**: Identifies duplicate entries
- **Empty content**: Finds empty or whitespace-only entries

### Outlier Detection
- **Text length outliers**: Uses IQR method to detect unusual text lengths
- **Multivariate outliers**: Uses Isolation Forest for complex outlier detection
- **Statistical analysis**: Provides detailed statistics about data distribution

### Content Analysis
- **Text uniqueness**: Checks for repetitive content
- **Emoji usage**: Analyzes special character patterns
- **Content similarity**: Detects potential data leakage between bio and posts
- **Language patterns**: Identifies unusual text patterns

### Label Distribution Analysis
- **Class imbalance**: Detects and reports imbalanced datasets
- **Sample counts**: Ensures sufficient samples per class
- **Diversity metrics**: Calculates entropy and diversity measures
- **Recommendations**: Suggests data augmentation strategies

## Output Files

When validation runs, it creates several output files:

```
validation_results/
├── validation_report.txt          # Detailed validation report
├── plots/
│   ├── main_dataset/
│   │   ├── label_distribution.png
│   │   └── text_length_analysis.png
│   └── feedback_dataset/
│       ├── label_distribution.png
│       └── text_length_analysis.png
```

## Integration with Training Pipeline

The validation is automatically integrated into the training pipeline:

1. **Pre-training validation**: Runs before training starts
2. **Configurable behavior**: Can be enabled/disabled via parameters
3. **Strict mode**: Can abort training on validation failures
4. **Logging**: All validation results are logged for debugging

## Configuration

Validation settings can be configured in the `DataSchema` class:

```python
schema = DataSchema(
    required_columns=["bio", "posts", "label"],
    text_min_length=10,
    text_max_length=10000,
    label_min_count=5,
    max_duplicates_ratio=0.1
)
```

## Example Usage in Code

```python
from src.logic.data_validation import validate_training_data, DataValidator

# Simple validation
result = validate_training_data()
if result.is_valid:
    print("✅ Data is valid for training")
else:
    print("❌ Data validation failed")
    for error in result.errors:
        print(f"  - {error}")

# Custom validation
validator = DataValidator()
df = pd.read_csv("persona_dataset.csv")
validation_result = validator.validate_dataset(df, "my_dataset")

# Generate report
report = validator.generate_validation_report("my_report.txt")
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Missing data files**: Ensure `persona_dataset.csv` exists in the project root

3. **Permission errors**: Check write permissions for output directories

4. **Memory issues**: For large datasets, validation may use significant memory

### Validation Warnings

Common warnings and their meanings:

- **Class imbalance**: Consider using class weights or oversampling
- **High emoji usage**: May indicate social media content (usually normal)
- **Text length outliers**: Check for unusually short or long entries
- **Low uniqueness**: May indicate repetitive or synthetic data

## Advanced Usage

### Custom Validation Rules

You can create custom validation rules by extending the `DataValidator` class:

```python
class CustomValidator(DataValidator):
    def _custom_validation(self, df):
        # Add your custom validation logic
        pass
```

### Batch Validation

For large datasets, you can validate in batches:

```python
validator = DataValidator()
for chunk in pd.read_csv("large_dataset.csv", chunksize=1000):
    result = validator.validate_dataset(chunk, "chunk")
    # Process results
```

## Performance Considerations

- **Large datasets**: Validation scales linearly with dataset size
- **Memory usage**: Isolation Forest outlier detection uses more memory
- **Processing time**: Content analysis can be slow for very large texts
- **Caching**: Results are not cached, so repeated validation re-runs all checks

## Contributing

To add new validation features:

1. Extend the `DataValidator` class
2. Add new validation methods
3. Update the `ValidationResult` dataclass if needed
4. Add tests for new functionality
5. Update this documentation 