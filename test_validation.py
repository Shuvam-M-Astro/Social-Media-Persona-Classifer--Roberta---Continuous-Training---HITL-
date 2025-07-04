#!/usr/bin/env python3
"""
Test script for data validation functionality.
Run this to test the validation features without training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.logic.data_validation import validate_training_data, DataValidator, ValidationResult

def test_validation():
    """Test the validation functionality."""
    print("🧪 Testing Data Validation Functionality")
    print("=" * 50)
    
    # Test 1: Basic validation
    print("\n1. Running basic validation...")
    result = validate_training_data()
    
    print(f"✅ Validation completed: {'PASSED' if result.is_valid else 'FAILED'}")
    
    if result.errors:
        print("\n❌ Errors found:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print("\n⚠️  Warnings found:")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    if result.recommendations:
        print("\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")
    
    # Test 2: Custom validator
    print("\n2. Testing custom validator...")
    validator = DataValidator()
    
    # Load and validate main dataset
    try:
        import pandas as pd
        df = pd.read_csv("persona_dataset.csv")
        validation_result = validator.validate_dataset(df, "test_dataset")
        
        print(f"Custom validation: {'PASSED' if validation_result.is_valid else 'FAILED'}")
        print(f"Statistics: {len(validation_result.statistics)} metrics collected")
        print(f"Outliers detected: {len(validation_result.outliers)}")
        
    except Exception as e:
        print(f"Custom validation failed: {e}")
    
    # Test 3: Generate report
    print("\n3. Generating validation report...")
    try:
        report = validator.generate_validation_report("test_validation_report.txt")
        print("✅ Validation report generated: test_validation_report.txt")
        print("\nReport preview:")
        print("-" * 30)
        lines = report.split('\n')[:20]  # Show first 20 lines
        for line in lines:
            print(line)
        if len(report.split('\n')) > 20:
            print("... (truncated)")
    except Exception as e:
        print(f"Report generation failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Validation testing completed!")
    print("\nTo run validation with training:")
    print("  python src/logic/train_model.py --validate-only")
    print("\nTo run training with validation:")
    print("  python src/logic/train_model.py")
    print("\nTo run training without validation:")
    print("  python src/logic/train_model.py --no-validation")

if __name__ == "__main__":
    test_validation() 