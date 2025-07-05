#!/usr/bin/env python3
"""
Example script demonstrating the new performance optimization features
for the Social Media Persona Classifier training module.

This script shows how to:
1. Use mixed precision training (FP16)
2. Enable gradient accumulation for larger effective batch sizes
3. Apply model pruning to reduce model size
4. Use quantization for faster inference
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'logic'))

from train_model import (
    train_model,
    create_optimized_config,
    benchmark_model_performance,
    optimize_existing_model,
    get_performance_config,
    update_performance_config
)

def example_1_basic_training_with_optimizations():
    """Example 1: Basic training with all optimizations enabled."""
    print("=== Example 1: Training with All Performance Optimizations ===")
    
    # Create a configuration with all optimizations enabled
    config = create_optimized_config(
        enable_mixed_precision=True,
        enable_gradient_accumulation=True,
        gradient_accumulation_steps=4,
        enable_pruning=True,
        pruning_amount=0.3,
        pruning_type='l1_unstructured',
        enable_quantization=True,
        quantization_type='dynamic'
    )
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train the model with optimizations
    print("\nStarting training with optimizations...")
    success = train_model(
        enable_validation=True,
        strict_validation=False,
        performance_config=config
    )
    
    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")
    
    return success

def example_2_benchmark_comparison():
    """Example 2: Compare performance before and after optimizations."""
    print("\n=== Example 2: Performance Benchmark Comparison ===")
    
    # Benchmark original model (if it exists)
    if os.path.exists("final_roberta_persona"):
        print("Benchmarking original model...")
        original_benchmark = benchmark_model_performance("final_roberta_persona")
        print(f"Original model performance: {original_benchmark}")
    
    # Optimize existing model
    print("\nOptimizing existing model...")
    success = optimize_existing_model(
        model_path="final_roberta_persona",
        enable_pruning=True,
        enable_quantization=True,
        pruning_amount=0.3
    )
    
    if success:
        # Benchmark optimized model
        print("Benchmarking optimized model...")
        optimized_benchmark = benchmark_model_performance("final_roberta_persona_optimized")
        print(f"Optimized model performance: {optimized_benchmark}")
        
        # Compare results
        if 'original_benchmark' in locals() and 'optimized_benchmark' in locals():
            print("\nPerformance Comparison:")
            if 'model_size_mb' in original_benchmark and 'model_size_mb' in optimized_benchmark:
                size_reduction = (original_benchmark['model_size_mb'] - optimized_benchmark['model_size_mb']) / original_benchmark['model_size_mb'] * 100
                print(f"  Size reduction: {size_reduction:.1f}%")
            
            if 'avg_inference_time_ms' in original_benchmark and 'avg_inference_time_ms' in optimized_benchmark:
                speed_improvement = (original_benchmark['avg_inference_time_ms'] - optimized_benchmark['avg_inference_time_ms']) / original_benchmark['avg_inference_time_ms'] * 100
                print(f"  Speed improvement: {speed_improvement:.1f}%")

def example_3_custom_configurations():
    """Example 3: Different optimization configurations for different use cases."""
    print("\n=== Example 3: Custom Optimization Configurations ===")
    
    # Configuration for speed-focused training
    speed_config = create_optimized_config(
        enable_mixed_precision=True,
        enable_gradient_accumulation=True,
        gradient_accumulation_steps=8,  # Larger effective batch size
        enable_pruning=False,  # No pruning for speed
        enable_quantization=False  # No quantization for speed
    )
    print("Speed-focused configuration:")
    print(f"  Mixed precision: {speed_config['enable_mixed_precision']}")
    print(f"  Gradient accumulation steps: {speed_config['gradient_accumulation_steps']}")
    print(f"  Pruning: {speed_config['enable_pruning']}")
    print(f"  Quantization: {speed_config['enable_quantization']}")
    
    # Configuration for size-focused optimization
    size_config = create_optimized_config(
        enable_mixed_precision=True,
        enable_gradient_accumulation=True,
        gradient_accumulation_steps=4,
        enable_pruning=True,
        pruning_amount=0.5,  # More aggressive pruning
        pruning_type='l1_unstructured',
        enable_quantization=True,
        quantization_type='dynamic'
    )
    print("\nSize-focused configuration:")
    print(f"  Pruning amount: {size_config['pruning_amount']}")
    print(f"  Pruning type: {size_config['pruning_type']}")
    print(f"  Quantization: {size_config['enable_quantization']}")
    
    # Configuration for balanced approach
    balanced_config = create_optimized_config(
        enable_mixed_precision=True,
        enable_gradient_accumulation=True,
        gradient_accumulation_steps=4,
        enable_pruning=True,
        pruning_amount=0.2,  # Moderate pruning
        enable_quantization=True,
        quantization_type='dynamic'
    )
    print("\nBalanced configuration:")
    print(f"  Pruning amount: {balanced_config['pruning_amount']}")
    print(f"  All optimizations enabled: {balanced_config['enable_mixed_precision'] and balanced_config['enable_pruning'] and balanced_config['enable_quantization']}")

def example_4_performance_monitoring():
    """Example 4: Monitor and update performance configuration."""
    print("\n=== Example 4: Performance Configuration Management ===")
    
    # Get current configuration
    current_config = get_performance_config()
    print("Current performance configuration:")
    for key, value in current_config.items():
        print(f"  {key}: {value}")
    
    # Update configuration
    print("\nUpdating configuration...")
    update_performance_config({
        'gradient_accumulation_steps': 6,
        'pruning_amount': 0.4,
        'enable_bf16': True  # Use bfloat16 if available
    })
    
    # Get updated configuration
    updated_config = get_performance_config()
    print("Updated performance configuration:")
    for key, value in updated_config.items():
        print(f"  {key}: {value}")

def main():
    """Main function to run all examples."""
    print("🚀 Performance Optimization Examples for Social Media Persona Classifier")
    print("=" * 70)
    
    try:
        # Example 1: Basic training with optimizations
        example_1_basic_training_with_optimizations()
        
        # Example 2: Benchmark comparison
        example_2_benchmark_comparison()
        
        # Example 3: Custom configurations
        example_3_custom_configurations()
        
        # Example 4: Performance monitoring
        example_4_performance_monitoring()
        
        print("\n✅ All examples completed successfully!")
        print("\nTo run specific optimizations from command line:")
        print("  python src/logic/train_model.py --enable-all-optimizations")
        print("  python src/logic/train_model.py --benchmark")
        print("  python src/logic/train_model.py --optimize-model")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 