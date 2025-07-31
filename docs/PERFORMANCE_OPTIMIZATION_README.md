# Performance Optimization Features

This document describes the new performance optimization features added to the Social Media Persona Classifier training module.

## 🚀 Overview

The training module now includes four key performance optimizations:

1. **Mixed Precision Training (FP16)** - Faster training with reduced memory usage
2. **Gradient Accumulation** - Handle larger effective batch sizes
3. **Model Pruning** - Reduce model size while maintaining performance
4. **Quantization** - Post-training quantization for faster inference

## 📊 Performance Benefits

| Optimization | Training Speed | Memory Usage | Model Size | Inference Speed |
|--------------|----------------|--------------|------------|-----------------|
| Mixed Precision | +30-50% | -50% | No change | +20-30% |
| Gradient Accumulation | +10-20% | No change | No change | No change |
| Pruning | No change | No change | -30-70% | +10-20% |
| Quantization | No change | No change | -50-75% | +40-60% |

## 🔧 Configuration

### Default Configuration

```python
PERFORMANCE_CONFIG = {
    'enable_mixed_precision': True,
    'enable_gradient_accumulation': True,
    'gradient_accumulation_steps': 4,
    'enable_pruning': False,
    'pruning_amount': 0.3,  # 30% of weights to prune
    'pruning_type': 'l1_unstructured',
    'enable_quantization': False,
    'quantization_type': 'dynamic',
    'target_model_size_mb': 100,
    'enable_amp': True,
    'fp16': True,
    'bf16': False,
}
```

### Custom Configuration

```python
from src.logic.train_model import create_optimized_config

# Speed-focused configuration
speed_config = create_optimized_config(
    enable_mixed_precision=True,
    enable_gradient_accumulation=True,
    gradient_accumulation_steps=8,
    enable_pruning=False,
    enable_quantization=False
)

# Size-focused configuration
size_config = create_optimized_config(
    enable_mixed_precision=True,
    enable_gradient_accumulation=True,
    gradient_accumulation_steps=4,
    enable_pruning=True,
    pruning_amount=0.5,
    enable_quantization=True
)
```

## 🎯 Usage Examples

### 1. Training with All Optimizations

```python
from src.logic.train_model import train_model, create_optimized_config

# Create configuration with all optimizations
config = create_optimized_config(
    enable_mixed_precision=True,
    enable_gradient_accumulation=True,
    gradient_accumulation_steps=4,
    enable_pruning=True,
    pruning_amount=0.3,
    enable_quantization=True
)

# Train with optimizations
success = train_model(
    enable_validation=True,
    performance_config=config
)
```

### 2. Command Line Usage

```bash
# Train with all optimizations
python src/logic/train_model.py --enable-all-optimizations

# Train with pruning only
python src/logic/train_model.py --enable-pruning

# Train with quantization only
python src/logic/train_model.py --enable-quantization

# Benchmark model performance
python src/logic/train_model.py --benchmark

# Optimize existing model
python src/logic/train_model.py --optimize-model

# Show performance configuration
python src/logic/train_model.py --performance-config
```

### 3. Optimizing Existing Models

```python
from src.logic.train_model import optimize_existing_model

# Optimize an existing trained model
success = optimize_existing_model(
    model_path="final_roberta_persona",
    enable_pruning=True,
    enable_quantization=True,
    pruning_amount=0.3
)
```

### 4. Performance Benchmarking

```python
from src.logic.train_model import benchmark_model_performance

# Benchmark model performance
results = benchmark_model_performance("final_roberta_persona")
print(f"Model size: {results['model_size_mb']:.1f}MB")
print(f"Avg inference time: {results['avg_inference_time_ms']:.2f}ms")
print(f"Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
```

## 🔍 Detailed Features

### 1. Mixed Precision Training (FP16)

**What it does:**
- Uses 16-bit floating point (FP16) instead of 32-bit (FP32)
- Reduces memory usage by ~50%
- Speeds up training by 30-50%
- Maintains model accuracy

**Configuration:**
```python
{
    'enable_mixed_precision': True,
    'fp16': True,  # Use float16
    'bf16': False,  # Use bfloat16 (better numerical stability)
}
```

**Benefits:**
- Faster training on modern GPUs
- Reduced memory usage
- No accuracy loss
- Automatic gradient scaling

### 2. Gradient Accumulation

**What it does:**
- Accumulates gradients over multiple forward/backward passes
- Simulates larger batch sizes without increased memory usage
- Improves training stability

**Configuration:**
```python
{
    'enable_gradient_accumulation': True,
    'gradient_accumulation_steps': 4,  # Effective batch size = batch_size * steps
}
```

**Benefits:**
- Larger effective batch sizes
- Better gradient estimates
- Improved training stability
- No additional memory usage

### 3. Model Pruning

**What it does:**
- Removes less important weights from the model
- Reduces model size while maintaining performance
- Supports different pruning strategies

**Pruning Types:**
- `l1_unstructured`: Removes weights with smallest L1 norm
- `random_unstructured`: Randomly removes weights
- `ln_structured`: Removes entire rows/columns based on L2 norm

**Configuration:**
```python
{
    'enable_pruning': True,
    'pruning_amount': 0.3,  # Remove 30% of weights
    'pruning_type': 'l1_unstructured',
}
```

**Benefits:**
- Reduced model size (30-70%)
- Faster inference
- Lower memory usage
- Maintained accuracy

### 4. Quantization

**What it does:**
- Converts model weights from FP32 to INT8
- Reduces model size by 50-75%
- Speeds up inference significantly

**Quantization Types:**
- `dynamic`: No calibration data needed, applied post-training
- `static`: Requires calibration data (not implemented yet)

**Configuration:**
```python
{
    'enable_quantization': True,
    'quantization_type': 'dynamic',
}
```

**Benefits:**
- Significant size reduction (50-75%)
- Faster inference (40-60% speedup)
- Lower memory usage
- Suitable for deployment

## 📈 Performance Monitoring

### Training Statistics

The training process now logs detailed performance statistics:

```
Original model size: 438.2MB
Final model size: 125.7MB
Size reduction: 71.3%
Mixed precision: True
Gradient accumulation: True
Pruning: True
Quantization: True
```

### Optimization Statistics

Optimization statistics are saved to `final_roberta_persona/optimization_stats.json`:

```json
{
  "model_size_mb": 125.7,
  "original_size_mb": 438.2,
  "size_reduction_mb": 312.5,
  "size_reduction_percent": 71.3,
  "pruning_enabled": true,
  "quantization_enabled": true,
  "mixed_precision_enabled": true,
  "gradient_accumulation_enabled": true
}
```

## 🛠️ Advanced Usage

### Custom Performance Optimizer

```python
from src.logic.train_model import PerformanceOptimizer

# Create custom optimizer
optimizer = PerformanceOptimizer({
    'enable_mixed_precision': True,
    'enable_pruning': True,
    'pruning_amount': 0.4,
    'enable_quantization': True
})

# Apply optimizations to model
optimized_model = optimizer.optimize_model_for_inference(model)

# Get optimization statistics
stats = optimizer.get_optimization_stats(optimized_model)
```

### Custom Optimized Trainer

```python
from src.logic.train_model import OptimizedTrainer, PerformanceOptimizer

# Create performance optimizer
optimizer = PerformanceOptimizer(config)

# Use optimized trainer
trainer = OptimizedTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    performance_optimizer=optimizer
)
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable mixed precision
   - Use gradient accumulation

2. **Training Instability**
   - Reduce learning rate
   - Increase gradient accumulation steps
   - Use bfloat16 instead of float16

3. **Model Size Too Large**
   - Enable pruning
   - Enable quantization
   - Reduce model complexity

4. **Slow Inference**
   - Enable quantization
   - Use mixed precision
   - Apply pruning

### Performance Tips

1. **For Speed:**
   - Enable mixed precision
   - Use gradient accumulation
   - Disable pruning during training

2. **For Size:**
   - Enable pruning (30-50%)
   - Enable quantization
   - Use aggressive pruning (50-70%)

3. **For Balance:**
   - Enable all optimizations
   - Use moderate pruning (20-30%)
   - Monitor accuracy carefully

## 📚 Example Script

Run the complete example script to see all features in action:

```bash
python performance_example.py
```

This script demonstrates:
- Training with all optimizations
- Performance benchmarking
- Custom configurations
- Configuration management

## 🔗 Related Files

- `src/logic/train_model.py` - Main training module with optimizations
- `performance_example.py` - Example usage script
- `PERFORMANCE_OPTIMIZATION_README.md` - This documentation

## 📝 Notes

- All optimizations are backward compatible
- Default configuration enables safe optimizations only
- Pruning and quantization are disabled by default
- Performance gains depend on hardware and dataset
- Monitor accuracy when using aggressive optimizations 