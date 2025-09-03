# DiffusionDet Training with Hugging Face Accelerate + DeepSpeed

This implementation provides an alternative training pipeline for DiffusionDet using Hugging Face Accelerate and DeepSpeed instead of the default Detectron2 training framework.

## Features

- **Same LoRA Configuration**: Uses identical LoRA target modules and configuration as the original `train_net.py`
- **Hugging Face Accelerate**: Modern distributed training framework with better usability
- **DeepSpeed Integration**: Memory-efficient training with ZeRO optimization
- **MLflow Logging**: Same experiment tracking as the original implementation
- **Multi-GPU Support**: Easy scaling to multiple GPUs
- **Mixed Precision**: FP16 training for faster performance
- **Progress Tracking**: Beautiful tqdm progress bars with real-time metrics and ETA
- **Enhanced Logging**: Comprehensive timing and performance monitoring

## Installation

Install additional dependencies:

```bash
pip install -r requirements_accelerate.txt
```

## Usage

### Basic Training

```bash
python train_accelerate.py \
    --config-file configs/diffdet.tables.res50.cpu.yaml \
    --output-dir ./output_accelerate \
    --gradient-accumulation-steps 4 \
    --use-mlflow \
    --seed 42
```

### Training with DeepSpeed

```bash
python train_accelerate.py \
    --config-file configs/diffdet.tables.res50.cpu.yaml \
    --output-dir ./output_accelerate \
    --use-deepspeed \
    --fp16 \
    --gradient-accumulation-steps 4 \
    --use-mlflow \
    --seed 42
```

### Multi-GPU Training

```bash
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    train_accelerate.py \
    --config-file configs/diffdet.tables.res50.cpu.yaml \
    --output-dir ./output_accelerate \
    --use-deepspeed \
    --fp16 \
    --gradient-accumulation-steps 4 \
    --use-mlflow \
    --seed 42
```

### Using the Launch Script

```bash
./run_accelerate_training.sh
```

## Configuration Files

- `configs/deepspeed_config.json`: DeepSpeed ZeRO stage 2 configuration
- `configs/accelerate_config.yaml`: Accelerate configuration for distributed training
- `configs/diffdet.tables.res50.cpu.yaml`: Model and training configuration (same as original)

## Key Differences from Original

### Advantages
1. **Better Memory Management**: DeepSpeed ZeRO reduces memory usage
2. **Easier Multi-GPU Setup**: Accelerate handles distributed training complexity
3. **Modern Framework**: Built on HuggingFace ecosystem
4. **Flexible Deployment**: Easy to scale from single GPU to multi-node

### LoRA Configuration
The LoRA configuration is identical to the original:

```python
target_modules = [
    "head.head_series.*.self_attn.out_proj",
    "head.head_series.*.self_attn",
    "head.head_series.*.inst_interact.dynamic_layer",
    "head.head_series.*.inst_interact.out_layer",
    "head.head_series.{0-5}.class_logits",
    "head.head_series.{0-5}.bboxes_delta",
    "head.time_mlp.1",
    "head.time_mlp.3",
    "head.head_series.*.block_time_mlp.1",
]

modules_to_save = [
    "head.head_series.*.class_logits",
    "head.time_mlp.0",
]
```

## Command Line Arguments

- `--config-file`: Path to Detectron2 config file (required)
- `--output-dir`: Output directory for checkpoints and logs
- `--device`: Force specific device (cuda/cpu/mps)
- `--seed`: Random seed for reproducibility
- `--gradient-accumulation-steps`: Number of steps to accumulate gradients
- `--fp16`: Enable mixed precision training
- `--use-deepspeed`: Enable DeepSpeed optimization
- `--use-mlflow`: Enable MLflow experiment tracking
- `--opts`: Additional config overrides (same format as Detectron2)

## Output Structure

```
output_accelerate/
├── checkpoint-1000/          # Model checkpoints
├── checkpoint-2000/
├── logs/                     # Training logs
└── accelerate_state/         # Accelerate state for resuming
```

## Performance Notes

1. **Memory Usage**: DeepSpeed ZeRO stage 2 can reduce memory usage by ~40%
2. **Training Speed**: Comparable to original with potential improvements on multi-GPU
3. **Convergence**: Same convergence characteristics as original training

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: 
   - Reduce batch size in config
   - Increase gradient accumulation steps
   - Enable DeepSpeed with `--use-deepspeed`

2. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements_accelerate.txt`
   - Check Detectron2 installation

3. **Multi-GPU Issues**:
   - Verify Accelerate configuration: `accelerate config`
   - Check CUDA/NCCL setup

### Environment Variables

Set these for optimal performance:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify GPUs
export NCCL_DEBUG=INFO              # Debug distributed training
export TOKENIZERS_PARALLELISM=false # Avoid tokenizer warnings
```

## Comparison with Original

| Feature | Original (train_net.py) | Accelerate (train_accelerate.py) |
|---------|------------------------|-----------------------------------|
| Framework | Detectron2 | HuggingFace Accelerate |
| LoRA Config | ✅ Same | ✅ Same |
| Multi-GPU | DDP | Accelerate + DeepSpeed |
| Memory Opt | Limited | DeepSpeed ZeRO |
| MLflow | ✅ | ✅ |
| Ease of Use | Complex setup | Simple launch |
| Checkpointing | Detectron2 format | HuggingFace format |
