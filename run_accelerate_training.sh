#!/bin/bash

# DiffusionDet Training with Accelerate + DeepSpeed
# This script launches the training with the new accelerate-based trainer

# Configuration
CONFIG_FILE="configs/diffdet.tables.res50.cpu.yaml"
OUTPUT_DIR="./output_accelerate"
NUM_GPUS=1

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting DiffusionDet training with Accelerate + DeepSpeed..."
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"

# Launch training
if [ $NUM_GPUS -gt 1 ]; then
    # Multi-GPU training
    accelerate launch \
        --config_file configs/accelerate_config.yaml \
        train_accelerate.py \
        --config-file $CONFIG_FILE \
        --output-dir $OUTPUT_DIR \
        --use-deepspeed \
        --fp16 \
        --gradient-accumulation-steps 4 \
        --use-mlflow \
        --seed 42
else
    # Single GPU training
    python train_accelerate.py \
        --config-file $CONFIG_FILE \
        --output-dir $OUTPUT_DIR \
        --gradient-accumulation-steps 4 \
        --use-mlflow \
        --seed 42 \
        --device cpu
fi

echo "Training completed!"
