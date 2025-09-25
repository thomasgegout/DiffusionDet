#!/bin/bash

# DiffusionDet Training with DeepSpeed
# Simplified script for training with Hugging Face Accelerate and DeepSpeed

# Activate virtual environment
source .venv/bin/activate

# Configuration
CONFIG_FILE="configs/diffdet.tables.res50.gpu.yaml"
OUTPUT_DIR="./output_accelerate"
DEEPSPEED_CONFIG="configs/deepspeed.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting DiffusionDet training with DeepSpeed..."
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"

# Launch training with Accelerate + DeepSpeed
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --deepspeed_config_file "$DEEPSPEED_CONFIG" \
    train_accelerate.py \
    --config-file "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --use-deepspeed \
    --use-mlflow \
    --seed 42

echo "Training completed!"
