#!/bin/bash

# DiffusionDet Training (DeepSpeed launcher)
# This script launches training using the DeepSpeed launcher when GPUs are available.

# Activate virtual environment
source .venv/bin/activate

# Configuration
CONFIG_FILE="configs/diffdet.tables.res50.gpu.yaml"
OUTPUT_DIR="./output_accelerate"
# Set to 0 to force CPU-only run. Set to 1 (or more) to run DeepSpeed on GPU(s).
NUM_GPUS=1
DEEPSPEED_CONFIG="configs/deepspeed_config.json"
# Toggle mixed precision (1=enable fp16, 0=disable)
USE_FP16=0

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting DiffusionDet training with DeepSpeed..."
echo "Config: $CONFIG_FILE"
echo "DeepSpeed config: $DEEPSPEED_CONFIG"
echo "Output: $OUTPUT_DIR"
echo "GPUs requested: $NUM_GPUS"

# Launch training
if [ "$NUM_GPUS" -gt 0 ]; then
    # Use Hugging Face Accelerate to launch with DeepSpeed plugin.
    # This avoids CLI mismatches where the system `deepspeed` binary doesn't accept
    # the `--deepspeed_config` argument. Accelerate accepts `--deepspeed_config_file`.
    # Optionally disable fp16: when USE_FP16=0 we create a temporary copy of the
    # DeepSpeed config with fp16.disabled and omit the --fp16 flag so both
    # Accelerate/DeepSpeed and the training script agree.
    if [ "$USE_FP16" -eq 0 ]; then
        echo "Disabling fp16: creating temporary DeepSpeed config without fp16"
        TMP_DS_CONFIG="/tmp/deepspeed_config_no_fp16_$$.json"
        python3 - <<PY
import json
fn = "$DEEPSPEED_CONFIG"
with open(fn,'r') as f:
    cfg = json.load(f)
cfg.setdefault('fp16', {})['enabled'] = False
with open("$TMP_DS_CONFIG", 'w') as f:
    json.dump(cfg, f, indent=2)
print("wrote", "$TMP_DS_CONFIG")
PY
        DS_CONFIG_TO_USE="$TMP_DS_CONFIG"
        FP16_FLAG=""
    else
        DS_CONFIG_TO_USE="$DEEPSPEED_CONFIG"
        FP16_FLAG="--fp16"
    fi

    echo "Launching with Accelerate+DeepSpeed (num_processes=$NUM_GPUS, fp16=$USE_FP16)"
    accelerate launch \
        --config_file configs/accelerate_config.yaml \
        --num_processes $NUM_GPUS \
        --deepspeed_config_file "$DS_CONFIG_TO_USE" \
        train_accelerate.py \
        --config-file "$CONFIG_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --use-deepspeed \
        $FP16_FLAG \
        --gradient-accumulation-steps 4 \
        --use-mlflow \
        --seed 42
    # Clean up temp file if any
    if [ -n "$TMP_DS_CONFIG" ] && [ -f "$TMP_DS_CONFIG" ]; then
        rm -f "$TMP_DS_CONFIG"
    fi
else
    # CPU fallback
    echo "No GPUs requested, running on CPU"
    python train_accelerate.py \
        --config-file "$CONFIG_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --gradient-accumulation-steps 4 \
        --use-mlflow \
        --seed 42 \
        --device cpu
fi

echo "Training completed!"
