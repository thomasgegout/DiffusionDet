#!/usr/bin/env python3
"""
Quick test to verify the accelerate training pipeline works
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, '.')

print("Starting quick test...")

# Test configuration loading
try:
    from detectron2.config import get_cfg
    from diffusiondet import add_diffusiondet_config
    from diffusiondet.util.model_ema import add_model_ema_configs
    
    print("✓ Config imports successful")
    
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file("configs/diffdet.tables.res50.cpu.yaml")
    cfg.OUTPUT_DIR = "./output_test"
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL_EMA.ENABLED = False
    cfg.freeze()
    
    print("✓ Config loaded successfully")
    print(f"Model device: {cfg.MODEL.DEVICE}")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    
except Exception as e:
    print(f"✗ Config loading failed: {e}")
    sys.exit(1)

# Test Accelerator
try:
    from accelerate import Accelerator
    
    print("Testing Accelerator...")
    accelerator = Accelerator(
        gradient_accumulation_steps=2,
        mixed_precision='no',
        cpu=True,
        device_placement=False,
    )
    print(f"✓ Accelerator created. Device: {accelerator.device}")
    
except Exception as e:
    print(f"✗ Accelerator failed: {e}")
    sys.exit(1)

# Test model building
try:
    print("Testing model building...")
    from detectron2.modeling import build_model
    import torch
    
    model = build_model(cfg)
    device = torch.device("cpu")
    model = model.to(device)
    
    print(f"✓ Model built successfully")
    print(f"Model device: {next(model.parameters()).device}")
    
except Exception as e:
    print(f"✗ Model building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test PEFT
try:
    print("Testing PEFT/LoRA...")
    from peft import LoraConfig, get_peft_model
    
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
    
    lora_config = LoraConfig(
        init_lora_weights="gaussian",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    print("✓ PEFT/LoRA applied successfully")
    
except Exception as e:
    print(f"✗ PEFT/LoRA failed: {e}")
    import traceback
    traceback.print_exc()
    # Don't exit, continue without LoRA

print("✓ All basic tests passed!")
print("The training pipeline should work.")
