#!/usr/bin/env python3
"""
Step-by-step debugging script
"""

print("Step 1: Basic imports")
import sys
print("✓ sys imported")

print("Step 2: Torch import")
import torch
print("✓ torch imported")

print("Step 3: Accelerate import")
from accelerate import Accelerator
print("✓ accelerate imported")

print("Step 4: Detectron2 imports")
from detectron2.config import get_cfg
print("✓ detectron2.config imported")

from detectron2.modeling import build_model
print("✓ detectron2.modeling imported")

print("Step 5: DiffusionDet imports")
sys.stdout.flush()  # Force output
from diffusiondet import add_diffusiondet_config
print("✓ diffusiondet imported")

print("Step 6: Test config creation")
cfg = get_cfg()
add_diffusiondet_config(cfg)
print("✓ config created")

print("Step 7: Test config file loading")
cfg.merge_from_file("configs/diffdet.tables.res50.cpu.yaml")
cfg.freeze()
print("✓ config file loaded")

print("Step 8: Test accelerator creation")
accelerator = Accelerator()
print("✓ accelerator created")

print("All steps completed successfully!")
