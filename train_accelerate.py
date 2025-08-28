#!/usr/bin/env python3
"""
DiffusionDet Training Script with Hugging Face Accelerate
This is an experimental script that combines Accelerate with Detectron2 components
"""

import os
import torch
import logging
from accelerate import Accelerator
from accelerate.utils import set_seed

from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.logger import setup_logger

from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs


def setup_cfg(config_file):
    """Setup configuration"""
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def main():
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16",  # Enable mixed precision
        gradient_accumulation_steps=1,
        log_with="tensorboard",  # or "wandb"
        project_dir="./logs"
    )
    
    # Setup logging
    if accelerator.is_main_process:
        setup_logger()
    
    logger = logging.getLogger(__name__)
    
    # Load config
    cfg = setup_cfg("configs/diffdet.tables.res50.gpu.yaml")
    
    # Set seed for reproducibility
    set_seed(cfg.SEED)
    
    # Build model
    model = build_model(cfg)
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    
    # Build data loader
    mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    
    # Prepare everything with accelerator
    model, optimizer, data_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, data_loader, lr_scheduler
    )
    
    # Training loop
    model.train()
    for iteration, batch in enumerate(data_loader):
        if iteration >= cfg.SOLVER.MAX_ITER:
            break
            
        with accelerator.accumulate(model):
            # Forward pass
            losses = model(batch)
            loss = sum(losses.values())
            
            # Backward pass
            accelerator.backward(loss)
            
            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            if accelerator.is_main_process and iteration % 20 == 0:
                logger.info(f"Iter: {iteration}, Loss: {loss.item():.4f}")
                accelerator.log({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
    
    # Save model
    if accelerator.is_main_process:
        accelerator.save_model(model, cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
