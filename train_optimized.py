#!/usr/bin/env python3
"""
Enhanced DiffusionDet Training Script with Optimizations
This script includes all the training optimizations:
- AMP (Automatic Mixed Precision)
- EMA (Exponential Moving Average)
- Optimized LR scheduling
- Training monitoring hooks
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from train_net import Trainer, setup, main as original_main, default_argument_parser
from training_optimizations import ThroughputHook, MemoryHook, add_training_optimizations, print_training_config
from cosine_scheduler import build_cosine_annealing_lr_scheduler

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger


class OptimizedTrainer(Trainer):
    """
    Enhanced Trainer with additional optimizations
    """
    
    def __init__(self, cfg):
        # Apply training optimizations before initialization
        add_training_optimizations()
        
        # Call parent constructor
        super().__init__(cfg)
        
        # Print configuration
        if comm.is_main_process():
            print_training_config(cfg)
    
    def build_hooks(self):
        """
        Build enhanced hooks with monitoring
        """
        hooks = super().build_hooks()
        
        # Add monitoring hooks
        if comm.is_main_process():
            hooks.extend([
                ThroughputHook(warmup_iter=100),
                MemoryHook(log_period=500),
            ])
            
        return [hook for hook in hooks if hook is not None]
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        Build enhanced LR scheduler
        """
        # You can switch between schedulers here
        use_cosine_annealing = getattr(cfg.SOLVER, 'USE_COSINE_ANNEALING', False)
        
        if use_cosine_annealing:
            return build_cosine_annealing_lr_scheduler(cfg, optimizer)
        else:
            # Use default step scheduler
            return super().build_lr_scheduler(cfg, optimizer)


def main(args):
    """
    Enhanced main function
    """
    cfg = setup(args)
    
    # Use optimized trainer
    if args.eval_only:
        # For evaluation, use original logic
        return original_main(args)
    
    trainer = OptimizedTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    
    # Setup logging
    setup_logger(name="fvcore")
    logger = logging.getLogger("diffusiondet")
    logger.info("Starting enhanced DiffusionDet training...")
    
    main(args)
