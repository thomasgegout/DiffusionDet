"""
Training optimizations for DiffusionDet
This file contains additional optimizations that can be applied
"""

import torch
import logging
from detectron2.engine.train_loop import HookBase
from detectron2.utils.logger import setup_logger

logger = logging.getLogger(__name__)


class ThroughputHook(HookBase):
    """
    Hook to monitor training throughput (images/sec)
    """
    
    def __init__(self, warmup_iter=100):
        self.warmup_iter = warmup_iter
        self.start_time = None
        self.total_images = 0
        
    def before_train(self):
        self.start_time = None
        self.total_images = 0
        
    def after_step(self):
        if self.trainer.iter == self.warmup_iter:
            self.start_time = self.trainer.storage.history("time").latest()[1]
            self.total_images = 0
            
        if self.trainer.iter > self.warmup_iter and self.start_time is not None:
            batch_size = len(self.trainer.data_loader_iter.peek())
            self.total_images += batch_size
            
            if self.trainer.iter % 100 == 0:
                current_time = self.trainer.storage.history("time").latest()[1]
                elapsed = current_time - self.start_time
                if elapsed > 0:
                    throughput = self.total_images / elapsed
                    logger.info(f"Training throughput: {throughput:.2f} images/sec")


class MemoryHook(HookBase):
    """
    Hook to monitor GPU memory usage
    """
    
    def __init__(self, log_period=500):
        self.log_period = log_period
        
    def after_step(self):
        if self.trainer.iter % self.log_period == 0 and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")


def add_training_optimizations():
    """
    Additional training optimizations
    """
    # Set optimal settings for PyTorch
    torch.backends.cudnn.benchmark = True  # Optimize cudnn for fixed input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for speed
    
    # Set number of threads for data loading
    torch.set_num_threads(4)  # Limit PyTorch threads to leave room for dataloader
    
    logger.info("Applied training optimizations:")
    logger.info("- CUDNN benchmark enabled")
    logger.info("- PyTorch threads limited to 4")


def print_training_config(cfg):
    """
    Print important training configuration
    """
    logger.info("=== Training Configuration ===")
    logger.info(f"AMP Enabled: {cfg.SOLVER.AMP.ENABLED}")
    logger.info(f"EMA Enabled: {cfg.MODEL_EMA.ENABLED}")
    if cfg.MODEL_EMA.ENABLED:
        logger.info(f"EMA Decay: {cfg.MODEL_EMA.DECAY}")
    logger.info(f"Optimizer: {cfg.SOLVER.OPTIMIZER}")
    logger.info(f"Base LR: {cfg.SOLVER.BASE_LR}")
    logger.info(f"Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    logger.info(f"Max Iterations: {cfg.SOLVER.MAX_ITER}")
    logger.info(f"Warmup Iterations: {cfg.SOLVER.WARMUP_ITERS}")
    logger.info(f"DataLoader Workers: {cfg.DATALOADER.NUM_WORKERS}")
    logger.info(f"Gradient Clipping: {cfg.SOLVER.CLIP_GRADIENTS.ENABLED}")
    if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
        logger.info(f"Clip Value: {cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE}")
    logger.info("===============================")
