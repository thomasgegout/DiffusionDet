#!/usr/bin/env python3
"""
DiffusionDet Training Script with Hugging Face Accelerate and DeepSpeed.

This script replaces the Detectron2 training framework with Hugging Face Accelerate
while maintaining the same LoRA configuration and model architecture.
"""

import os
import math
import argparse
import logging
import warnings
from tqdm.auto import tqdm
import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler

# Hugging Face imports
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed, ProjectConfiguration, DummyOptim, DummyScheduler
from transformers import get_cosine_schedule_with_warmup

# PEFT imports
from peft import LoraConfig, get_peft_model

# Detectron2 imports (for model and data loading)
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances

# DiffusionDet imports
from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs

# MLflow imports
from mlflow_hooks import start_mlflow_run, end_mlflow_run, log_model_architecture
import mlflow
from mlflow import MlflowClient
from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv(".env")

warnings.filterwarnings("ignore", category=FutureWarning, module="detectron2")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register datasets (same as original)
register_coco_instances(
    "pubtables_train", 
    {}, 
    "datasets/PubTables-1M/train.json", 
    "/home/exouser/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db/PubTables-1M-Structure/images"
)

register_coco_instances(
    "pubtables_val", 
    {}, 
    "datasets/PubTables-1M/val_50percent.json", 
    "/home/exouser/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db/PubTables-1M-Structure/images"
)

register_coco_instances(
    "pubtables_test", 
    {}, 
    "datasets/PubTables-1M/test.json",
    "/home/exouser/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db/PubTables-1M-Structure/images"
)


class CosineAnnealingWarmupLR(_LRScheduler):
    """
    Cosine Annealing LR Scheduler with Warmup for better convergence
    (Same as original implementation)
    """
    
    def __init__(self, optimizer, max_iter, warmup_iters=1000, warmup_factor=0.001, eta_min_ratio=0.001, last_epoch=-1):
        self.max_iter = max_iter
        self.warmup_iters = warmup_iters
        self.warmup_factor = warmup_factor
        self.eta_min_ratio = eta_min_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Warmup phase: linear increase
            alpha = self.last_epoch / self.warmup_iters
            warmup_lr = [
                base_lr * (self.warmup_factor * (1 - alpha) + alpha)
                for base_lr in self.base_lrs
            ]
            return warmup_lr
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_iters) / (self.max_iter - self.warmup_iters)
            progress = min(progress, 1.0)
            
            cosine_lr = [
                self.eta_min_ratio * base_lr + 
                (base_lr - self.eta_min_ratio * base_lr) * 
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]
            return cosine_lr


class DiffusionDetTrainer:
    """
    Main trainer class using Hugging Face Accelerate and DeepSpeed
    """
    
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
    
        # Initialize accelerator with DeepSpeed
        self.setup_accelerator()

        self.cfg.MODEL.DEVICE = str(self.accelerator.device).split(':')[0]

        # Set seed for reproducibility
        if args.seed is not None:
            set_seed(args.seed)
        
        # Build model, data, and optimizer
        self.model = self.build_model()
        self.train_dataloader = self.build_train_dataloader()
        self.val_dataloader = self.build_val_dataloader()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        
        # Apply LoRA
        self.model = self.build_peft_model()
        
        # Prepare with accelerator
        self.prepare_training()
        
        # Initialize MLflow
        self.setup_mlflow()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.start_time = time.time()
        self.step_times = []
        
    def setup_accelerator(self):
        """Setup Accelerator with DeepSpeed configuration"""
        # Project configuration
        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir,
            logging_dir=os.path.join(self.args.output_dir, "logs"),
        )
        
        # When using accelerate launch with --deepspeed_config_file, 
        # accelerate will automatically create the DeepSpeed plugin
        # based on the config file, so we don't need to create it manually
        
        # Create accelerator instance - let accelerate decide the device
        self.accelerator = Accelerator(
            project_config=project_config,
            log_with="mlflow" if self.args.use_mlflow else None,
        )
        
        logger.info(f"Using device: {self.accelerator.device}")
        
        self.cfg.MODEL.DEVICE = str(self.accelerator.device)
        
    def build_model(self):
        """Build the DiffusionDet model and load pretrained weights using Detectron2's checkpointer"""
        # Verify the device in config matches accelerator device
        accelerator_device_type = str(self.accelerator.device).split(':')[0]
        if self.cfg.MODEL.DEVICE != accelerator_device_type:
            logger.warning(f"Config device ({self.cfg.MODEL.DEVICE}) doesn't match accelerator device ({accelerator_device_type})")
            
        logger.info(f"Building model with Detectron2 using device: {self.cfg.MODEL.DEVICE}")
        model = build_model(self.cfg)
        
        # Use Detectron2's DetectionCheckpointer to load weights properly
        checkpointer = DetectionCheckpointer(
            model,
            save_dir=self.cfg.OUTPUT_DIR,
        )
        
        # Load pretrained weights if specified in config
        if self.cfg.MODEL.WEIGHTS:
            logger.info(f"Loading pretrained weights using DetectionCheckpointer: {self.cfg.MODEL.WEIGHTS}")
            start_iter = checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=False)
            logger.info(f"Model weights loaded successfully. Start iteration: {start_iter}")
        else:
            logger.info("No pretrained weights specified, using randomly initialized weights")
        
        # Log the device information for debugging
        logger.info(f"Initial model device: {next(model.parameters()).device}")
        logger.info(f"Model will be placed on accelerator device: {self.accelerator.device}")
        return model
    
    def build_peft_model(self):
        """Apply LoRA to the model (same configuration as original)"""
        # Same target modules as in train_net.py
        target_modules = [
            # Detection head attention layers
            "head.head_series.*.self_attn.out_proj",
            "head.head_series.*.self_attn",
            
            # Dynamic Convolution Layers
            "head.head_series.*.inst_interact.dynamic_layer",
            "head.head_series.*.inst_interact.out_layer",

            # Classification and regression heads
            "head.head_series.{0-5}.class_logits",
            "head.head_series.{0-5}.bboxes_delta",

            # Time embedding
            "head.time_mlp.1",
            "head.time_mlp.3",

            # Block Time MLP
            "head.head_series.*.block_time_mlp.1",
            
            # Backbone FPN layers - lateral and output convolutions
            "backbone.fpn_lateral2",
            "backbone.fpn_output2", 
            "backbone.fpn_lateral3",
            "backbone.fpn_output3",
            "backbone.fpn_lateral4",
            "backbone.fpn_output4",
            "backbone.fpn_lateral5",
            "backbone.fpn_output5",
            
            # Key ResNet backbone convolutions
            "backbone.bottom_up.stem.conv1",
            
            # ResNet Stage 2 convolutions (selective)
            "backbone.bottom_up.res2.0.conv1",
            "backbone.bottom_up.res2.0.conv3",
            
            # ResNet Stage 3 convolutions (selective)
            "backbone.bottom_up.res3.0.conv1", 
            "backbone.bottom_up.res3.0.conv3",
            
            # ResNet Stage 4 convolutions (selective)
            "backbone.bottom_up.res4.0.conv1",
            "backbone.bottom_up.res4.0.conv3", 
            
            # ResNet Stage 5 convolutions (selective)
            "backbone.bottom_up.res5.0.conv1",
            "backbone.bottom_up.res5.0.conv3",
        ]
        
        modules_to_save = [
            # Classification heads - must be fully adapted for new classes
            "head.head_series.*.class_logits",
            
            # Time embeddings - critical for diffusion process
            "head.time_mlp.0",  # SinusoidalPositionEmbeddings
        ]
        
        # Create LoRA configuration (same as original)
        lora_config = LoraConfig(
            init_lora_weights="gaussian",
            target_modules=target_modules,
            modules_to_save=modules_to_save,
            r=16,  # Default rank
            lora_alpha=32,  # Default alpha
            lora_dropout=0.1,  # Default dropout
            bias="none",
        )
        
        # Apply LoRA
        peft_model = get_peft_model(self.model, lora_config)
        peft_model.print_trainable_parameters()
        
        return peft_model
    
    def build_train_dataloader(self):
        """Build training data loader"""
        train_loader = build_detection_train_loader(
            self.cfg,
            mapper=DiffusionDetDatasetMapper(self.cfg, True)
        )
        return train_loader
    
    def build_val_dataloader(self):
        """Build validation data loader"""
        val_loader = build_detection_test_loader(
            self.cfg,
            self.cfg.DATASETS.TEST[0],
            mapper=DiffusionDetDatasetMapper(self.cfg, False)
        )
        return val_loader
    
    def build_optimizer(self):
        """Build optimizer (same configuration as original)"""
        # When using DeepSpeed with optimizer in config file, return None
        # DeepSpeed will create its own optimizer from the config
        if self.args.use_deepspeed:
            logger.info("Returning None for optimizer because DeepSpeed config defines optimizer")
            return None
        
        # Get parameters that require gradients
        params_with_grad = []
        params_without_grad = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_with_grad.append(param)
            else:
                params_without_grad.append(param)
        
        logger.info(f"Parameters with gradient: {len(params_with_grad)}")
        logger.info(f"Parameters without gradient: {len(params_without_grad)}")
        
        # Create optimizer (AdamW as specified in config)
        optimizer = AdamW(
            params_with_grad,
            lr=self.cfg.SOLVER.BASE_LR,
            weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        return optimizer
    
    def build_scheduler(self):
        """Build learning rate scheduler"""
        # When using DeepSpeed with scheduler in config file, return None
        # DeepSpeed will create its own scheduler from the config
        if self.args.use_deepspeed:
            logger.info("Returning None for scheduler because DeepSpeed config defines scheduler")
            return None
            
        total_steps = self.cfg.SOLVER.MAX_ITER
        warmup_steps = self.cfg.SOLVER.WARMUP_ITERS
        
        if warmup_steps > 0:
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            scheduler = CosineAnnealingWarmupLR(
                self.optimizer,
                max_iter=total_steps,
                warmup_iters=0,
                eta_min_ratio=0.001
            )
        
        return scheduler
    
    def prepare_training(self):
        """Prepare training components with accelerator"""
        # When using DeepSpeed with optimizer/scheduler in config, 
        # we only prepare model and dataloaders
        if self.args.use_deepspeed and self.optimizer is None:
            logger.info("Preparing model and dataloaders only (DeepSpeed handles optimizer/scheduler)")
            (
                self.model,
                self.train_dataloader,
                self.val_dataloader,
            ) = self.accelerator.prepare(
                self.model,
                self.train_dataloader,
                self.val_dataloader,
            )
            # With DeepSpeed, the model is wrapped and contains the optimizer and scheduler
            # Access them through the DeepSpeed engine
            if hasattr(self.model, 'optimizer'):
                self.optimizer = self.model.optimizer
            if hasattr(self.model, 'lr_scheduler'):
                self.scheduler = self.model.lr_scheduler
            else:
                self.scheduler = None
                
            logger.info(f"DeepSpeed optimizer: {type(self.optimizer) if self.optimizer else 'None'}")
            logger.info(f"DeepSpeed scheduler: {type(self.scheduler) if self.scheduler else 'None'}")
        else:
            # Standard preparation for non-DeepSpeed or DeepSpeed without config optimizer
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.val_dataloader,
                self.scheduler
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.val_dataloader,
                self.scheduler
            )
    
    def setup_mlflow(self):
        """Setup MLflow tracking with proper environment configuration"""
        if self.args.use_mlflow and self.accelerator.is_main_process:
            # Load environment variables
            load_dotenv(".env")
            
            # Setup MLflow tracking URI from environment
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
            if mlflow_uri:
                mlflow.set_tracking_uri(mlflow_uri)
                logger.info(f"MLflow tracking URI set to: {mlflow_uri}")
            
            # Start MLflow run with experiment name from environment or default
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "thomas/DiffusionDet_Accelerate")
            run_name = os.getenv("MLFLOW_RUN_NAME", "DiffusionDet_Accelerate_Training")
            
            # Create custom start function similar to mlflow_hooks
            self._start_mlflow_run_accelerate(experiment_name, run_name)
            
            # Log model architecture
            log_model_architecture(self.model)
    
    def _start_mlflow_run_accelerate(self, experiment_name, run_name):
        """Start MLflow run similar to mlflow_hooks.py"""
        try:
            # Configure MLflow client
            client = MlflowClient()
            
            # Check if experiment exists, if not create it
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(
                    experiment_name,
                    artifact_location=os.getenv("MLFLOW_ARTIFACT_URI"),
                    tags={"version": "v1", "priority": "P1", "framework": "accelerate"},
                )
                logger.info(f"Created new MLflow experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing MLflow experiment: {experiment_name}")
            
            # Set experiment tags
            client.set_experiment_tag(experiment_id, "framework", "accelerate")
            client.set_experiment_tag(experiment_id, "model", "DiffusionDet")
            client.set_experiment_tag(experiment_id, "training_type", "LoRA")
            
            # Start MLflow run
            run = client.create_run(
                experiment_id=experiment_id,
                run_name=run_name,
                tags={
                    "framework": "accelerate",
                    "model": "DiffusionDet",
                    "cloud": "gcp",
                    "project": os.getenv("GCP_PROJECT_ID", "docugami"),
                    "environment": os.getenv("ENVIRONMENT", "development"),
                    "run_name": run_name,
                    "training_type": "LoRA",
                    "device": self.args.device or "auto"
                }
            )
            
            # Set the active run
            mlflow.start_run(run_id=run.info.run_id)
            
            # Log configuration parameters
            self._log_config_params(client, run.info.run_id)
            
            logger.info(f"MLflow run started: {run.info.run_id}")
            logger.info(f"MLflow run name: {run_name}")
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            # Continue training without MLflow
            self.args.use_mlflow = False
    
    def _log_config_params(self, client, run_id):
        """Log configuration parameters to MLflow"""
        try:
            params = {
                "model_type": "DiffusionDet",
                "framework": "accelerate",
                "dataset": "PubTables-1M",
                "batch_size": str(self.cfg.SOLVER.IMS_PER_BATCH),
                "learning_rate": str(self.cfg.SOLVER.BASE_LR),
                "max_iter": str(self.cfg.SOLVER.MAX_ITER),
                                    "device": str(self.accelerator.device),
                "gradient_accumulation_steps": str(self.args.gradient_accumulation_steps),
                "fp16": str(self.args.fp16),
                "use_deepspeed": str(self.args.use_deepspeed),
                "seed": str(self.args.seed),
                "backbone_multiplier": str(getattr(self.cfg.SOLVER, 'BACKBONE_MULTIPLIER', 1.0)),
                "optimizer": str(getattr(self.cfg.SOLVER, 'OPTIMIZER', 'AdamW')),
                "weight_decay": str(self.cfg.SOLVER.WEIGHT_DECAY),
                "eval_period": str(self.cfg.TEST.EVAL_PERIOD),
                "checkpoint_period": str(self.cfg.SOLVER.CHECKPOINT_PERIOD),
                "gcp_project": os.getenv("GCP_PROJECT_ID", "docugami")
            }
            
            for key, value in params.items():
                if value is not None and value != "None":
                    client.log_param(run_id, key, value)
            
            logger.info("Configuration parameters logged to MLflow")
            
        except Exception as e:
            logger.warning(f"Failed to log config parameters: {e}")
    
    def compute_loss(self, batch):
        """Compute loss for a batch during training"""
        # Forward pass - Accelerate should handle device placement
        outputs = self.model(batch)
        
        # DiffusionDet returns losses in the outputs dict during training
        losses = {k: v for k, v in outputs.items() if 'loss' in k}
        total_loss = sum(losses.values())
        
        return total_loss, losses
    
    def compute_eval_loss(self, batch):
        """Compute loss for a batch during evaluation"""
        # Temporarily set model to training mode to get losses
        was_training = self.model.training
        self.model.train()
        
        try:
            # Forward pass
            outputs = self.model(batch)
            
            # Extract losses
            losses = {k: v for k, v in outputs.items() if 'loss' in k}
            total_loss = sum(losses.values())
            
            return total_loss, losses
        finally:
            # Restore original mode
            if not was_training:
                self.model.eval()
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        with self.accelerator.accumulate(self.model):
            total_loss, losses = self.compute_loss(batch)
            
            # Backward pass
            self.accelerator.backward(total_loss)
            
            # Gradient clipping - let accelerate handle this for DeepSpeed
            if self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
                )
            
            # For DeepSpeed, accelerate handles optimizer and scheduler steps
            if not self.args.use_deepspeed:
                # Only call these directly if not using DeepSpeed
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        return total_loss, losses
    
    def evaluate(self):
        """Evaluation step with progress tracking"""
        self.model.eval()
        
        # Simple evaluation - in practice you'd want to use COCOEvaluator
        total_val_loss = 0
        num_batches = 0
        max_eval_batches = 50  # Limit evaluation batches for speed
        
        # Create evaluation progress bar for main process
        eval_iter = iter(self.val_dataloader)
        if self.accelerator.is_main_process:
            eval_progress = tqdm(
                total=max_eval_batches,
                desc="Evaluating",
                unit="batch",
                leave=False,
                dynamic_ncols=True
            )
        
        with torch.no_grad():
            try:
                for i in range(max_eval_batches):
                    try:
                        batch = next(eval_iter)
                    except StopIteration:
                        break
                    
                    val_loss, _ = self.compute_eval_loss(batch)
                    total_val_loss += val_loss.item()
                    num_batches += 1
                    
                    if self.accelerator.is_main_process:
                        eval_progress.set_postfix({
                            'val_loss': f'{val_loss.item():.4f}',
                            'avg_loss': f'{total_val_loss/num_batches:.4f}'
                        })
                        eval_progress.update(1)
            
            finally:
                if self.accelerator.is_main_process:
                    eval_progress.close()
        
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
        return {
            "val_loss": avg_val_loss,
            "val_batches": num_batches
        }
    
    def log_metrics(self, metrics, step):
        """Log metrics to MLflow and console"""
        if self.accelerator.is_main_process:
            # Calculate timing information
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Calculate step timing
            if len(self.step_times) > 0:
                avg_step_time = sum(self.step_times[-10:]) / min(len(self.step_times), 10)  # Average of last 10 steps
                steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                eta_seconds = (self.cfg.SOLVER.MAX_ITER - step) * avg_step_time
                eta_str = f"{eta_seconds // 3600:.0f}h {(eta_seconds % 3600) // 60:.0f}m {eta_seconds % 60:.0f}s"
            else:
                steps_per_sec = 0
                eta_str = "N/A"
            
            # Enhanced log string with timing
            log_str = f"Step {step}/{self.cfg.SOLVER.MAX_ITER} "
            log_str += f"({step/self.cfg.SOLVER.MAX_ITER*100:.1f}%) - "
            log_str += f"Time: {elapsed_time//3600:.0f}h {(elapsed_time%3600)//60:.0f}m {elapsed_time%60:.0f}s - "
            log_str += f"Speed: {steps_per_sec:.2f} steps/s - "
            log_str += f"ETA: {eta_str} - "
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    log_str += f"{key}={value:.6f} "
                else:
                    log_str += f"{key}={value} "
            
            logger.info(log_str)
            
            # Log to MLflow using client for robustness
            if self.args.use_mlflow and mlflow.active_run():
                try:
                    client = MlflowClient()
                    run_id = mlflow.active_run().info.run_id
                    
                    # Log all provided metrics
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            client.log_metric(run_id, key, value, step=step)
                    
                    # Log timing metrics
                    client.log_metric(run_id, "steps_per_second", steps_per_sec, step=step)
                    client.log_metric(run_id, "elapsed_time_hours", elapsed_time / 3600, step=step)
                    client.log_metric(run_id, "avg_step_time", avg_step_time if len(self.step_times) > 0 else 0, step=step)
                    
                    # Log ETA as a metric (in seconds)
                    if eta_str != "N/A":
                        client.log_metric(run_id, "eta_seconds", eta_seconds, step=step)
                
                except Exception as e:
                    logger.warning(f"Failed to log metrics to MLflow: {e}")
                    # Continue training even if MLflow logging fails
    
    def save_checkpoint(self, step):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(checkpoint_dir)
            
            # Save training state
            self.accelerator.save_state(checkpoint_dir)
            
            logger.info(f"Checkpoint saved at step {step}")
    
    def train(self):
        """Main training loop with tqdm progress tracking"""
        logger.info("Starting training...")
        logger.info(f"Total steps: {self.cfg.SOLVER.MAX_ITER}")
        logger.info(f"Evaluation every: {self.cfg.TEST.EVAL_PERIOD} steps")
        logger.info(f"Checkpoint every: {self.cfg.SOLVER.CHECKPOINT_PERIOD} steps")
        
        self.model.train()
        
        # Initialize progress bar
        if self.accelerator.is_main_process:
            progress_bar = tqdm(
                total=self.cfg.SOLVER.MAX_ITER,
                desc="Training",
                unit="step",
                dynamic_ncols=True,
                leave=True
            )
        
        # Training loop
        step = 0
        train_iterator = iter(self.train_dataloader)
        
        try:
            while step < self.cfg.SOLVER.MAX_ITER:
                step_start_time = time.time()
                
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    # Reset iterator when dataset is exhausted
                    train_iterator = iter(self.train_dataloader)
                    batch = next(train_iterator)
                    self.epoch += 1
                    if self.accelerator.is_main_process:
                        logger.info(f"Starting epoch {self.epoch}")
                
                # Training step
                total_loss, losses = self.train_step(batch)
                
                # Gather losses from all processes
                total_loss = self.accelerator.gather(total_loss).mean()
                
                # Track step timing
                step_time = time.time() - step_start_time
                self.step_times.append(step_time)
                if len(self.step_times) > 100:  # Keep only last 100 step times
                    self.step_times = self.step_times[-100:]
                
                # Update progress bar
                if self.accelerator.is_main_process:
                    # Calculate metrics for progress bar
                    avg_step_time = sum(self.step_times[-10:]) / min(len(self.step_times), 10)
                    steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    
                    progress_bar.set_postfix({
                        'loss': f'{total_loss.item():.4f}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                        'step/s': f'{steps_per_sec:.2f}',
                        'epoch': self.epoch
                    })
                    progress_bar.update(1)
                
                # Log metrics
                if step % 20 == 0:  # Log every 20 steps
                    metrics = {
                        "train_loss": total_loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "epoch": self.epoch,
                        "step_time": step_time,
                    }
                    # Add individual losses
                    for key, value in losses.items():
                        gathered_loss = self.accelerator.gather(value).mean()
                        metrics[f"train_{key}"] = gathered_loss.item()
                    
                    self.log_metrics(metrics, step)
                
                # Evaluation
                if step > 0 and step % self.cfg.TEST.EVAL_PERIOD == 0:
                    if self.accelerator.is_main_process:
                        progress_bar.set_description("Evaluating")
                    
                    eval_metrics = self.evaluate()
                    self.log_metrics(eval_metrics, step)
                    self.model.train()  # Back to training mode
                    
                    if self.accelerator.is_main_process:
                        progress_bar.set_description("Training")
                
                # Save checkpoint
                if step > 0 and step % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                    if self.accelerator.is_main_process:
                        progress_bar.set_description("Saving checkpoint")
                    
                    self.save_checkpoint(step)
                    
                    if self.accelerator.is_main_process:
                        progress_bar.set_description("Training")
                
                step += 1
                self.global_step = step
        
        except KeyboardInterrupt:
            if self.accelerator.is_main_process:
                progress_bar.close()
            logger.info("Training interrupted by user")
        
        finally:
            # Close progress bar
            if self.accelerator.is_main_process:
                progress_bar.close()
            
            # Final checkpoint
            self.save_checkpoint(step)
            
            # End MLflow run
            if self.args.use_mlflow and self.accelerator.is_main_process:
                end_mlflow_run(self.accelerator.unwrap_model(self.model))
        
        logger.info("Training completed!")
        
        # Print final training summary
        if self.accelerator.is_main_process:
            total_time = time.time() - self.start_time
            avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
            logger.info("Training Summary:")
            logger.info(f"  Total steps: {step}")
            logger.info(f"  Total time: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s")
            logger.info(f"  Average step time: {avg_step_time:.3f}s")
            logger.info(f"  Average steps per second: {1.0/avg_step_time:.2f}")
            logger.info(f"  Epochs completed: {self.epoch}")


def setup_config(args):
    """Setup configuration"""
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)  # Add this BEFORE loading config file
    
    # Load config file
    cfg.merge_from_file(args.config_file)
    
    # Override with command line arguments
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # Set output directory
    cfg.OUTPUT_DIR = args.output_dir
    
    # Disable EMA for simplicity in this implementation
    cfg.MODEL_EMA.ENABLED = False
    
    return cfg


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DiffusionDet training with Accelerate")
    
    # Required arguments
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--output-dir", default="./output_accelerate", help="Output directory")
    
    # Optional arguments
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu/mps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, 
                       help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--use-deepspeed", action="store_true", help="Use DeepSpeed")
    parser.add_argument("--use-mlflow", action="store_true", help="Use MLflow logging")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[], 
                       help="Modify config options using the command-line")
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup configuration
    cfg = setup_config(args)
    
    # Initialize trainer
    trainer = DiffusionDetTrainer(cfg, args)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
