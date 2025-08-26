"""
MLflow hooks for DiffusionDet training pipeline.
Provides custom hooks to log training and validation metrics to MLflow.
"""

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
import detectron2.utils.comm as comm
from detectron2.engine import hooks
from detectron2.utils.events import EventStorage
import os
from dotenv import load_dotenv
import urllib3


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv("diffusiondet/.env")

class MLflowHook(hooks.HookBase):
    """
    Custom hook to log training metrics to MLflow
    """
    def __init__(self, cfg, log_period=20):
        """
        Args:
            cfg: Configuration object
            log_period (int): Log metrics every N iterations
        """
        self.cfg = cfg
        self._period = log_period
        self.client = MlflowClient()

    def after_step(self):
        """Log training metrics after each step"""
        if self.trainer.iter % self._period == 0 and mlflow.active_run():
            # Get the latest metrics from the trainer's storage
            storage = self.trainer.storage
            run_id = mlflow.active_run().info.run_id
            
            # Log training loss
            if "total_loss" in storage.latest():
                self.client.log_metric(run_id, "train_loss", storage.latest()["total_loss"][0], step=self.trainer.iter)
            
            # Log learning rate
            if "lr" in storage.latest():
                self.client.log_metric(run_id, "learning_rate", storage.latest()["lr"][0], step=self.trainer.iter)
                
            # Log other losses if available
            for key, (value, _) in storage.latest().items():
                if "loss" in key.lower() and key != "total_loss":
                    self.client.log_metric(run_id, f"train_{key}", value, step=self.trainer.iter)


class MLflowEvalHook(hooks.EvalHook):
    """
    Custom evaluation hook that logs validation metrics to MLflow
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = MlflowClient()

    def _do_eval(self):
        """Perform evaluation and log results to MLflow"""
        results = self._func()
        
        if results and mlflow.active_run():
            run_id = mlflow.active_run().info.run_id
            # Log validation metrics
            for task, metrics in results.items():
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            self.client.log_metric(run_id, f"val_{metric_name}", metric_value, step=self.trainer.iter)
        
        if comm.is_main_process():
            self.trainer.storage.put_scalars(**results, smoothing_hint=False)
        
        return results

def start_mlflow_run(cfg, experiment_name="thomas/DiffusionDet_Training_5"):
    """
    Initialize MLflow experiment and start a new run
    
    Args:
        cfg: Configuration object
        experiment_name (str): Name of the MLflow experiment
    """
    if comm.is_main_process():
        # Configure for Google Cloud
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

        client = MlflowClient()
        
        # Check if experiment exists, if not create it with artifact location
        try:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(
                    experiment_name,
                    artifact_location=os.getenv("MLFLOW_ARTIFACT_URI"),
                    tags={"version": "v1", "priority": "P1"},
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception:
            # Fallback: create experiment if get_experiment_by_name fails
            experiment_id = client.create_experiment(
                experiment_name,
                artifact_location=os.getenv("MLFLOW_ARTIFACT_URI"),
                tags={"version": "v1", "priority": "P1"},
            )

        # Set experiment tags
        client.set_experiment_tag(experiment_id, "framework", "detectron2")
        client.set_experiment_tag(experiment_id, "model", "DiffusionDet")

        # Start MLflow run using client
        run = client.create_run(
            experiment_id=experiment_id,
            run_name=os.getenv("MLFLOW_RUN_NAME", "DiffusionDet_Training"),
            tags={
                "framework": "detectron2",
                "model": "DiffusionDet",
                "cloud": "gcp",
                "project": "docugami",
                "environment": os.getenv("ENVIRONMENT", "development"),
                "run_name": os.getenv("MLFLOW_RUN_NAME", "DiffusionDet_Training")
            }
        )
        
        # Set the active run
        mlflow.start_run(run_id=run.info.run_id)
        
        # Log configuration parameters using client
        params = {
            "model_type": "DiffusionDet",
            "dataset": "PubTables-1M",
            "batch_size": str(cfg.SOLVER.IMS_PER_BATCH),
            "learning_rate": str(cfg.SOLVER.BASE_LR),
            "max_iter": str(cfg.SOLVER.MAX_ITER),
            "device": str(cfg.MODEL.DEVICE),
            "backbone_multiplier": str(cfg.SOLVER.BACKBONE_MULTIPLIER),
            "optimizer": str(cfg.SOLVER.OPTIMIZER),
            "weight_decay": str(cfg.SOLVER.WEIGHT_DECAY),
            "momentum": str(getattr(cfg.SOLVER, 'MOMENTUM', None)),
            "eval_period": str(cfg.TEST.EVAL_PERIOD),
            "checkpoint_period": str(cfg.SOLVER.CHECKPOINT_PERIOD),
            "gcp_project": "docugami"
        }
        
        for key, value in params.items():
            if value is not None:
                client.log_param(run.info.run_id, key, value)


def end_mlflow_run(model=None):
    """
    End the current MLflow run and optionally log the model
    
    Args:
        model: PyTorch model to log (optional)
    """
    if comm.is_main_process() and mlflow.active_run():
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        
        if model is not None:
            try:
                # Log the final model using client
                mlflow.pytorch.log_model(model, artifact_path="model")
                print("Model logged successfully to MLflow")
            except Exception as e:
                print(f"Warning: Could not log model to MLflow: {e}")
                # Try alternative approach - save model as artifact
                try:
                    import tempfile
                    import torch
                    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
                        torch.save(model.state_dict(), tmp.name)
                        client.log_artifact(run_id, tmp.name, "model")
                        os.unlink(tmp.name)
                    print("Model state dict logged as artifact using client")
                except Exception as e2:
                    print(f"Alternative model logging also failed: {e2}")
        
        # End run using standard mlflow call (client doesn't have end_run method)
        mlflow.end_run()

def log_model_architecture(model):
    """
    Log model architecture information to MLflow
    
    Args:
        model: PyTorch model
    """
    if comm.is_main_process() and mlflow.active_run():
        client = MlflowClient()
        run_id = mlflow.active_run().info.run_id
        
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Log parameters using client
            client.log_param(run_id, "total_parameters", str(total_params))
            client.log_param(run_id, "trainable_parameters", str(trainable_params))
            client.log_param(run_id, "model_architecture", str(model.__class__.__name__))
            
        except Exception as e:
            print(f"Warning: Could not log model architecture: {e}")