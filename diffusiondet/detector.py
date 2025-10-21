# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
DiffusionDet: Diffusion Model for Object Detection

This implementation is based on the paper:
"DiffusionDet: Diffusion Model for Object Detection" by Shoufa Chen et al.
Paper: https://arxiv.org/abs/2211.09788
GitHub Repository: https://github.com/ShoufaChen/DiffusionDet

DiffusionDet formulates object detection as a denoising diffusion process from noisy boxes to object boxes.
The model learns to reverse the forward diffusion process that gradually adds noise to ground truth boxes,
enabling it to generate high-quality object detections through an iterative denoising procedure.
"""

import math
import random
from typing import List, Tuple, Optional, Union
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, detector_postprocess

from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import nested_tensor_from_tensor_list
from EVA_backbone.eva_02 import EVA02_ViT, SimpleFeaturePyramid
from functools import partial
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

__all__ = ["DiffusionDet"]

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

def exists(x) -> bool:
    """
    Check if a value exists (is not None).
    
    This utility function is commonly used in diffusion model implementations
    to handle optional parameters and conditional logic.
    
    Args:
        x: Any value to check for existence
        
    Returns:
        bool: True if x is not None, False otherwise
        
    Reference:
        Utility function pattern from DiffusionDet implementation:
        https://github.com/ShoufaChen/DiffusionDet
    """
    return x is not None


def default(val, d):
    """
    Return val if it exists, otherwise return default value d.
    
    This is a common pattern in diffusion models for handling optional parameters
    with fallback defaults. Supports both static values and callable defaults.
    
    Args:
        val: The primary value to use if it exists
        d: Default value or callable that returns default value
        
    Returns:
        val if val is not None, otherwise d (or d() if callable)
        
    Reference:
        Utility function from DiffusionDet repository for parameter handling.
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Extract values from tensor a at indices specified by t, reshaping for broadcasting.
    
    This function is fundamental to diffusion models as it extracts the appropriate
    noise schedule parameters (α, β, etc.) for each timestep in a batch.
    The reshaping ensures proper broadcasting when combining with input tensors.
    
    Args:
        a (torch.Tensor): 1D tensor containing values indexed by timestep
        t (torch.Tensor): Tensor of timestep indices, shape (batch_size,)
        x_shape (Tuple[int, ...]): Shape of the tensor these values will be applied to
        
    Returns:
        torch.Tensor: Extracted values reshaped for broadcasting with x_shape
        
    Mathematical Background:
        In DDPM, we need to extract β_t, α̅_t, etc. for each timestep t in the batch.
        This function handles the indexing and reshaping: a[t] -> shape (batch_size, 1, 1, ...)
        
    Reference:
        Standard operation in diffusion models, see DDPM paper (Ho et al., 2020)
        and DiffusionDet implementation.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Create cosine noise schedule for diffusion process.
    
    The cosine schedule provides a more stable training process compared to linear schedules
    by reducing noise more gradually at the beginning and end of the diffusion process.
    This schedule was introduced to improve sample quality and training stability.
    
    Args:
        timesteps (int): Total number of diffusion timesteps
        s (float, optional): Small offset to prevent β_t from being too small. Defaults to 0.008.
        
    Returns:
        torch.Tensor: Beta values for each timestep, shape (timesteps,)
        
    Mathematical Formulation:
        α̅_t = cos²((t/T + s)/(1 + s) * π/2)
        β_t = 1 - α̅_t/α̅_{t-1}
        
    Reference:
        "Improved Denoising Diffusion Probabilistic Models" by Nichol & Dhariwal (2021)
        https://openreview.net/forum?id=-NEXDKk8gZ
        
        Used in DiffusionDet for more stable training dynamics.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


@META_ARCH_REGISTRY.register()
class DiffusionDet(nn.Module):
    """
    DiffusionDet: Diffusion Model for Object Detection
    
    This class implements the DiffusionDet architecture that formulates object detection
    as a denoising diffusion process. Unlike traditional detectors that directly predict
    object locations, DiffusionDet learns to iteratively refine noisy bounding boxes
    into precise object detections through a reverse diffusion process.
    
    Key Components:
    1. Forward diffusion: Gradually adds noise to ground truth boxes
    2. Reverse diffusion: Learns to denoise boxes back to ground truth
    3. DDIM sampling: Fast inference with fewer denoising steps
    4. Dynamic head: Processes boxes at each denoising step
    
    Architecture Overview:
    - Backbone: Feature extraction (e.g., ResNet, FPN)
    - Dynamic Head: Multi-scale feature processing with timestep conditioning
    - Diffusion Process: Noise schedule and sampling procedures
    - Loss Function: Hungarian matching with classification and box regression losses
    
    Training Process:
    1. Sample random timestep t
    2. Add noise to ground truth boxes: x_t = √α̅_t * x_0 + √(1-α̅_t) * ε
    3. Predict original boxes x_0 from noisy boxes x_t
    4. Compute loss between predicted and ground truth boxes
    
    Inference Process:
    1. Start with random noise boxes
    2. Iteratively denoise using learned reverse process
    3. Apply post-processing (NMS, score thresholding)
    
    Reference:
        Paper: "DiffusionDet: Diffusion Model for Object Detection"
        Authors: Shoufa Chen, Peize Sun, Yibing Song, Ping Luo
        arXiv: https://arxiv.org/abs/2211.09788
        GitHub: https://github.com/ShoufaChen/DiffusionDet
    """

    def __init__(self, cfg):
        """
        Initialize DiffusionDet model with configuration parameters.
        
        Sets up the complete DiffusionDet pipeline including backbone, diffusion schedules,
        dynamic head, and loss computation components.
        
        Args:
            cfg: Configuration object containing model hyperparameters including:
                - MODEL.DiffusionDet.NUM_CLASSES: Number of object classes
                - MODEL.DiffusionDet.NUM_PROPOSALS: Number of proposal boxes
                - MODEL.DiffusionDet.HIDDEN_DIM: Hidden dimension for transformer
                - MODEL.DiffusionDet.SAMPLE_STEP: Number of DDIM sampling steps
                - MODEL.DiffusionDet.SNR_SCALE: Signal-to-noise ratio scaling factor
                - Loss weights and training parameters
        
        Diffusion Parameters Initialized:
        - β_t: Noise schedule (cosine schedule)
        - α_t: 1 - β_t 
        - α̅_t: Cumulative product of α values
        - Various derived quantities for efficient computation
        
        Reference:
            Configuration follows DiffusionDet paper specifications and
            repository implementation patterns.
        """
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        # Model architecture parameters
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES
        self.num_proposals = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS


        embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
        
        # Build Backbone for feature extraction
        self.backbone = SimpleFeaturePyramid(
            net=EVA02_ViT(
                img_size=1024,
                patch_size=16,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                drop_path_rate=dp,
                window_size=14,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                window_block_indexes=[
                    # 2, 5, 8 11 for global attention
                    0,
                    1,
                    3,
                    4,
                    6,
                    7,
                    9,
                    10,
                ],
                residual_block_indexes=[],
                use_rel_pos=True,
                out_feature="last_feat",
            ),
            in_feature="last_feat",
            out_channels=256,
            scale_factors=(2.0, 1.0, 0.5),  # (4.0, 2.0, 1.0, 0.5) in ViTDet
            top_block=LastLevelMaxPool(),
            norm="LN",
            square_pad=1024,
        )
        
        self.size_divisibility = self.backbone.size_divisibility

        # Diffusion process parameters
        timesteps = 1000  # Standard DDPM timesteps
        sampling_timesteps = cfg.MODEL.DiffusionDet.SAMPLE_STEP
        self.objective = 'pred_x0'  # Predict x_0 directly (vs. pred_noise)
        
        # Create cosine noise schedule β_t
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas  # α_t = 1 - β_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # α̅_t = ∏_{i=1}^t α_i
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # DDIM sampling configuration
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.  # η parameter for DDIM (1.0 = DDPM)
        self.self_condition = False  # Self-conditioning as in some diffusion variants
        
        # DiffusionDet specific parameters
        self.scale = cfg.MODEL.DiffusionDet.SNR_SCALE  # Box coordinate scaling
        self.box_renewal = True  # Filter and replenish boxes during sampling
        self.use_ensemble = True  # Ensemble predictions across timesteps

        # Register diffusion schedule buffers (computed once, used throughout training/inference)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Pre-computed values for efficient forward diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Pre-computed values for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head for processing boxes with timestep conditioning
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())
        
        # Loss configuration parameters
        class_weight = cfg.MODEL.DiffusionDet.CLASS_WEIGHT
        giou_weight = cfg.MODEL.DiffusionDet.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DiffusionDet.L1_WEIGHT
        no_object_weight = cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.use_nms = cfg.MODEL.DiffusionDet.USE_NMS

        # Build Hungarian matcher and loss criterion
        matcher = HungarianMatcherDynamicK(
            cfg=cfg, cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal
        )
        
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,)

        # Image normalization parameters
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def predict_noise_from_start(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        Predict noise ε from current noisy boxes x_t and predicted clean boxes x_0.
        
        This function implements the noise prediction formula derived from the forward
        diffusion process. Given x_t and x_0, we can analytically compute the noise
        that was added during the forward process.
        
        Args:
            x_t (torch.Tensor): Noisy boxes at timestep t, shape (batch_size, num_proposals, 4)
            t (torch.Tensor): Timestep indices, shape (batch_size,)
            x0 (torch.Tensor): Predicted clean boxes, shape (batch_size, num_proposals, 4)
            
        Returns:
            torch.Tensor: Predicted noise ε, shape (batch_size, num_proposals, 4)
            
        Mathematical Derivation:
            From the forward process: x_t = √α̅_t * x_0 + √(1-α̅_t) * ε
            Solving for ε: ε = (x_t - √α̅_t * x_0) / √(1-α̅_t)
            Rearranged: ε = (1/√(1-α̅_t)) * x_t - (√α̅_t/√(1-α̅_t)) * x_0
            
        Reference:
            Standard DDPM formulation, see Ho et al. (2020) and DiffusionDet paper.
        """
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats: List[torch.Tensor], images_whwh: torch.Tensor, 
                         x: torch.Tensor, t: torch.Tensor, x_self_cond: Optional[torch.Tensor] = None, 
                         clip_x_start: bool = False) -> Tuple[ModelPrediction, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass of the denoising network to predict clean boxes from noisy inputs.
        
        This is the core denoising function that takes noisy boxes at timestep t and
        predicts the original clean boxes x_0. The model is trained to reverse the
        forward diffusion process by learning the mapping from (x_t, t) → x_0.
        
        Args:
            backbone_feats (List[torch.Tensor]): Multi-scale feature maps from backbone
            images_whwh (torch.Tensor): Image dimensions [width, height, width, height], shape (batch_size, 4)
            x (torch.Tensor): Noisy boxes in normalized coordinates, shape (batch_size, num_proposals, 4)
            t (torch.Tensor): Timestep indices, shape (batch_size,)
            x_self_cond (Optional[torch.Tensor]): Self-conditioning input (unused in current implementation)
            clip_x_start (bool): Whether to clip predicted x_0 to valid range
            
        Returns:
            Tuple containing:
                - ModelPrediction: Named tuple with pred_noise and pred_x_start
                - List[torch.Tensor]: Classification predictions from all decoder layers
                - List[torch.Tensor]: Box coordinate predictions from all decoder layers
                
        Process Flow:
        1. Convert normalized boxes to absolute coordinates for dynamic head
        2. Run dynamic head with timestep conditioning
        3. Convert predicted absolute boxes back to normalized coordinates
        4. Apply coordinate scaling and clamping
        5. Compute corresponding noise prediction
        
        Mathematical Background:
            The model learns f_θ(x_t, t) ≈ x_0, where:
            - x_t = √α̅_t * x_0 + √(1-α̅_t) * ε (forward process)
            - The goal is to recover x_0 given x_t and t
            
        Reference:
            DiffusionDet paper Section 3.2: "Diffusion Model for Box Detection"
            GitHub: DiffusionDet/models/diffusiondet.py
        """
        # Convert normalized coordinates to absolute coordinates for processing
        x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2  # Scale from [-scale, scale] to [0, 1]
        x_boxes = box_cxcywh_to_xyxy(x_boxes)  # Convert to (x1, y1, x2, y2) format
        x_boxes = x_boxes * images_whwh[:, None, :]  # Scale to image dimensions

        # Forward pass through dynamic head with timestep conditioning
        outputs_class, outputs_coord = self.head(backbone_feats, x_boxes, t, None)

        # Process final layer predictions to get x_0 (clean boxes)
        x_start = outputs_coord[-1]  # Shape: (batch, num_proposals, 4) in absolute coordinates
        x_start = x_start / images_whwh[:, None, :]  # Normalize to [0, 1]
        x_start = box_xyxy_to_cxcywh(x_start)  # Convert to (cx, cy, w, h) format
        x_start = (x_start * 2 - 1.) * self.scale  # Scale to [-scale, scale]
        x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
        
        # Compute corresponding noise prediction using analytical formula
        pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord

    @torch.no_grad()
    def ddim_sample(self, batched_inputs: List[dict], backbone_feats: List[torch.Tensor], 
                   images_whwh: torch.Tensor, images: ImageList, clip_denoised: bool = True, 
                   do_postprocess: bool = True) -> List[dict]:
        """
        DDIM (Denoising Diffusion Implicit Models) sampling for fast inference.
        
        DDIM enables faster sampling by using fewer denoising steps while maintaining
        high sample quality. Instead of the full 1000-step DDPM process, DDIM can
        generate high-quality detections in as few as 4-10 steps.
        
        The key insight is that DDIM uses a deterministic sampling process that
        directly jumps between non-consecutive timesteps, significantly reducing
        inference time without major quality loss.
        
        Args:
            batched_inputs (List[dict]): Batch of input images and metadata
            backbone_feats (List[torch.Tensor]): Multi-scale backbone features
            images_whwh (torch.Tensor): Image dimensions, shape (batch_size, 4)
            images (ImageList): Processed image tensors
            clip_denoised (bool): Whether to clip denoised predictions to valid range
            do_postprocess (bool): Whether to apply post-processing (resize, etc.)
            
        Returns:
            List[dict]: Detection results with 'instances' key containing:
                - pred_boxes: Detected bounding boxes
                - scores: Confidence scores  
                - pred_classes: Predicted class labels
                
        DDIM Sampling Process:
        1. Start with random noise: x_T ~ N(0, I)
        2. For each reverse timestep (T-1, T-2, ..., 0):
           a. Predict x_0 from current x_t using denoising network
           b. Compute x_{t-1} using DDIM update rule
           c. Optional: Apply box renewal (filter low-quality boxes)
        3. Final x_0 becomes the detection result
        4. Apply ensemble and NMS if configured
        
        Mathematical Formulation (DDIM):
            x_{t-1} = √α_{t-1} * pred_x_0 + √(1-α_{t-1}-σ_t²) * pred_noise + σ_t * ε
            where σ_t = η * √((1-α_{t-1})/(1-α_t)) * √(1-α_t/α_{t-1})
            
        Box Renewal Strategy:
            During sampling, boxes with low confidence are filtered out and replaced
            with fresh random boxes, helping maintain diversity and improve detection
            of hard-to-detect objects.
            
        Reference:
            DDIM paper: "Denoising Diffusion Implicit Models" by Song et al. (2021)
            DiffusionDet adaptation: Section 3.3 "Box Renewal" 
            GitHub: DiffusionDet/models/diffusiondet.py - ddim_sample method
        """
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # Create DDIM timestep schedule: [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # Initialize with random noise boxes
        img = torch.randn(shape, device=self.device)

        # For ensemble: collect predictions from multiple timesteps
        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        
        # Reverse diffusion process
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            # Predict clean boxes and noise
            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            # Box renewal: filter out low-confidence predictions and maintain diversity
            if self.box_renewal:
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                # Keep only high-confidence predictions
                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
                
            # Final step: return clean boxes
            if time_next < 0:
                img = x_start
                continue

            # DDIM update rule
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            # Core DDIM update equation
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            # Replenish filtered boxes with random noise
            if self.box_renewal:
                img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
                
            # Collect ensemble predictions
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        # Process final results
        if self.use_ensemble and self.sampling_timesteps > 1:
            # Combine ensemble predictions
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            
            # Apply NMS to ensemble results
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            # Create result instance
            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            # Use single final prediction
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            
        # Apply post-processing if requested
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: add noise to clean boxes according to noise schedule.
        
        This function implements the forward diffusion process q(x_t | x_0) that
        gradually adds Gaussian noise to ground truth boxes. During training,
        we sample random timesteps and noise levels to create training targets
        for the reverse denoising process.
        
        Args:
            x_start (torch.Tensor): Clean ground truth boxes, shape (batch_size, num_proposals, 4)
            t (torch.Tensor): Timestep indices, shape (batch_size,)
            noise (Optional[torch.Tensor]): Gaussian noise to add. If None, samples fresh noise.
            
        Returns:
            torch.Tensor: Noisy boxes x_t at timestep t, same shape as x_start
            
        Mathematical Formulation:
            q(x_t | x_0) = N(x_t; √α̅_t * x_0, (1 - α̅_t) * I)
            
            Sampling formula:
            x_t = √α̅_t * x_0 + √(1 - α̅_t) * ε, where ε ~ N(0, I)
            
        Key Properties:
            - As t → 0: x_t ≈ x_0 (minimal noise)
            - As t → T: x_t ≈ N(0, I) (pure noise)
            - Allows direct sampling at any timestep t without iterating
            
        Reference:
            DDPM paper Equation (4): "Denoising Diffusion Probabilistic Models"
            DiffusionDet adapts this for bounding box coordinates
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def forward(self, batched_inputs: List[dict], do_postprocess: bool = True) -> Union[List[dict], dict]:
        """
        Main forward pass for both training and inference modes of DiffusionDet.
        
        This method orchestrates the complete DiffusionDet pipeline, handling both training
        and inference workflows. During training, it applies forward diffusion to ground truth
        boxes and trains the denoising network. During inference, it uses DDIM sampling to
        generate object detections from pure noise.
        
        The key innovation of DiffusionDet is treating object detection as a denoising problem:
        instead of directly predicting bounding boxes, the model learns to iteratively refine
        noisy boxes into precise object locations through a learned reverse diffusion process.
    
        Args:
            batched_inputs (List[dict]): Batch of input samples from DatasetMapper, each containing:
                Training mode:
                    - 'image': Input image tensor, shape (C, H, W)
                    - 'instances': Ground truth Instances with gt_boxes and gt_classes
                Inference mode:
                    - 'image': Input image tensor, shape (C, H, W) 
                    - 'height', 'width': Original image dimensions for post-processing
                    
            do_postprocess (bool, optional): Whether to apply post-processing during inference
                (resize detections to original image size). Defaults to True.
                
        Returns:
            Training mode (dict): Loss dictionary containing:
                - 'loss_ce': Classification loss (cross-entropy or focal loss)
                - 'loss_bbox': L1 bounding box regression loss
                - 'loss_giou': Generalized IoU loss for better localization
                - Auxiliary losses if deep supervision is enabled (loss_ce_0, loss_bbox_0, etc.)
                
            Inference mode (List[dict]): List of detection results, one per image:
                - Each dict contains 'instances' key with detected boxes, scores, and classes
                
        Training Workflow:
        1. **Feature Extraction**: Backbone network extracts multi-scale features
        2. **Forward Diffusion**: Apply noise to ground truth boxes at random timestep t
        3. **Denoising Prediction**: Dynamic head predicts clean boxes from noisy input
        4. **Loss Computation**: Hungarian matching between predictions and ground truth
        
        Inference Workflow:  
        1. **Feature Extraction**: Same backbone feature extraction
        2. **DDIM Sampling**: Iteratively denoise random boxes over multiple timesteps
        3. **Post-processing**: Apply NMS, score filtering, and coordinate transformation
        
        Mathematical Foundation:
        Training objective learns to reverse the forward diffusion process:
            Forward: q(x_t | x_0) = N(x_t; √α̅_t x_0, (1-α̅_t)I)
            Reverse: p_θ(x_{t-1} | x_t) ≈ N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
            
        The model learns f_θ(x_t, t) → x_0, predicting clean boxes directly rather than noise.
        
        Architecture Flow:
        ```
        Input Images → Backbone Features → Dynamic Head → Predictions
                                      ↗               ↗
                              Diffused Boxes    Timestep t
        ```
        
        Key Innovations:
        - **Box-level diffusion**: Applies diffusion directly to bounding box coordinates
        - **Timestep conditioning**: Dynamic head receives timestep embedding
        - **Box renewal**: During inference, filters low-quality boxes and replenishes
        - **Ensemble sampling**: Combines predictions across multiple timesteps
        
        Reference:
            Paper: "DiffusionDet: Diffusion Model for Object Detection" 
            Section 3: "Method" - describes the complete training and inference pipeline
            GitHub: https://github.com/ShoufaChen/DiffusionDet/blob/main/diffusiondet/diffusiondet.py
        """
        # Preprocess input images: normalization, padding, batching
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction using backbone network (e.g., ResNet-50 + FPN)
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Inference Mode: DDIM sampling for detection generation
        if not self.training:
            results = self.ddim_sample(batched_inputs, features, images_whwh, images)
            return results

        # Training Mode: Forward diffusion + denoising network training
        if self.training:
            # Extract ground truth instances and prepare diffusion targets
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets, x_boxes, noises, t = self.prepare_targets(gt_instances)
            t = t.squeeze(-1)  # Remove extra dimension from timesteps
            
            # Convert normalized boxes to absolute coordinates for dynamic head
            x_boxes = x_boxes * images_whwh[:, None, :]

            # Forward pass through dynamic head with timestep conditioning
            outputs_class, outputs_coord = self.head(features, x_boxes, t, None)
            
            # Prepare output dictionary for loss computation
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

            # Add auxiliary outputs for deep supervision (multi-layer losses)
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            # Compute Hungarian matching loss between predictions and ground truth
            loss_dict = self.criterion(output, targets)
            
            # Apply loss weights as specified in configuration
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

    def prepare_diffusion_repeat(self, gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare diffusion training samples using ground truth box repetition strategy.
        
        This method creates training proposals by repeating each ground truth box multiple
        times to reach the target number of proposals (typically 300-500). The repetition
        ensures that all training samples are anchored on actual objects, which can help
        the model learn strong object priors and improve convergence.
        
        This is an alternative to the concatenation strategy (prepare_diffusion_concat)
        and tends to provide stronger object-centric supervision at the cost of reduced
        diversity in the training distribution.
        
        Args:
            gt_boxes (torch.Tensor): Ground truth boxes in normalized (cx, cy, w, h) format,
                                   shape (num_gt_boxes, 4), values in [0, 1]
                                   
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - diff_boxes: Noisy boxes after forward diffusion in (x1, y1, x2, y2) format,
                            shape (num_proposals, 4)
                - noise: Gaussian noise tensor used in diffusion process, 
                        shape (num_proposals, 4)
                - t: Sampled timestep for this training sample, shape (1,)
                
        Algorithm Steps:
        1. **Handle Empty Ground Truth**: Create placeholder box if no ground truth exists
        2. **Calculate Repetition Pattern**: Distribute boxes to reach exact num_proposals
        3. **Apply Repetition**: Use torch.repeat_interleave with computed pattern
        4. **Random Shuffling**: Shuffle repetition pattern to avoid spatial clustering
        5. **Coordinate Scaling**: Transform from [0,1] to [-scale, scale] range
        6. **Forward Diffusion**: Apply q(x_t | x_0) with sampled timestep and noise
        7. **Final Processing**: Clamp, rescale, and convert to (x1,y1,x2,y2) format
        
        Repetition Strategy Details:
            - Base repetitions per box: num_proposals // num_gt_boxes
            - Remainder distribution: Some boxes get +1 extra repetition
            - Random shuffling prevents spatial bias in repeated boxes
            - Ensures exactly num_proposals total boxes
            
        Mathematical Process:
            1. Sample t ~ Uniform(0, T-1) and ε ~ N(0, I)
            2. Scale boxes: x_0 = (gt_boxes * 2 - 1) * scale  # [0,1] → [-scale,scale]
            3. Repeat boxes: x_start = repeat_interleave(x_0, repeat_pattern)
            4. Forward diffusion: x_t = √α̅_t * x_start + √(1-α̅_t) * ε
            5. Clamp and rescale: x_t ∈ [0, 1]
            6. Convert format: (cx,cy,w,h) → (x1,y1,x2,y2)
            
        Use Cases:
            - When strong object supervision is desired
            - For datasets with few objects per image
            - When model struggles with object detection fundamentals
            - As ablation study comparing repetition vs concatenation strategies
            
        Coordinate Transformations:
        ```
        Input: (cx, cy, w, h) ∈ [0, 1]
        → Scale: (cx, cy, w, h) ∈ [-scale, scale]  
        → Diffuse: Add noise according to timestep
        → Clamp: Ensure valid coordinate range
        → Convert: (x1, y1, x2, y2) format for processing
        ```
        
        Reference:
            DiffusionDet paper Section 3.1: "Box Generation Process"
            Alternative strategy mentioned in implementation details
            GitHub: DiffusionDet/diffusiondet/diffusiondet.py - prepare_diffusion_repeat
        """
        # Sample random timestep t from [0, T-1] and generate noise
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        
        # Handle edge case: no ground truth boxes in image
        if not num_gt:  
            # Create placeholder box at image center with full size
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        # Calculate how many times to repeat each box to reach num_proposals
        num_repeat = self.num_proposals // num_gt  # Base repetitions per box
        
        # Handle remainder: some boxes get +1 extra repetition
        # First (num_gt - remainder) boxes get num_repeat repetitions
        # Last (remainder) boxes get (num_repeat + 1) repetitions
        remainder = self.num_proposals % num_gt
        repeat_tensor = [num_repeat] * (num_gt - remainder) + [num_repeat + 1] * remainder
        assert sum(repeat_tensor) == self.num_proposals  # Verify exact count
        
        # Shuffle repetition pattern to avoid spatial clustering of same boxes
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        # Transform coordinates from [0,1] normalized to [-scale, scale] diffusion range
        gt_boxes = (gt_boxes * 2. - 1.) * self.scale
        
        # Repeat each box according to calculated pattern
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        # Apply forward diffusion process: x_t = √α̅_t * x_0 + √(1-α̅_t) * ε
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Clamp to valid range and transform back to [0,1]
        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2) format
        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_diffusion_concat(self, gt_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare diffusion training samples using box concatenation strategy.
        
        This method creates training proposals by combining ground truth boxes with
        randomly generated placeholder boxes to reach the target number of proposals.
        This is the preferred strategy in DiffusionDet as it balances ground truth
        supervision with exploration of the full coordinate space.
        
        The concatenation approach provides more diverse training samples compared to
        the repetition strategy, helping the model generalize better to various object
        configurations while still maintaining strong supervision from ground truth boxes.
        
        Args:
            gt_boxes (torch.Tensor): Ground truth boxes in normalized (cx, cy, w, h) format,
                                   shape (num_gt_boxes, 4), values in [0, 1]
                                   
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - diff_boxes: Noisy boxes after forward diffusion in (x1, y1, x2, y2) format,
                            shape (num_proposals, 4)
                - noise: Gaussian noise tensor used in diffusion process,
                        shape (num_proposals, 4)  
                - t: Sampled timestep for this training sample, shape (1,)
                
        Algorithm Steps:
        1. **Sample Timestep and Noise**: Random t ∈ [0, T-1] and ε ~ N(0, I)
        2. **Handle Empty Ground Truth**: Create placeholder if no objects exist
        3. **Box Count Balancing**:
           - If num_gt < num_proposals: Add random placeholder boxes
           - If num_gt > num_proposals: Randomly subsample ground truth boxes  
           - If num_gt = num_proposals: Use all ground truth boxes
        4. **Coordinate Scaling**: Transform [0,1] → [-scale, scale]
        5. **Forward Diffusion**: Apply q(x_t | x_0) with noise
        6. **Post-processing**: Clamp, rescale, format conversion
        
        Random Placeholder Box Generation:
            - Centers (cx, cy): Gaussian N(0.5, (1/6)²) - centered around image middle
            - Sizes (w, h): Gaussian N(0.5, (1/6)²) with minimum clipping
            - Statistics: 3σ ≈ 0.5, so ~99.7% of boxes within valid [0,1] range
            - Size clipping: Ensures positive dimensions (width, height > 1e-4)
            
        Three Handling Cases:
        
        **Case 1: num_gt < num_proposals** (Most common)
        ```python
        placeholders = generate_random_boxes(num_proposals - num_gt)
        x_start = concat(gt_boxes, placeholders)
        ```
        
        **Case 2: num_gt > num_proposals** (Dense scenes)
        ```python
        mask = random_selection(num_proposals out of num_gt)  
        x_start = gt_boxes[mask]
        ```
        
        **Case 3: num_gt = num_proposals** (Exact match)
        ```python
        x_start = gt_boxes  # Use all ground truth
        ```
        
        Mathematical Process:
            1. Generate proposals: x_start ∈ [0, 1]^(num_proposals, 4)
            2. Scale to diffusion space: x_start = (x_start * 2 - 1) * scale
            3. Sample timestep: t ~ Uniform(0, T-1)
            4. Forward diffusion: x_t = √α̅_t * x_start + √(1-α̅_t) * ε
            5. Clamp: x_t ∈ [-scale, scale]
            6. Rescale: x_t ∈ [0, 1]  
            7. Format: (cx,cy,w,h) → (x1,y1,x2,y2)
            
        Statistical Properties:
            - Ground truth boxes: Perfect object supervision
            - Random boxes: Explore full coordinate space  
            - Gaussian distribution: Natural sampling of plausible locations
            - Size clipping: Prevents degenerate zero-area boxes
            
        Advantages over Repetition:
            ✓ More diverse training distribution
            ✓ Better generalization to unseen configurations
            ✓ Natural handling of variable object counts
            ✓ Balanced exploration vs exploitation
            
        Reference:
            DiffusionDet paper Section 3.1: "Box Generation Process"
            Primary strategy used in all experiments
            GitHub: DiffusionDet/diffusiondet/diffusiondet.py - prepare_diffusion_concat
        """
        # Sample random timestep and generate noise for forward diffusion
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        
        # Handle edge case: no ground truth objects in image
        if not num_gt:
            # Create single placeholder box at image center
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        # Balance ground truth and random boxes to reach target num_proposals
        if num_gt < self.num_proposals:
            # Case 1: Add random placeholder boxes
            num_placeholders = self.num_proposals - num_gt
            
            # Generate random boxes with Gaussian distribution around image center
            # σ = 1/6 ensures 3σ = 0.5, so ~99.7% of values in [0,1] after adding 0.5
            box_placeholder = torch.randn(num_placeholders, 4, device=self.device) / 6. + 0.5
            
            # Ensure positive box dimensions (width and height must be > 0)
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            
            # Concatenate ground truth with placeholders
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
            
        elif num_gt > self.num_proposals:
            # Case 2: Randomly subsample ground truth boxes
            # Create random selection mask and shuffle for unbiased sampling
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
            
        else:
            # Case 3: Perfect match - use all ground truth boxes
            x_start = gt_boxes

        # Transform coordinates from [0,1] normalized to [-scale, scale] diffusion space
        x_start = (x_start * 2. - 1.) * self.scale

        # Apply forward diffusion process: x_t = √α̅_t * x_0 + √(1-α̅_t) * ε
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Clamp to valid diffusion range and rescale back to [0,1]
        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2) format for dynamic head
        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_targets(self, targets: List[Instances]) -> Tuple[List[dict], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare comprehensive training targets for DiffusionDet by processing ground truth instances.
        
        This function serves as the main preprocessing pipeline for training data, converting
        raw ground truth annotations into the format required by DiffusionDet's loss functions.
        It applies forward diffusion to create noisy box inputs and prepares target dictionaries
        compatible with Hungarian matching and set-based losses.
        
        The function handles coordinate transformations, format conversions, and creates all
        necessary target fields for computing classification, localization, and auxiliary losses
        during the training process.
        
        Args:
            targets (List[Instances]): List of ground truth instances, one per image in batch.
                Each Instances object contains:
                - gt_classes (torch.Tensor): Class labels, shape (num_objects,)
                - gt_boxes (Boxes): Ground truth bounding boxes in absolute coordinates
                - image_size (Tuple[int, int]): Original image dimensions (height, width)
                
        Returns:
            Tuple[List[dict], torch.Tensor, torch.Tensor, torch.Tensor]:
                - new_targets: List of target dictionaries for Hungarian matcher, one per image
                - diffused_boxes: Noisy boxes after forward diffusion, shape (batch_size, num_proposals, 4)
                - noises: Noise tensors used in diffusion, shape (batch_size, num_proposals, 4)
                - ts: Timestep indices for each image, shape (batch_size, 1)
                
        Target Dictionary Structure:
            Each target dictionary contains the following fields for Hungarian matching:
            
            **Core Targets:**
            - 'labels': Ground truth class indices, shape (num_gt_objects,)
            - 'boxes': Normalized boxes in (cx, cy, w, h) format, shape (num_gt_objects, 4)
            
            **Coordinate References:**
            - 'boxes_xyxy': Original boxes in absolute (x1, y1, x2, y2) format
            - 'image_size_xyxy': Image dimensions as [width, height, width, height]
            - 'image_size_xyxy_tgt': Tiled image dimensions for broadcasting with boxes
            
            **Auxiliary Information:**
            - 'area': Ground truth box areas for loss weighting
            
        Processing Pipeline:
        1. **Extract Metadata**: Image dimensions and create coordinate scaling tensors
        2. **Coordinate Normalization**: Convert absolute boxes to [0,1] normalized coordinates  
        3. **Format Conversion**: Transform (x1,y1,x2,y2) → (cx,cy,w,h) center format
        4. **Apply Diffusion**: Use prepare_diffusion_concat to create noisy training inputs
        5. **Create Target Dict**: Package all required fields for loss computation
        6. **Batch Assembly**: Stack results across all images in batch
        
        Coordinate Transformation Chain:
        ```
        Input: (x1, y1, x2, y2) in pixels
        → Normalize: (x1/W, y1/H, x2/W, y2/H) ∈ [0,1]
        → Center format: (cx, cy, w, h) ∈ [0,1]  
        → Diffusion: Add noise → create training input boxes
        ```
        
        Hungarian Matching Integration:
            The target dictionaries are designed for SetCriterionDynamicK which uses
            Hungarian algorithm to find optimal assignment between predictions and
            ground truth objects. Key requirements:
            - Consistent coordinate formats between predictions and targets
            - Area information for loss weighting (small vs large objects)
            - Image size information for coordinate denormalization
            
        Memory and Efficiency:
            - Processes entire batch simultaneously for efficiency
            - Reuses coordinate transformation tensors across objects
            - Minimizes GPU memory transfers by batching operations
            
        Error Handling:
            - Empty ground truth: handled by prepare_diffusion_concat with placeholder
            - Variable object counts: natural handling through list-based targets
            - Coordinate validation: ensured through normalization and clamping
            
        Reference:
            DiffusionDet paper Section 3.2: "Training Objective"
            Sets up targets for Equation (3) loss computation
            GitHub: DiffusionDet/diffusiondet/diffusiondet.py - prepare_targets
        """
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        
        # Process each image in the batch independently
        for targets_per_image in targets:
            target = {}
            
            # Extract image dimensions and create coordinate scaling tensor
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            
            # Extract ground truth annotations
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy  # Normalize to [0,1]
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)  # Convert to (cx, cy, w, h) format
            
            # Apply forward diffusion to create noisy training input
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            
            # Prepare target dictionary for Hungarian matching loss
            
            # Core target information
            target["labels"] = gt_classes.to(self.device)  # Class indices
            target["boxes"] = gt_boxes.to(self.device)     # Normalized (cx,cy,w,h) boxes
            
            # Coordinate reference information  
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device)  # Original absolute coordinates
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)              # [W,H,W,H] for scaling
            
            # Create tiled image size tensor for broadcasting operations
            # Shape: (num_gt_objects, 4) - same image size repeated for each object
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            
            # Additional information for loss computation
            target["area"] = targets_per_image.gt_boxes.area().to(self.device)  # Box areas for weighting
            
            new_targets.append(target)

        # Stack batch results for efficient processing
        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)

    def inference(self, box_cls: torch.Tensor, box_pred: torch.Tensor, 
                 image_sizes: List[torch.Size]) -> Union[List[Instances], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Convert raw model predictions into final detection results with comprehensive post-processing.
        
        This function transforms the raw classification logits and bounding box coordinates
        from the DiffusionDet model into properly formatted detection instances. It handles
        different loss formulations (focal vs softmax), applies Non-Maximum Suppression (NMS)
        when configured, and supports both regular inference and ensemble modes.
        
        The function is crucial for bridging the gap between the model's internal representations
        and the final object detection outputs expected by downstream applications.
        
        Args:
            box_cls (torch.Tensor): Classification predictions with shape (batch_size, num_proposals, K)
                - For focal/FED loss: Raw logits for each class
                - For standard loss: Logits including "no object" class at index -1
            box_pred (torch.Tensor): Bounding box predictions in absolute coordinates (x1,y1,x2,y2),
                                   shape (batch_size, num_proposals, 4)
            image_sizes (List[torch.Size]): Original image dimensions for each image in batch
            
        Returns:
            Regular inference mode (List[Instances]): 
                Detection results, one Instances object per image containing:
                - pred_boxes (Boxes): Final detected bounding boxes  
                - scores (torch.Tensor): Confidence scores for each detection
                - pred_classes (torch.Tensor): Predicted class labels
                
            Ensemble mode (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                Raw predictions for ensemble combination:
                - box_pred_per_image: Box coordinates, shape (num_detections, 4)
                - scores_per_image: Confidence scores, shape (num_detections,)
                - labels_per_image: Class labels, shape (num_detections,)
                
        Processing Pipeline:
        
        **Focal/FED Loss Path:**
        1. **Score Activation**: Apply sigmoid to get class probabilities
        2. **Label Expansion**: Create label tensor for all class-proposal combinations  
        3. **Top-K Selection**: Select highest scoring detections across all classes
        4. **Box Expansion**: Replicate box predictions for multi-class scoring
        5. **Post-processing**: Apply NMS and create Instances
        
        **Standard Softmax Path:**
        1. **Score Computation**: Apply softmax excluding "no object" class
        2. **Class Selection**: Take argmax for labels, max for scores
        3. **Post-processing**: Apply NMS and create Instances
        
        Focal Loss Processing Details:
        ```python
        scores = sigmoid(box_cls)  # Shape: (batch, proposals, classes)
        scores_flat = scores.flatten()  # Shape: (batch * proposals * classes)
        top_scores, top_indices = scores_flat.topk(num_proposals)
        # Map indices back to original class labels and box predictions
        ```
        
        Standard Loss Processing Details:
        ```python  
        probs = softmax(box_cls[:, :, :-1])  # Exclude "no object" class
        scores, labels = probs.max(dim=-1)   # Best class per proposal
        ```
        
        Non-Maximum Suppression (NMS):
            When enabled, applies batched NMS with IoU threshold 0.5 to remove
            redundant overlapping detections. Uses efficient CUDA implementation
            from Detectron2 for optimal performance.
            
        Ensemble Mode Behavior:
            When use_ensemble=True and sampling_timesteps > 1, returns raw predictions
            without creating Instances objects. This allows ddim_sample to collect
            predictions from multiple timesteps for ensemble combination.
            
        Class Label Handling:
            - Focal loss: Labels range [0, num_classes-1], all classes considered
            - Standard loss: Labels range [0, num_classes-1], "no object" excluded
            - Multi-class scoring: Each proposal can be assigned to any class
            
        Memory Optimization:
            - Processes images individually to handle variable sizes efficiently
            - Uses in-place operations where possible to reduce memory footprint
            - Efficient tensor indexing for top-k selection and NMS filtering
            
        Performance Considerations:
            - Focal loss path requires more computation but better handles class imbalance
            - Standard softmax path is faster but may struggle with rare classes
            - NMS adds computational cost but significantly improves precision
            - Ensemble mode trades inference speed for improved accuracy
            
        Error Handling and Edge Cases:
            - Empty predictions: Handled gracefully with empty Instances
            - Invalid coordinates: Prevented by model clamping and normalization
            - Variable batch sizes: Natural support through list-based processing
            
        Integration with DDIM Sampling:
            During iterative denoising, this function processes intermediate predictions
            and can return raw outputs for ensemble combination across timesteps.
            Final timestep uses full post-processing pipeline.
            
        Reference:
            DiffusionDet paper Section 3.3: "Inference and Box Renewal"
            Standard object detection post-processing adapted for diffusion models
            GitHub: DiffusionDet/diffusiondet/diffusiondet.py - inference method
        """
        assert len(box_cls) == len(image_sizes), "Batch size mismatch between predictions and image sizes"
        results = []

        # Path 1: Focal Loss or FED Loss Processing
        if self.use_focal or self.use_fed_loss:
            # Apply sigmoid activation to get class probabilities [0,1]
            scores = torch.sigmoid(box_cls)  # Shape: (batch_size, num_proposals, num_classes)
            
            # Create label tensor: [0, 1, 2, ..., num_classes-1] for each proposal
            # This enables multi-class scoring where each proposal-class pair gets a score
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            # Process each image in the batch independently
            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                
                # Flatten scores across proposals and classes, then select top-k
                # This allows competition between all proposal-class combinations
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(
                    self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                
                # Expand box predictions to match the flattened score structure
                # Each box is repeated for each class, then indexed by topk_indices
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                # Return raw predictions for ensemble mode (DDIM sampling)
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                # Apply Non-Maximum Suppression to remove overlapping detections
                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                # Create final detection instance for this image
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # Path 2: Standard Softmax-based Classification
            # Apply softmax and select best class (excluding "no object" class at index -1)
            # The [:, :, :-1] slice removes the "no object" class from consideration
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            # Process each image in the batch independently  
            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                # Return raw predictions for ensemble mode (DDIM sampling)
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                # Apply Non-Maximum Suppression to remove overlapping detections
                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                    
                # Create final detection instance for this image
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs: List[dict]) -> Tuple[ImageList, torch.Tensor]:
        """
        Preprocess input images for DiffusionDet with normalization, padding, and dimension extraction.
        
        This function performs essential preprocessing steps to prepare raw input images for
        the DiffusionDet model. It handles normalization using ImageNet statistics, creates
        properly padded batches respecting backbone constraints, and extracts dimension
        information needed for coordinate transformations throughout the detection pipeline.
        
        The preprocessing ensures that images are in the correct format and scale expected
        by the backbone network while preserving the information needed to transform
        predictions back to the original image coordinate system.
        
        Args:
            batched_inputs (List[dict]): List of input dictionaries from DatasetMapper, each containing:
                - 'image': Raw image tensor in (C, H, W) format, typically uint8 values [0, 255]
                - 'height': Original image height before any preprocessing (optional)
                - 'width': Original image width before any preprocessing (optional)
                - Other metadata fields for downstream processing
                
        Returns:
            Tuple[ImageList, torch.Tensor]:
                - ImageList: Detectron2 ImageList object containing:
                    - .tensor: Batched image tensor, shape (batch_size, C, H_padded, W_padded)
                    - .image_sizes: List of original (H, W) tuples before padding
                - torch.Tensor: Image dimensions in [W, H, W, H] format for coordinate scaling,
                              shape (batch_size, 4)
                              
        Preprocessing Pipeline:
        
        **1. Image Normalization:**
        ```python
        normalized = (image - pixel_mean) / pixel_std
        ```
        - Subtracts pixel mean: [0.485, 0.456, 0.406] × 255 for RGB channels
        - Divides by pixel std: [0.229, 0.224, 0.225] × 255 for RGB channels  
        - Converts from [0, 255] uint8 to normalized float32 range
        - Uses ImageNet statistics for pretrained backbone compatibility
        
        **2. Batch Creation and Padding:**
        - Creates ImageList with size_divisibility constraint (typically 32)
        - Pads smaller images to match largest dimensions in batch
        - Ensures all spatial dimensions are divisible by backbone stride
        - Minimizes padding overhead while maintaining efficient computation
        
        **3. Dimension Extraction:**
        - Records original image dimensions before padding
        - Formats as [W, H, W, H] for easy coordinate transformations
        - Enables conversion between absolute and normalized coordinates
        
        Coordinate System Support:
            The [W, H, W, H] format enables efficient coordinate transformations:
            ```python
            # Normalize absolute coordinates to [0,1]
            normalized_boxes = absolute_boxes / images_whwh
            
            # Convert back to absolute coordinates
            absolute_boxes = normalized_boxes * images_whwh
            ```
        
        Memory and Performance Optimizations:
            - Single GPU transfer for entire batch using ImageList
            - Efficient padding strategy minimizes memory overhead
            - Preserves original image sizes for accurate post-processing
            - Compatible with various backbone architectures and input sizes
            
        Size Divisibility Constraint:
            The backbone network (e.g., ResNet + FPN) requires input dimensions
            to be divisible by a specific value (typically 32) due to:
            - Multiple downsampling layers creating spatial feature pyramids
            - Need for consistent feature map sizes across different scales
            - Efficient convolution operations with standard kernel sizes
            
        Normalization Statistics:
            Uses ImageNet mean and standard deviation for consistency with
            pretrained backbone weights:
            - Mean: [123.675, 116.28, 103.53] (RGB format)
            - Std: [58.395, 57.12, 57.375] (RGB format)
            - Applied per-channel for proper color space normalization
            
        Error Handling:
            - Handles variable input image sizes gracefully
            - Ensures proper tensor shapes and types for downstream processing
            - Compatible with both training and inference modes
            - Maintains batch structure throughout processing
            
        Integration with Detection Pipeline:
            - Normalized images → Backbone → Features → Dynamic Head
            - images_whwh used throughout for coordinate transformations
            - Original sizes preserved for final result post-processing
            
        Reference:
            Standard Detectron2 preprocessing adapted for DiffusionDet
            Compatible with ResNet, ResNeXt, and other common backbones
            GitHub: DiffusionDet/diffusiondet/diffusiondet.py - preprocess_image
        """
        # Step 1: Apply normalization to each image and transfer to GPU device
        # The normalizer lambda function applies ImageNet mean/std normalization
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        
        # Step 2: Create padded ImageList respecting backbone size divisibility
        # ImageList handles batching with minimal padding while ensuring compatibility
        images = ImageList.from_tensors(images, self.size_divisibility)

        # Step 3: Extract original image dimensions for coordinate transformations
        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]  # Get height and width from original image tensor
            # Format as [width, height, width, height] for easy coordinate scaling
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        
        # Stack into batch tensor for efficient processing
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh