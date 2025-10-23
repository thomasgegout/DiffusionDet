# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)


class PositionEmbeddingSine(nn.Module):
    """
    Sine-cosine positional encoding for spatial positions.
    Similar to DINO's implementation.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, boxes):
        """
        Args:
            boxes: (batch_size, num_boxes, 4) in format (x1, y1, x2, y2)
        Returns:
            pos: (batch_size, num_boxes, num_pos_feats * 4) positional embeddings
        """
        # Extract center coordinates and dimensions
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        
        # Normalize if needed
        if self.normalize:
            cx = cx / (boxes[..., 2].max() + 1e-6)
            cy = cy / (boxes[..., 3].max() + 1e-6)
            w = w / (boxes[..., 2].max() + 1e-6)
            h = h / (boxes[..., 3].max() + 1e-6)
        
        # Create sine-cosine embeddings for each dimension
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=boxes.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        def get_embeddings(coords):
            """Generate sine-cosine embeddings for a coordinate."""
            # coords: (batch_size, num_boxes)
            # Add dimension for broadcasting: (batch_size, num_boxes, 1)
            coords = coords.unsqueeze(-1) * self.scale
            # Broadcast and divide: (batch_size, num_boxes, num_pos_feats)
            pos = coords / dim_t.unsqueeze(0).unsqueeze(0)
            # Apply sin to even indices, cos to odd indices
            pos_sin = pos[:, :, 0::2].sin()
            pos_cos = pos[:, :, 1::2].cos()
            # Interleave sin and cos
            pos = torch.stack([pos_sin, pos_cos], dim=-1).flatten(-2)
            return pos
        
        # Generate embeddings for all bbox components
        pos_cx = get_embeddings(cx)  # (batch_size, num_boxes, num_pos_feats)
        pos_cy = get_embeddings(cy)
        pos_w = get_embeddings(w)
        pos_h = get_embeddings(h)
        
        # Concatenate all positional embeddings
        pos = torch.cat([pos_cx, pos_cy, pos_w, pos_h], dim=-1)  # (batch_size, num_boxes, num_pos_feats * 4)
        return pos


class DynamicHead(nn.Module):

    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler
        
        # Build heads.
        num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES
        d_model = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD
        nhead = cfg.MODEL.DiffusionDet.NHEADS
        dropout = cfg.MODEL.DiffusionDet.DROPOUT
        activation = cfg.MODEL.DiffusionDet.ACTIVATION
        num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.num_heads = num_heads
        self.return_intermediate = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION

        # Gaussian random feature embedding layer for time
        self.d_model = d_model
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Improved Positional Encodings (DINO-style)
        num_proposals = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
        # Learned query positional embeddings
        self.query_pos_embed = nn.Embedding(num_proposals, d_model)
        # Sine-cosine spatial positional encoder
        self.pos_encoder = PositionEmbeddingSine(d_model // 4, normalize=True)
        # Projection layer to combine spatial encodings with model dimension
        self.spatial_pos_proj = nn.Linear(d_model, d_model)

        # Init parameters.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.num_classes = num_classes
        if self.use_focal or self.use_fed_loss:
            prior_prob = cfg.MODEL.DiffusionDet.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):
        
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
      
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    def get_combined_pos_encoding(self, bboxes):
        """
        Generate combined positional encodings from learned and spatial components.
        
        Args:
            bboxes: (batch_size, num_proposals, 4) bounding boxes
            
        Returns:
            combined_pos: (batch_size, num_proposals, d_model) combined positional encodings
        """
        batch_size, num_proposals = bboxes.shape[:2]
        
        # Learned query positional embeddings (num_proposals, d_model)
        query_pos = self.query_pos_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Spatial positional encodings from bbox coordinates
        spatial_pos = self.pos_encoder(bboxes)  # (batch_size, num_proposals, d_model)
        spatial_pos = self.spatial_pos_proj(spatial_pos)  # Project to d_model
        
        # Combine learned and spatial encodings
        combined_pos = query_pos + spatial_pos
        
        return combined_pos

    def forward(self, features, init_bboxes, t, init_features):
        # assert t shape (batch_size)
        time = self.time_mlp(t)

        inter_class_logits = []
        inter_pred_bboxes = []

        bs = len(features[0])
        bboxes = init_bboxes

        # Compute combined positional encodings (learned + spatial)
        pos_encodings = self.get_combined_pos_encoding(bboxes)

        if init_features is not None:
            init_features = init_features[None].repeat(1, bs, 1)
            proposal_features = init_features.clone()
        else:
            proposal_features = None
        
        for head_idx, rcnn_head in enumerate(self.head_series):
            # Pass positional encodings to each head
            class_logits, pred_bboxes, proposal_features = rcnn_head(
                features, bboxes, proposal_features, self.box_pooler, time, pos_encodings
            )
            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)
            bboxes = pred_bboxes.detach()
            # Update positional encodings for new bboxes
            pos_encodings = self.get_combined_pos_encoding(bboxes)

        if self.return_intermediate:
            return torch.stack(inter_class_logits), torch.stack(inter_pred_bboxes)

        return class_logits[None], pred_bboxes[None]


class RCNNHead(nn.Module):

    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))

        # cls.
        num_cls = cfg.MODEL.DiffusionDet.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.DiffusionDet.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)
        
        # pred.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        if self.use_focal or self.use_fed_loss:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

    def forward(self, features, bboxes, pro_features, pooler, time_emb, pos_encodings=None):
        """
        :param bboxes: (N, nr_boxes, 4)
        :param pro_features: (N, nr_boxes, d_model)
        :param pos_encodings: (N, nr_boxes, d_model) combined positional encodings
        """

        N, nr_boxes = bboxes.shape[:2]
        
        # roi_feature.
        proposal_boxes = list()
        for b in range(N):
            proposal_boxes.append(Boxes(bboxes[b]))
        roi_features = pooler(features, proposal_boxes)

        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)

        # self_att with positional encodings.
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)
        
        # Add positional encodings to queries and keys if available
        if pos_encodings is not None:
            pos_encodings = pos_encodings.permute(1, 0, 2)  # (nr_boxes, N, d_model)
            query = pro_features + pos_encodings
            key = pro_features + pos_encodings
        else:
            query = pro_features
            key = pro_features
        
        pro_features2 = self.self_attn(query, key, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # inst_interact.
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # obj_feature.
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)
        
        fc_feature = obj_features.transpose(0, 1).reshape(N * nr_boxes, -1)

        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)
        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.DiffusionDet.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.DiffusionDet.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
