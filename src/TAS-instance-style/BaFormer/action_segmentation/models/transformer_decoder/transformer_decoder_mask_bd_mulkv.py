# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from torch.nn import Conv1d
from detectron2.utils.registry import Registry
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .position_encoding import  PositionalEncoding
from .transformer import Transformer

from . import TRANSFORMER_DECODER_REGISTRY


@TRANSFORMER_DECODER_REGISTRY.register()
class TransformerDecoderMask_Boundary_MulKV(nn.Module):
    '''soft attn mask from sigmoid'''
    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_decode,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dropout: float,
            deep_supervision: bool,
            mask_dim: int,
            enforce_input_project: bool,
            layer_in_decode_block:int,
            threshold: float
    ):
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        self.num_decode = num_decode
        self.num_heads = nheads
        self.layer_in_decode_block = layer_in_decode_block
        self.threshold = threshold

        self.pe = PositionalEncoding(d_hid=in_channels, n_position=num_queries)
        # Temporal Anchor Queries: init queries at evenly spaced temporal anchors
        # t_i = i / (num_queries - 1) in [0, 1], broadcast across the hidden_dim channels
        # instead of a pure random init, so each query starts with a coarse notion of
        # "where in the video" it is responsible for (Pillar 2, proposal_imprv_baformer04).
        anchor_t = torch.linspace(0, 1, num_queries).view(1, num_queries, 1)
        base_query = anchor_t.repeat(1, 1, hidden_dim)
        self.query = nn.Parameter(base_query + 0.02 * torch.randn(1, num_queries, hidden_dim))
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_decode):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv1d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1))
            else:
                self.input_proj.append(nn.Sequential())

        self.trans_decode = nn.ModuleList([])
        for i in range(self.num_decode):                         #in_dim, head_dim, heads, hidden_dim
            self.trans_decode.append(Transformer_decoder_layer(in_channels, in_channels, nheads, hidden_dim, dropout))

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.aux_loss = deep_supervision
        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3) 

        # Pillar 1 & 2 (proposal_imprv_baformer13): Temporal ASPP Boundary Engine (T-ASPP)
        # Fuses kinematic diffs, shallow features, deep features, and query mask transition gradient.
        self.boundary_head = TemporalASPPBoundaryEngine(in_dim=mask_dim, hidden_dim=hidden_dim, dropout=dropout)


    @classmethod
    def from_config(cls, cfg,):
        ret = {}
        if cfg.model.action_seg.backbone.name is not None:
            ret["in_channels"] = cfg.model.action_seg.backbone.embed_dim
        else:
            ret["in_channels"] = cfg.model.action_seg.frame_decoder.embed_dim
        ret["mask_classification"] = True

        ret["num_classes"] = cfg.dataset.n_classes
        ret["num_decode"] = cfg.model.action_seg.transformer_decoder.dec_layers
        ret["hidden_dim"] = cfg.model.action_seg.transformer_decoder.hidden_dim
        ret["num_queries"] = cfg.dataset.num_query # cfg.model.action_seg.transformer_decoder.num_queries
        # Transformer parameters:
        ret["nheads"] = cfg.model.action_seg.transformer_decoder.nheads
        ret["dropout"] = cfg.model.action_seg.transformer_decoder.dropout
        ret["enforce_input_project"] = True # cfg.model.action_seg.transformer_decoder.enforce_input_project 
        ret["mask_dim"] = cfg.model.action_seg.transformer_decoder.mask_dim
        ret["deep_supervision"] = cfg.model.action_seg.transformer_decoder.deep_supervision
        ret["threshold"] = cfg.model.action_seg.transformer_decoder.threshold
        ret["layer_in_decode_block"] = cfg.model.action_seg.transformer_decoder.layer_in_decode_block
        return ret

    def forward(self, x, mask_features, mask=None):
        # Extract shallow features for MS-BPE before decoder layer slicing
        if len(x) > 1:
            shallow_feat = x[1]
        else:
            shallow_feat = x[0]

        if len(x) != self.num_decode:
            x = x[-self.num_decode:]
        assert self.num_decode == len(x)
        src = []
        for i in range(self.num_decode):
            src.append(self.input_proj[i](x[i]).transpose(-1,-2))

        output = self.pe(self.query) + self.query #blc

        predictions_class = []
        predictions_mask = []
        predictions_boundary = []

        outputs_class, outputs_mask, attn_mask, outputs_boundary = self.forward_prediction_heads(output, mask_features, shallow_feat)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        predictions_boundary.append(outputs_boundary)

        for i in range(self.num_decode): #sepearate the decode to decode_level splits
            output = self.trans_decode[i](output, src[i], attn_mask)
            outputs_class, outputs_mask, attn_mask, outputs_boundary = self.forward_prediction_heads(output, mask_features, shallow_feat)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_boundary.append(outputs_boundary)


        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_boundarys': predictions_boundary[-1],
            'pred_query_features': self.decoder_norm(output),  # Pillar 2 (proposal_imprv_baformer14): For DQCR
            'mask_features': mask_features,
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask, predictions_boundary
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, shallow_feat=None):
        decoder_output = self.decoder_norm(output) #b,l,c
        outputs_class = self.class_embed(decoder_output) #just use one linear for different layer
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bcl->bql", mask_embed, mask_features)

        # Pillar 1 (proposal_imprv_baformer14): Tri-Channel Uncompressed Query Transition Tensor
        prob_m = outputs_mask.sigmoid()
        B, Q, L = prob_m.shape
        diff_m = torch.zeros_like(prob_m)
        if L > 1:
            diff_m[:, :, 1:] = torch.abs(prob_m[:, :, 1:] - prob_m[:, :, :-1])

        # Channel 1: Max single-query transition jump (captures sharpest query switch)
        q_max = diff_m.max(dim=1, keepdim=True)[0]  # [B, 1, L]
        # Channel 2: Top-3 active query handover mean
        k_top = min(3, Q)
        q_top3 = diff_m.topk(k=k_top, dim=1)[0].mean(dim=1, keepdim=True)  # [B, 1, L]
        # Channel 3: Active query total variation (queries with delta > 0.05)
        active_mask = (diff_m > 0.05).float()
        q_act_tv = (diff_m * active_mask).sum(dim=1, keepdim=True) / (active_mask.sum(dim=1, keepdim=True) + 1e-4)  # [B, 1, L]

        q_trans = torch.cat([q_max, q_top3, q_act_tv], dim=1)  # [B, 3, L]

        ## boundary: TemporalASPPBoundaryEngine over deep, shallow, and tri-channel transition features
        if shallow_feat is None:
            shallow_feat = mask_features
        outputs_boundary = self.boundary_head(mask_features, shallow_feat, q_trans)  # [B, 1, L]

        # Pillar 3 (proposal_imprv_baformer14): Continuous Boundary Barrier in Attention
        bd_barrier = outputs_boundary.sigmoid()  # [B, 1, L]
        modulated_mask = outputs_mask.sigmoid() * (1.0 - 0.45 * bd_barrier)
        attn_mask = modulated_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        return outputs_class, outputs_mask, attn_mask, outputs_boundary

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, outputs_boundary):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boundarys": c, }
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_boundary[:-1], )
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

class TemporalASPPBoundaryEngine(nn.Module):
    """Temporal Atrous Spatial Pyramid Pooling Boundary Engine (T-ASPP) for Pillar 1 & 2 (proposal_imprv_baformer13).

    Fuses:
    1. Deep context features (from ASFormer deep layer 10)
    2. Shallow high-resolution features (from ASFormer shallow layer 1-2)
    3. Multi-stride local temporal differences (diff1 stride-1, diff2 stride-2)
    4. Dual-stream instance query mask transition gradient (q_diff)
    
    Processes through a 4-branch Temporal ASPP with multi-scale receptive fields:
    - Micro branch (k=3, d=1, pad=1): RF ~ 5 frames (precise frame cut)
    - Meso branch  (k=5, d=3, pad=6): RF ~ 17 frames (gesture completion)
    - Macro branch (k=5, d=6, pad=12): RF ~ 33 frames (macro state change)
    - Global branch: AdaptiveAvgPool1d(1) -> 1x1 conv -> full clip global context
    """

    def __init__(self, in_dim=64, hidden_dim=64, dropout=0.25):
        super().__init__()
        self.proj_deep = nn.Conv1d(in_dim, hidden_dim // 2, kernel_size=1)
        self.proj_shallow = nn.Conv1d(in_dim, hidden_dim // 2, kernel_size=1)
        self.proj_diff1 = nn.Conv1d(in_dim, hidden_dim // 4, kernel_size=3, padding=1)
        self.proj_diff2 = nn.Conv1d(in_dim, hidden_dim // 4, kernel_size=5, padding=2)
        # Pillar 1 (proposal_imprv_baformer14): 3-channel input for Max, Top-3, and Active-TV query deltas
        self.proj_qdiff = nn.Conv1d(3, hidden_dim // 4, kernel_size=3, padding=1)

        fused_dim = (hidden_dim // 2) * 2 + (hidden_dim // 4) * 3
        branch_dim = max(16, hidden_dim // 4)

        # Branch 1: Micro-scale (dilation=1, kernel=3)
        self.branch_micro = nn.Sequential(
            nn.Conv1d(fused_dim, branch_dim, kernel_size=3, dilation=1, padding=1),
            nn.GroupNorm(2, branch_dim),
            nn.SiLU(inplace=True),
        )
        # Branch 2: Meso-scale (dilation=3, kernel=5)
        self.branch_meso = nn.Sequential(
            nn.Conv1d(fused_dim, branch_dim, kernel_size=5, dilation=3, padding=6),
            nn.GroupNorm(2, branch_dim),
            nn.SiLU(inplace=True),
        )
        # Branch 3: Macro-scale (dilation=6, kernel=5)
        self.branch_macro = nn.Sequential(
            nn.Conv1d(fused_dim, branch_dim, kernel_size=5, dilation=6, padding=12),
            nn.GroupNorm(2, branch_dim),
            nn.SiLU(inplace=True),
        )
        # Branch 4: Global Context
        self.branch_global = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(fused_dim, branch_dim, kernel_size=1),
            nn.SiLU(inplace=True),
        )

        aspp_out_dim = branch_dim * 4
        self.head = nn.Sequential(
            nn.Conv1d(aspp_out_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim // 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, deep_feat, shallow_feat, q_diff=None):
        B, C, L = deep_feat.shape
        if shallow_feat.shape[-1] != L:
            shallow_feat = F.interpolate(shallow_feat, size=L, mode='linear', align_corners=False)

        diff1 = torch.zeros_like(deep_feat)
        if L > 2:
            diff1[:, :, 1:-1] = torch.abs(deep_feat[:, :, 2:] - deep_feat[:, :, :-2])

        diff2 = torch.zeros_like(shallow_feat)
        if L > 4:
            diff2[:, :, 2:-2] = torch.abs(shallow_feat[:, :, 4:] - shallow_feat[:, :, :-4])

        p_deep = self.proj_deep(deep_feat)
        p_shallow = self.proj_shallow(shallow_feat)
        p_d1 = self.proj_diff1(diff1)
        p_d2 = self.proj_diff2(diff2)

        if q_diff is not None:
            if q_diff.shape[1] == 1:
                q_diff = q_diff.repeat(1, 3, 1)
            if q_diff.shape[-1] != L:
                q_diff = F.interpolate(q_diff, size=L, mode='linear', align_corners=False)
            p_qd = self.proj_qdiff(q_diff)
        else:
            p_qd = torch.zeros(B, self.proj_qdiff.out_channels, L, device=deep_feat.device, dtype=deep_feat.dtype)

        fused = torch.cat([p_deep, p_shallow, p_d1, p_d2, p_qd], dim=1)

        b1 = self.branch_micro(fused)
        b2 = self.branch_meso(fused)
        b3 = self.branch_macro(fused)
        b4 = self.branch_global(fused)
        b4 = F.interpolate(b4, size=L, mode='nearest')

        aspp_fused = torch.cat([b1, b2, b3, b4], dim=1)
        return self.head(aspp_fused)  # [B, 1, L]

# Backward compatibility alias
MultiScaleBoundaryEngine = TemporalASPPBoundaryEngine


class Sample_pe(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, n_position, d_hid))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        return self.pos_embed[:, :x.size(1)].clone().detach().to(x.device)

class Transformer_layer(nn.Module):
    def __init__(self, in_dim, head_dim, heads, hidden_dim):
        super().__init__()
        self.attn = MultiHeadAttention(dim=in_dim, head_dim=head_dim, heads=heads) # same as pre_norm+ attention
        self.ffn = Feedforward(dim=in_dim, hidden_dim=hidden_dim)# same as pre_norm+ ffn
    def forward(self, x_q, x_kv):
        out_attn = self.attn(x_q, x_kv)
        out = self.ffn(out_attn)
        return out

class Transformer_decoder_layer(nn.Module):
    def __init__(self, in_dim, head_dim, heads, hidden_dim, dropout = 0.1):
        super().__init__()
        self.cross_attn = MultiHeadAttention(dim=in_dim, head_dim=head_dim, heads=heads, dropout = dropout, is_cross=True)
        self.self_attn = MultiHeadAttention(dim=in_dim, head_dim=head_dim, heads=heads, dropout = dropout, is_cross=False) # same as pre_norm+ attention
        self.ffn = Feedforward(dim=in_dim, hidden_dim=hidden_dim)# same as pre_norm+ ffn

    def forward(self, x_q, x_kv, mask):
        out_attn = self.cross_attn(x_q, x_kv, mask) 
        out_attn = self.self_attn(out_attn, out_attn) 
        out = self.ffn(out_attn)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, head_dim, heads, dropout=0.1, is_cross=False):
        super().__init__()
        inner_dim = head_dim * heads
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.is_cross = is_cross

        self.pre_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, inner_dim)
        self.key_value = nn.Linear(dim, inner_dim * 2)

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(inner_dim, dim)

    def forward(self, x_q, x_kv, mask=None): # b l c
        residual = x_q
        # residual
        x_q = self.pre_norm(x_q)

        # MHA
        query = self.query(x_q)
        key_value = self.key_value(x_kv)
        kvq = list(torch.chunk(key_value, 2, dim=-1))
        kvq.append(query)
        k, v, q = map(lambda t: rearrange(t, 'b l (h c) -> b h l c', h=self.heads), kvq)#separate to: b h n d

        # attention
        energy = torch.einsum('bhqc, bhkc -> bhqk', q, k) * self.scale
        if self.is_cross and q.shape[2] > 1 and k.shape[2] > 1:
            # Pillar 3 (proposal_imprv_baformer12): Soft Monotonic Temporal Prior in Cross-Attention
            pos_q = torch.linspace(0, 1, q.shape[2], device=q.device).view(1, 1, -1, 1)
            pos_l = torch.linspace(0, 1, k.shape[2], device=k.device).view(1, 1, 1, -1)
            temporal_bias = -((pos_q - pos_l) ** 2) / (2 * (0.50 ** 2))
            energy = energy + temporal_bias

        if mask is not None:
            # mask has shape (B, heads, Q, L) with values in [0, 1]
            # Pillar 3 (proposal_imprv_baformer14): Continuous Boundary-Constrained Attention
            # Use smooth negative log penalty: log(clamp(mask, 1e-4, 1.0))
            # which smoothly approaches -inf as mask -> 0, avoiding hard discontinuity and preserving stable gradients.
            safe_mask = torch.clamp(mask, min=1e-4, max=1.0)
            attn_bias = torch.log(safe_mask)
            energy = energy + attn_bias
        attn = F.softmax(energy, dim=-1)
        attn = self.att_drop(attn)  # b n head q_win k_win
        out_att = torch.einsum('bhqk, bhkc -> bhqc', attn, v)
        out_att = rearrange(out_att, 'b h q c -> b q (h c)')
        out = self.projection(out_att)

        # residual
        out += residual # prenorm
        return out


class Feedforward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim))

    def forward(self, x):
        residual = x
        #pre_norm
        x = self.pre_norm(x)
        #ffn
        out = self.mlp(x)
        # residual
        out += residual # prenorm
        return out

class Pre_Post_Rearrange(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.rearrange1 = Rearrange('b l c -> b c l')# good for nn.averagepool
        self.rearrange2 = Rearrange('b c l -> b l c')

    def forward(self, x): # b,l,c -> b,c,l -> b, l', c
        out = self.rearrange1(x)
        out = self.fn(out)
        out = self.rearrange2(out)
        return out

class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Mlp2(nn.Module):
    """from groupvit"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerMlp(Mlp2):

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)