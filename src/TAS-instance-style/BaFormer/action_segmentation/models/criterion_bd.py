import torch
import torch.nn.functional as F
from torch import nn

import torch.distributed as dist
from detectron2.utils.comm import get_world_size

# from detectron2.config import configurable
from ..config import configurable

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from typing import List, Optional

import numpy as np
import pandas as pd
import os

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
dataset_names = ["50salads", "breakfast", "gtea"]
modes = ["training", "trainval"]
def get_pos_weight(
    dataset: str,
    split: int = 1,
    csv_dir: str = "./csv",
    data_dir: str = None,
    mode: str = "trainval",
    norm: Optional[float] = None,
) -> torch.Tensor:
    """
    pos_weight for binary cross entropy with logits loss
    pos_weight is defined as reciprocal of ratio of positive samples in the dataset
    """

    assert (
        mode in modes
    ), "You have to choose 'training' or 'trainval' as the dataset mode"

    assert (
        dataset in dataset_names
    ), "You have to select 50salads, gtea or breakfast as dataset."

    if mode == "training":
        df = pd.read_csv(os.path.join(csv_dir, dataset, "train{}.csv").format(split))
    elif mode == "trainval":
        df1 = pd.read_csv(os.path.join(csv_dir, dataset, "train{}.csv".format(split)))
        df2 = pd.read_csv(os.path.join(csv_dir, dataset, "val{}.csv".format(split)))
        df = pd.concat([df1, df2])

    n_classes = 2  # boundary or not
    nums = [0 for i in range(n_classes)]
    for i in range(len(df)):
        label_path = data_dir + df.iloc[i]["boundary"][10:]
        label = np.load(label_path).astype(np.int64)
        num, cnt = np.unique(label, return_counts=True)
        for n, c in zip(num, cnt):
            nums[n] += c

    pos_ratio = nums[1] / sum(nums)
    pos_weight = 1 / pos_ratio

    if norm is not None:
        pos_weight /= norm

    return torch.tensor(pos_weight)

def get_pos_weight_single(
    dataset: str,
    csv_dir: str = "./csv",
    data_dir: str = None,
    name:str = None,
    mode: str = "trainval",
    norm: Optional[float] = None,
) -> torch.Tensor:
    """
    pos_weight for binary cross entropy with logits loss
    pos_weight is defined as reciprocal of ratio of positive samples in the dataset
    """

    assert (
        mode in modes
    ), "You have to choose 'training' or 'trainval' as the dataset mode"

    assert (
        dataset in dataset_names
    ), "You have to select 50salads, gtea or breakfast as dataset."

    #/dataset/50salads/gt_boundary_arr/rgb-19-1.npy

    n_classes = 2  # boundary or not
    nums = [0 for i in range(n_classes)]
    label_path = os.path.join(data_dir, dataset, 'gt_boundary_arr', name[0]+'.npy')
    label = np.load(label_path).astype(np.int64)
    num, cnt = np.unique(label, return_counts=True)
    for n, c in zip(num, cnt):
        nums[n] += c

    pos_ratio = nums[1] / sum(nums)
    pos_weight = 1 / pos_ratio

    if norm is not None:
        pos_weight /= norm

    return torch.tensor(pos_weight)

def dice_loss(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)  ## same to IOU
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule

def sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid() #input[num_mask, L] target[num_mask, L], so the num_mask is same to batchsize, and BCE on each mask
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets) ## p_t within 0~1
    loss = ce_loss * ((1 - p_t) ** gamma) ## 1-p_t within 0~1, so gamma >1?

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


def boundary_dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Soft Dice Loss for continuous / Gaussian boundary heatmaps (proposal_imprv_baformer09).

    Penalizes global False Positives by expanding the denominator whenever
    background frames have non-zero predicted boundary probabilities.
    """
    p = torch.sigmoid(inputs)
    y = targets
    if p.shape != y.shape:
        min_l = min(p.shape[-1], y.shape[-1])
        p = p[..., :min_l]
        y = y[..., :min_l]

    p_flat = p.flatten(1) if p.dim() > 1 else p.unsqueeze(0)
    y_flat = y.flatten(1) if y.dim() > 1 else y.unsqueeze(0)

    intersection = 2.0 * (p_flat * y_flat).sum(dim=-1) + eps
    union = p_flat.sum(dim=-1) + y_flat.sum(dim=-1) + eps
    loss = 1.0 - (intersection / union)
    return loss.mean()


def binary_focal_loss_with_logits(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Pillar 4 (proposal_imprv_baformer14): Harmonically Self-Modulating Focal Loss (S-CSBL).

    Dynamically adjusts alpha_eff(t) based on continuous Gaussian target density y(t) in [0, 1]:
      alpha_eff(t) = 0.50 + 0.25 * targets
    - At boundary center (y = 1.0): alpha_eff = 0.75, providing 3:1 gradient priority to push peak p > 0.50.
    - In deep interior (y = 0.0): alpha_eff = 0.50, negative penalty weight is (1 - 0.50) = 0.50 (double
      standard 0.25), crushing interior noise floor below 0.05 naturally without artificial beta_interior.
    - Smoothly transitions across Gaussian skirt, eliminating step-function artifacts and TP suppression.
    """
    if inputs.shape != targets.shape:
        min_l = min(inputs.shape[-1], targets.shape[-1])
        inputs = inputs[..., :min_l]
        targets = targets[..., :min_l]

    p = torch.sigmoid(inputs)
    log_p = -F.softplus(-inputs)
    log_1_minus_p = -F.softplus(inputs)

    # Harmonically self-modulating alpha
    alpha_eff = 0.50 + 0.25 * targets
    pos_weight = alpha_eff * ((1.0 - p) ** gamma)
    neg_weight = (1.0 - alpha_eff) * (p ** gamma)

    loss = -(targets * pos_weight * log_p + (1.0 - targets) * neg_weight * log_1_minus_p)
    return loss.mean()


def decoder_query_repulsion_loss(query_features, indices, targets, margin=0.20):
    """Pillar 2 (proposal_imprv_baformer14): Decoder Query Contrastive Repulsion (DQCR).

    Prevents query collapse and mask bleeding on same-class adjacent instances (53.6% of dataset).
    For consecutively matched queries (ordered by temporal start frame), enforces a cosine margin
    repulsion loss in latent representation space:
        L_repulse = sum_i max(0, cos(z_i, z_{i+1}) - margin)
    """
    if query_features is None or not indices:
        return torch.tensor(0.0, device=query_features.device if query_features is not None else 'cpu')

    device = query_features.device
    loss_repulse = torch.tensor(0.0, device=device)
    total_pairs = 0

    B = query_features.shape[0]
    for b in range(B):
        if b >= len(indices):
            continue
        src_idx, tgt_idx = indices[b]
        if len(src_idx) < 2:
            continue

        src_idx = src_idx.to(device)
        tgt_idx = tgt_idx.to(device)

        target_b = targets[b]
        if 'instances' in target_b and target_b['instances'].shape[0] > 0:
            insts = target_b['instances'].to(device)
            if insts.dim() == 3:
                insts = insts.squeeze(0)
            start_times = insts[tgt_idx, 0]
        else:
            start_times = tgt_idx.float()

        sort_order = torch.argsort(start_times)
        ordered_src = src_idx[sort_order]

        matched_feats = query_features[b, ordered_src]  # [K, D]
        norm_feats = F.normalize(matched_feats, p=2, dim=-1)

        cos_sim = (norm_feats[:-1] * norm_feats[1:]).sum(dim=-1)  # [K-1]
        pair_loss = F.relu(cos_sim - margin)
        loss_repulse = loss_repulse + pair_loss.sum()
        total_pairs += len(cos_sim)

    if total_pairs > 0:
        return loss_repulse / total_pairs
    return loss_repulse


def same_class_contrastive_loss(mask_features, targets, tau=0.1):
    """Pillar 2 (proposal_imprv_baformer11): Same-Class Instance Contrastive Loss.

    For each boundary transition (especially same-class transitions), pulls positive
    frames within the same instance together and pushes negative frames across the
    boundary apart in representation space.
    """
    if mask_features is None or len(targets) == 0:
        return torch.tensor(0.0, device=mask_features.device if mask_features is not None else 'cpu')

    loss_list = []
    for b_idx, target in enumerate(targets):
        if 'instances' not in target:
            continue
        instances = target['instances']
        if instances.dim() == 3:
            instances = instances.squeeze(0)
        num_ins = instances.shape[0]
        if num_ins < 2:
            continue

        feat = mask_features[b_idx]  # [C, L]
        L = feat.shape[-1]
        feat_norm = F.normalize(feat, p=2, dim=0)  # [C, L] normalized per frame

        for i in range(num_ins - 1):
            s_curr = int(instances[i, 0])
            e_curr = int(instances[i, 1])
            c_curr = int(instances[i, 2])

            s_next = int(instances[i + 1, 0])
            e_next = int(instances[i + 1, 1])
            c_next = int(instances[i + 1, 2])

            b = min(s_next, L - 1)
            if b <= 2 or b >= L - 2:
                continue

            # Anchor: frame b - 1 (end of current instance)
            z_anchor = feat_norm[:, b - 1]  # [C]

            # Positive: frame within current instance
            pos_idx = max(s_curr, b - 4)
            z_pos = feat_norm[:, pos_idx]  # [C]

            # Negatives: frames across boundary in next instance
            neg_indices = [idx for idx in [b, min(b + 2, e_next - 1), min(b + 4, e_next - 1)] if 0 <= idx < L]
            if len(neg_indices) == 0:
                continue
            z_negs = feat_norm[:, neg_indices]  # [C, K]

            # Cosine similarities scaled by temperature
            pos_sim = torch.dot(z_anchor, z_pos) / tau
            neg_sims = torch.matmul(z_anchor.unsqueeze(0), z_negs).squeeze(0) / tau  # [K]

            # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sims], dim=0)
            labels = torch.zeros(1, dtype=torch.long, device=feat.device)
            loss_step = F.cross_entropy(logits.unsqueeze(0), labels)

            # If same_class transition, double the loss weight
            if c_curr == c_next:
                loss_step = loss_step * 1.5

            loss_list.append(loss_step)

    if len(loss_list) == 0:
        return torch.tensor(0.0, device=mask_features.device)
    return torch.stack(loss_list).mean()


def tmse_loss(logits: torch.Tensor, threshold: float = 4.0) -> torch.Tensor:
    """Truncated Mean Squared Error (T-MSE) Smoothing Loss (MS-TCN, ASFormer, FACT CVPR 2024).

    Penalizes abrupt frame-to-frame log-probability fluctuations across the temporal dimension,
    mitigating over-segmentation and label flickering.
    """
    log_p = F.log_softmax(logits, dim=1)
    diff = log_p[:, :, 1:] - log_p[:, :, :-1]
    truncated_diff = torch.clamp(diff, min=-threshold, max=threshold)
    return torch.mean(truncated_diff ** 2)


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

class SetCriterion_bd(nn.Module):
    @configurable
    def __init__(self, pos_weight_info, num_classes, matcher, weight_dict, eos_coef, losses, pos_weight, label_smooth, class_weights=None):
        super().__init__()
        self.num_classes = num_classes # for fill and compute empty_weight
        self.matcher = matcher
        self.weight_dict = weight_dict
        empty_weight = torch.ones(self.num_classes + 1)
        if class_weights is not None and len(class_weights) == self.num_classes:
            empty_weight[:self.num_classes] = torch.as_tensor(class_weights, dtype=torch.float32)
        empty_weight[-1] = eos_coef
        self.losses = losses

        self.label_smooth = label_smooth
        self.register_buffer("empty_weight", empty_weight)

        if isinstance(pos_weight, list):
            self.use_dynamic = True
            self.pos_weight = get_pos_weight(
                                            dataset=pos_weight_info[0],
                                            split=pos_weight_info[1],
                                            csv_dir=parent_directory +"/csv",
                                            data_dir = pos_weight_info[2],
                                            mode="training")
            self.pos_weight_ratio = pos_weight[0]
        else:
            self.use_dynamic = False
            self.pos_weight = pos_weight


    @classmethod
    def from_config(cls, cfg):
        pos_weight_info = (cfg.dataset.name, cfg.dataset.split, cfg.dataset.dataset_dir)

        ce_weight = cfg.model.ce_weight
        mask_weight = cfg.model.mask_weight
        dice_weight = cfg.model.dice_weight
        num_classes = cfg.dataset.n_classes
        class_weights = getattr(cfg.dataset, 'class_weights', None)

        matcher = HungarianMatcher(
            cost_class= ce_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )
        weight_dict = {"loss_ce": ce_weight, "loss_mask": mask_weight, "loss_dice": dice_weight} # for combine all the loss
        losses = ['labels', 'masks', 'boundarys']
        return {
            'pos_weight_info': pos_weight_info,
            'num_classes': num_classes,
            'matcher': matcher,
            'weight_dict': weight_dict,
            'eos_coef': cfg.model.eos_coef,
            'losses': losses,
            'pos_weight': cfg.dataset.pos_weight,
            'label_smooth': cfg.augmentation.label_smoothing_epsilon,
            'class_weights': class_weights,
        }


    def loss_labels(self, outputs, targets, indices, num_masks):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float() ##[bs, num_queries, cls]  each query has cls prob

        idx = self._get_src_permutation_idx(indices) #indices:tuple[bs](src_id(in order),tgt_id); idx:tuple batch_idx, src_idx (in order)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) #for bs, target_classes_o:change to the src_id order for correspnding
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)# all is empty class, but same shape with pred

        target_classes[idx] = target_classes_o # in src order, and selected from pred

        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight.to(target_classes.device), label_smoothing= self.label_smooth) #don't ignore the empty class learning here
        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, label_smoothing=0.1) #don't ignore the empty class learning here

        #
        # print('src',src_logits.argmax(-1))
        # print('tgt', target_classes)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        assert "pred_masks" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx] # selected src mask, same size as target
        masks = [t["masks"] for t in targets] # all target_mask [target_mask size]
        # TODO use valid to mask invalid areas due to padding in loss
        # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = torch.stack(masks, dim=0)
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        # src_masks = src_masks[:, 0].flatten(1)
        #
        # target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)

        # ##L_1 smooth loss
        # prediction = torch.sigmoid(src_masks)
        # F1_loss = F.smooth_l1_loss(prediction, target_masks)

        # Pillar 4 (proposal_imprv_baformer10): Query Mask TV Temporal Regularization
        src_prob = src_masks.sigmoid()
        loss_mask_tv = torch.mean(torch.abs(src_prob[:, 1:] - src_prob[:, :-1]))

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_masks),
            "loss_dice": dice_loss(src_masks, target_masks, num_masks),
            "loss_mask_tv": loss_mask_tv,
        }
        return losses

    def loss_boundarys(self, outputs, targets, indices=None, num_masks=None):
        assert "pred_boundarys" in outputs
        target_boundarys = torch.cat([target['boundarys'] for target in targets])[None, :].float()
        src_boundayrs = outputs['pred_boundarys'].flatten(0,1)

        # Calibrated Dual Boundary Objective (Pillar 4, proposal_imprv_baformer11):
        # 1. Balanced Focal Boundary Loss (alpha=0.75 restores positive transition gradient)
        loss_bd_focal = binary_focal_loss_with_logits(src_boundayrs, target_boundarys, alpha=0.75, gamma=2.0)
        # 2. Global Boundary Dice Loss (scaled 1.5x)
        loss_bd_dice = boundary_dice_loss(src_boundayrs, target_boundarys)

        loss_bd = loss_bd_focal + 1.5 * loss_bd_dice
        losses = {"loss_bd":  loss_bd}
        return losses

    def _get_src_permutation_idx(self, indices):  ## find target according to the prediction
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])  ## i:bs, (src_id, tgt_id) full the tensor with same size as scr with batch_idx
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):  ## loss ："labels" or "masks"
        loss_map = {"labels": self.loss_labels, "masks": self.loss_masks, "boundarys": self.loss_boundarys}  ## just find the loss functions here
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, pos_weight_info=None):
        self.pos_weight_info = pos_weight_info
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor( num_masks, dtype=torch.float, device=next(iter(outputs.values())).device )

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()  ###? why clamp here

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))  # outputs targets [label, mask]

        # Pillar 2 (proposal_imprv_baformer11): Same-Class Contrastive Loss
        if "mask_features" in outputs:
            losses["loss_contra"] = same_class_contrastive_loss(outputs["mask_features"], targets)

        # Pillar 2 (proposal_imprv_baformer14): Decoder Query Contrastive Repulsion (DQCR)
        if "pred_query_features" in outputs:
            losses["loss_repulse"] = decoder_query_repulsion_loss(outputs["pred_query_features"], indices, targets, margin=0.20)

        if "aux_outputs" in outputs:
            # Pillar 1 (proposal_imprv_baformer06): Decouple auxiliary boundary supervision.
            # Boundary predictions originate from the static frame encoder (mask_features) and do not
            # evolve across decoder layers. Supervise 'boundarys' ONLY on the final output (above),
            # while intermediate layers only supervise 'labels' and 'masks'. This eliminates 75%
            # redundant boundary gradient accumulation and stops boundary head overfitting.
            aux_losses = [l for l in self.losses if l != "boundarys"]
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in aux_losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        # Pillar 3 (proposal_imprv_baformer07): Direct Frame-Wise Supervision for ASFormer Encoder
        if "encoder_logits" in outputs and len(targets) > 0 and "frame_target" in targets[0]:
            enc_logits = outputs["encoder_logits"]  # [B, num_classes, L]
            ft = targets[0]["frame_target"].to(enc_logits.device).long()
            if ft.dim() == 1:
                ft = ft.unsqueeze(0)
            if enc_logits.shape[-1] != ft.shape[-1]:
                min_l = min(enc_logits.shape[-1], ft.shape[-1])
                enc_logits = enc_logits[:, :, :min_l]
                ft = ft[:, :min_l]
            loss_enc_ce = F.cross_entropy(enc_logits, ft, self.empty_weight[:self.num_classes].to(enc_logits.device))
            losses["loss_enc_ce"] = loss_enc_ce

            # Pillar 1 (proposal_imprv_baformer08): T-MSE Temporal Smoothing Loss
            loss_enc_smooth = tmse_loss(enc_logits, threshold=4.0)
            losses["loss_enc_smooth"] = loss_enc_smooth

        return losses


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True




def batch_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid() ##[num_queries, HW]
    inputs = inputs.flatten(1) ##[nun_queries, HW]
    # numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets) ##targets[num_total_objects, HW]
    numerator = 2 * torch.matmul(inputs, targets.transpose(-1,-2)) ##targets[num_total_objects, HW]

    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1) ##more overlap, less loss, so 1-IOU
    return loss


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1] ## input [num_queries, HW]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none" # in 1/N, the prob of each frame belongs to 1, N same as the bachsize
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    # loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
    #     "nc,mc->nm", focal_neg, (1 - targets)   #mix all the results here according to the targets
    # )  ##get nm like similarity metrix for pos and neg

    loss = torch.matmul(focal_pos, targets.transpose(-1,-2)) + torch.matmul( focal_neg, (1 - targets).transpose(-1,-2))
    return loss / hw


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Work out the mask padding size
        masks = [v["masks"] for v in targets] ##v:[num_query, h, w]
        # h_max = max([m.shape[1] for m in masks])
        # w_max = max([m.shape[2] for m in masks])

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]

            tgt_ids = targets[b]["labels"] ##[num_cls_id], all object class in the image, different from the num_classes
            tgt_mask = targets[b]["masks"].to(out_mask)

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids] ##[num_queries, num_total_targets], broadcast, get all the cls_id prob in each query, and ignore the object_cls not existed

            # # Downsample gt masks to save memory
            # tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest") ##[num_queries, H, W]
            # # Flatten spatial dimension
            # out_mask = out_mask.flatten(1)  # [batch_size * num_queries, H*W] ##[num_queries, H*W] batch_size=1, a sample
            # tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]

            # Compute the focal loss between masks
            cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask) ##out_mask:[num_queries, H*W], tgt_mask:[num_total_targets, H*W]
                                                                    ##cost_mask [num_queries, H*W]

            # Compute the dice loss betwen masks
            cost_dice = batch_dice_loss(out_mask, tgt_mask)  ## compute the IOU (overlap) of the mask (focus on positive pixel)
                                                            ## cost_dice [num_queries, H*W]

            # Final cost matrix
            C = (  ##[nm]=[num_queries, num_total_targets]
                self.cost_mask * cost_mask
                + self.cost_class * cost_class  # the detailed processing
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu() #[num_queries, num_total_targets]

            indices.append(linear_sum_assignment(C)) ## for
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) ## change numpy to tensor
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)  ##List20 [0]=[i,j] the indices of prediction and targets

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)



class HardHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Work out the mask padding size
        masks = [v["masks"] for v in targets] ##v:[num_query, h, w]
        # h_max = max([m.shape[1] for m in masks])
        # w_max = max([m.shape[2] for m in masks])

        indices = []

        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]

            tgt_ids = targets[b]["labels"] ##[num_cls_id], all object class in the image, different from the num_classes
            tgt_mask = targets[b]["masks"].to(out_mask)

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids] ##[num_queries, num_total_targets], broadcast, get all the cls_id prob in each query, and ignore the object_cls not existed

            # # Downsample gt masks to save memory
            # tgt_mask = F.interpolate(tgt_mask[:, None], size=out_mask.shape[-2:], mode="nearest") ##[num_queries, H, W]
            # # Flatten spatial dimension
            # out_mask = out_mask.flatten(1)  # [batch_size * num_queries, H*W] ##[num_queries, H*W] batch_size=1, a sample
            # tgt_mask = tgt_mask[:, 0].flatten(1)  # [num_total_targets, H*W]

            # Compute the focal loss between masks
            cost_mask = batch_sigmoid_focal_loss(out_mask, tgt_mask) ##out_mask:[num_queries, H*W], tgt_mask:[num_total_targets, H*W]
                                                                    ##cost_mask [num_queries, H*W]

            # Compute the dice loss betwen masks
            cost_dice = batch_dice_loss(out_mask, tgt_mask)  ## compute the IOU (overlap) of the mask (focus on positive pixel)
                                                            ## cost_dice [num_queries, H*W]

            # Final cost matrix
            C = (  ##[nm]=[num_queries, num_total_targets]
                self.cost_mask * cost_mask
                + self.cost_class * cost_class  # the detailed processing
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu() #[num_queries, num_total_targets]

            indices.append(linear_sum_assignment(C)) ## for
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) ## change numpy to tensor
            for i, j in indices
        ]

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)  ##List20 [0]=[i,j] the indices of prediction and targets

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
