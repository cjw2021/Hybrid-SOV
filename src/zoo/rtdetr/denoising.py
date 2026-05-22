"""by lyuwenyu
"""

import torch 

from .utils import inverse_sigmoid
from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh


from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from util import box_ops
import torch.nn.functional as F


def get_contrastive_denoising_training_group(targets,
                                             num_classes,
                                             num_verb_classes,
                                             num_queries,
                                             label_enc, verb_enc,
                                             num_denoising=100,
                                             label_noise_ratio=0.5,
                                             box_noise_scale=1.0,
                                             dn_args=None):
    """cnd"""
    if num_denoising <= 0:
        return None, None, None, None, None
    
    num_gts = [len(t['obj_labels']) for t in targets]
    device = targets[0]['obj_labels'].device

    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None, None

    # num_group = num_denoising // max_gt_num
    # num_group = 1 if num_group == 0 else num_group

    if (num_denoising > 3) and max_gt_num > 24:  # To avoid OOM
        num_denoising = num_denoising - 3
    # obj_dn group, mix_dn group, verb_dn group
    num_group = num_denoising // 3

    # pad gt to max_num of a batch
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_verb_class = torch.zeros([bs, max_gt_num, num_verb_classes], device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    input_query_bbox_sub = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]['obj_labels']
            input_query_bbox[i, :num_gt] = targets[i]['obj_boxes']
            input_query_bbox_sub[i, :num_gt] = targets[i]['sub_boxes']
            pad_gt_mask[i, :num_gt] = 1
    # each group has positive and negative queries.
    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_verb_class = input_query_verb_class.tile([1, 2 * num_group, 1])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    input_query_bbox_sub = input_query_bbox_sub.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])
    # positive and negative mask
    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask
    # contrastive denoising training positive index
    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])
    # total denoising queries
    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        # randomly put a new one here
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        known_bbox_sub = box_cxcywh_to_xyxy(input_query_bbox_sub)

        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        diff_sub = torch.tile(input_query_bbox_sub[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale

        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_sign_sub = torch.randint_like(input_query_bbox_sub, 0, 2) * 2.0 - 1.0

        rand_part = torch.rand_like(input_query_bbox)
        rand_part_sub = torch.rand_like(input_query_bbox_sub)

        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        rand_part_sub = (rand_part_sub + 1.0) * negative_gt_mask + rand_part_sub * (1 - negative_gt_mask)

        rand_part *= rand_sign
        rand_part_sub *= rand_sign_sub

        known_bbox += rand_part * diff
        known_bbox_sub += rand_part_sub * diff_sub

        known_bbox.clip_(min=0.0, max=1.0)
        known_bbox_sub.clip_(min=0.0, max=1.0)

        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox_sub = box_xyxy_to_cxcywh(known_bbox_sub)

        input_query_bbox = inverse_sigmoid(input_query_bbox)
        input_query_bbox_sub = inverse_sigmoid(input_query_bbox_sub)
    
    input_query_class = label_enc(input_query_class)  # [bs, max_gt_num*2*num_group, 256]

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    # match query cannot see the reconstruction
    attn_mask[num_denoising:, :num_denoising] = True

    # reconstruct cannot see each other
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1): num_denoising] = True
            attn_mask[max_gt_num * 2 * i: max_gt_num * 2 * (i + 1), :max_gt_num * 2 * i] = True
    
    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries]
    }

    return input_query_class, input_query_bbox, input_query_bbox_sub, attn_mask, dn_meta
