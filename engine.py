# ------------------------------------------------------------------------
# SOV-STG
# Copyright (c) 2023 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from QAHOI (https://github.com/cjw2021/QAHOI)
# Copyright (c) 2021 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from QPIC (https://github.com/hitachi-rd-cv/qpic)
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable
import torch
import itertools
import util.misc as utils
from datasets.hico_eval import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator
from datasets_hoi.hico_eval_triplet import HICOEvaluator as HICOEvaluator_hoi
from datasets_hoi.vcoco_eval import VCOCOEvaluator as VCOCOEvaluator_hoi
import copy
from loguru import logger
import numpy as np
import math


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, ema_m=None,
                    run=None, wandb_loss_log_all=True):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    assert args.dn_drop_lower_limit >= 0
    
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('dn_dy_weight', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('dn_drop', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename' and k != 'raw_img'} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):

            assert args.dn_dynamic_lower_limit >= 0 and args.dn_dynamic_lower_limit <= 1
            if args.dn_dynamic_coef_type == 'static':
                dn_dy_coef = args.dn_dynamic_lower_limit
                # args.dn_dynamic_lower_limit = 1.0
            elif args.dn_dynamic_coef_type == 'cosine':
                dn_dy_coef = args.dn_dynamic_lower_limit + (1 - args.dn_dynamic_lower_limit) * (1 - np.cos(np.pi * epoch/(2 * args.epochs)))
            elif args.dn_dynamic_coef_type == 'linear':
                dn_dy_coef = args.dn_dynamic_lower_limit + (1 - args.dn_dynamic_lower_limit) * (epoch / args.epochs)
            else:
                dn_dy_coef = 1.0

            dn_args=(targets, dn_dy_coef, args, args.scalar, args.label_noise_scale, args.verb_noise_scale, args.box_noise_scale, args.num_patterns)
            
            outputs = model(samples.tensors, dn_args)
            loss_dict = criterion(outputs, targets)

            dn_drop_weight = 1 - (1 - args.dn_drop_lower_limit) * math.log(epoch + 1, args.epochs)
            for k in loss_dict.keys():
                if "dn" in k:
                    loss_dict[k] = loss_dict[k] * dn_drop_weight
            losses = sum(loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced_scaled = sum(loss_dict_reduced.values())
        plain_loss_scaled = torch.tensor(0.0) + sum(loss_dict_reduced[k] for k in loss_dict_reduced.keys() if "dn" not in k)
        dn_loss_scaled = torch.tensor(0.0) + sum(loss_dict_reduced[k] for k in loss_dict_reduced.keys() if "dn" in k)
        loss_value = losses_reduced_scaled.item()
        plain_loss_value = plain_loss_scaled.item()
        dn_loss_value = dn_loss_scaled.item()
        if run:
            if wandb_loss_log_all:
                run.log(loss_dict_reduced)
            run.log({'loss': loss_value, 'plain_loss': plain_loss_value, 'dn_loss': dn_loss_value})

        bug_count = 0
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            optimizer.zero_grad()
            bug_count = bug_count + 1
            if bug_count > 5:
                sys.exit(1)
            continue
        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        metric_logger.update(loss=loss_value, plain_loss=plain_loss_value, dn_loss=dn_loss_value)
        metric_logger.update(**loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(dn_dy_weight=dn_dy_coef)
        metric_logger.update(dn_drop=dn_drop_weight)
        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if utils.get_rank() == 0:
        logger.info("\nAveraged stats: {}".format(metric_logger))

    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})

    return resstat


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader, subject_category_id, device, out_dir, epoch, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets_t = [{k: v.to(device) for k, v in t.items() if not k in ['file_name', "id", 'img_id']} for t in targets]

        outputs = model(samples.tensors, (targets_t,0,))
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(samples.device)
        results = postprocessors(outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    rank = utils.get_rank()
    if dataset_file == 'hico':
        if "HOI" in args.arch:
            evaluator = HICOEvaluator_hoi(preds, gts, args.hoi_path, out_dir, epoch, use_nms=args.use_nms, nms_thresh=args.nms_thresh, nms_alpha=args.nms_alpha, nms_beta=args.nms_beta)
        else:
            evaluator = HICOEvaluator(preds, gts, args.hoi_path, out_dir, epoch, use_nms=args.use_nms, nms_thresh=args.nms_thresh, nms_alpha=args.nms_alpha, nms_beta=args.nms_beta)

        stats = evaluator.evaluation_default()
        if rank == 0:
            logger.info('\n--------------------\ndefault mAP: {}\ndefault mAP rare: {}\ndefault mAP non-rare: {}\n--------------------'.format(stats['mAP_def'], stats['mAP_def_rare'], stats['mAP_def_non_rare']))
        stats_ko = evaluator.evaluation_ko()
        if rank == 0:
            logger.info('\n--------------------\nko mAP: {}\nko mAP rare: {}\nko mAP non-rare: {}\n--------------------'.format(stats_ko['mAP_ko'], stats_ko['mAP_ko_rare'], stats_ko['mAP_ko_non_rare']))
        stats.update(stats_ko)
        if args.eval_extra:
            evaluator.evaluation_extra()
    elif dataset_file == 'vcoco':
        if "HOI" in args.arch:
            evaluator = VCOCOEvaluator_hoi(preds, gts, subject_category_id, data_loader.dataset.correct_mat, use_nms=args.use_nms, nms_thresh=args.nms_thresh, nms_alpha=args.nms_alpha, nms_beta=args.nms_beta)
        else:
            evaluator = VCOCOEvaluator(preds, gts, subject_category_id, data_loader.dataset.correct_mat, use_nms=args.use_nms, nms_thresh=args.nms_thresh, nms_alpha=args.nms_alpha, nms_beta=args.nms_beta)
    elif dataset_file == 'vcoco_self_object':
        if "HOI" in args.arch:
            evaluator = VCOCOEvaluator_hoi(preds, gts, subject_category_id, data_loader.dataset.correct_mat, use_nms=args.use_nms, nms_thresh=args.nms_thresh, nms_alpha=args.nms_alpha, nms_beta=args.nms_beta)
        else:
            evaluator = VCOCOEvaluator(preds, gts, subject_category_id, data_loader.dataset.correct_mat, use_nms=args.use_nms, nms_thresh=args.nms_thresh, nms_alpha=args.nms_alpha, nms_beta=args.nms_beta)


        stats = evaluator.evaluate()
        if rank == 0:
            logger.info('\n--------------------\nmAP all: {:.4f} mAP thesis: {:.4f}\n--------------------'.format(stats['mAP_all'], stats['mAP_thesis']))

    return stats
