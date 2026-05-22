"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import torchvision

from src.core import register

from .box_ops import box_cxcywh_to_xyxy


__all__ = ['RTDETRPostProcessorHOI']


@register
class RTDETRPostProcessorHOI(nn.Module):
    __share__ = ['num_classes', 'use_focal_loss', 'num_top_queries', 'remap_mscoco_category']
    
    def __init__(self, num_classes=80, use_focal_loss=True, num_top_queries=300, remap_mscoco_category=False, subject_category_id=0) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = num_classes
        self.remap_mscoco_category = remap_mscoco_category 
        self.deploy_mode = False 

        self.subject_category_id = subject_category_id

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'
    
    # def forward(self, outputs, orig_target_sizes):
    def forward(self, outputs, target_sizes):

        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_logits'], \
                                                                        outputs['pred_verb_logits'], \
                                                                        outputs['pred_sub_boxes'], \
                                                                        outputs['pred_obj_boxes']



        ###########################################################################
        # HOI 600 Prediction
        ###########################################################################
        out_hoi_logits = outputs['pred_hoi_logits']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        
        hoi_scores = out_hoi_logits.sigmoid()
        obj_scores = out_obj_logits.softmax(-1)  # bs, nq, cls
        obj_labels = F.softmax(out_obj_logits, -1).max(-1)[1]  #idx: bs, nq
        ###########################################################################
        
        verb_scores = out_verb_logits.sigmoid()  # bs, nq, cls

        img_h, img_w = target_sizes.unbind(1)  # bs,
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)  # bs, 4
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)  # bs, nq, 4
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)  # bs, nq, 4
        
        sub_boxes = sub_boxes * scale_fct[:, None, :]  # bs, nq, 4
        obj_boxes = obj_boxes * scale_fct[:, None, :]  # bs, nq, 4

        results = []
        # HOI 600 Prediction
        for index in range(len(hoi_scores)):
            hs, os, ol, vs, sb, ob = hoi_scores[index], obj_scores[index], obj_labels[index], verb_scores[index], sub_boxes[index], obj_boxes[index]

            sl = torch.full_like(ol, self.subject_category_id) # nq,
            l = torch.cat((sl, ol))  # 2*nq,
            b = torch.cat((sb, ob))  # 2*nq, 4
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            ids = torch.arange(b.shape[0]) # 2*nq, [0, 1, 2, ..., 2 * nq - 1]

            results[-1].update({'hoi_scores': hs.to('cpu'), 
                'verb_scores': vs.to('cpu'), 'obj_scores': os.to('cpu'),
                'sub_ids': ids[:ids.shape[0] // 2], 'obj_ids': ids[ids.shape[0] // 2:]})

        return results
        

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self 

    @property
    def iou_types(self, ):
        return ('bbox', )

