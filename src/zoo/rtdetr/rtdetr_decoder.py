"""by lyuwenyu
"""

import math 
import copy 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob


from src.core import register


__all__ = ['RTDETRTransformer', ]

#### RTDETRTransformer_00001: enc_verb_by_sub_memory
#### 


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4,):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2,)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()


    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2] + sampling_offsets /
                self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4,):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(\
            self.with_pos_embed(tgt, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None,
                with_score=True,
                with_bbox=True):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        if with_bbox:
            ref_points_detach = F.sigmoid(ref_points_unact)
        else:
            ref_points_detach = ref_points_unact

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)

            if with_bbox:
                inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                if with_score:
                    dec_out_logits.append(score_head[i](output))
                if with_bbox:
                    if i == 0:
                        dec_out_bboxes.append(inter_ref_bbox)
                    else:
                        dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                if with_score:
                    dec_out_logits.append(score_head[i](output))
                if with_bbox:
                    dec_out_bboxes.append(inter_ref_bbox)
                break
            
            if with_bbox:
                ref_points = inter_ref_bbox
                ref_points_detach = inter_ref_bbox.detach(
                ) if self.training else inter_ref_bbox

        dec_out_bboxes = torch.stack(dec_out_bboxes) if with_bbox else None
        dec_out_logits = torch.stack(dec_out_logits) if with_score else None
        return output, dec_out_bboxes, dec_out_logits


def adaptive_shifted_MBR(reference_points_input):
    reference_points_input_x = (reference_points_input[:, :, 0] + reference_points_input[:, :, 4])/2
    reference_points_input_y = (reference_points_input[:, :, 1] + reference_points_input[:, :, 5])/2
    reference_points_input_w = torch.abs(reference_points_input[:, :, 0] - reference_points_input[:, :, 4]) \
        + (reference_points_input[:, :, 2] + reference_points_input[:, :, 6])/2
    reference_points_input_h = torch.abs(reference_points_input[:, :, 1] - reference_points_input[:, :, 5]) \
        + (reference_points_input[:, :, 3] + reference_points_input[:, :, 7])/2
    reference_points_input = torch.stack([reference_points_input_x, reference_points_input_y, reference_points_input_w, reference_points_input_h],-1)
    return reference_points_input


@register
class RTDETRTransformer(nn.Module):
    __share__ = ['num_classes']
    def __init__(self,
                 num_classes=80,
                 num_verb_classes=117,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True,
                 topk_cat="object"):

        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_verb_classes = num_verb_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        self.topk_cat = topk_cat

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        decoder_layer_sub = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder_sub = TransformerDecoder(hidden_dim, decoder_layer_sub, num_decoder_layers, eval_idx)

        decoder_layer_verb = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels, num_decoder_points)
        self.decoder_verb = TransformerDecoder(hidden_dim, decoder_layer_verb, num_decoder_layers, eval_idx)
        self.verb_embed = nn.Linear(hidden_dim, num_verb_classes)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0:
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            # self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            self.label_enc = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)  # for indicator
        

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.query_pos_head_sub = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self.query_pos_head_verb = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,)
        )
        self.enc_output_sub = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim,)
        )
        # self.enc_output_verb = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim,)
        # )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_score_head_verb = nn.Linear(hidden_dim, num_verb_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)
        self.enc_bbox_head_sub = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head_sub = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        self.verb_embed.bias.data = torch.ones(self.num_verb_classes) * bias
        self.verb_embed = _get_clones(self.verb_embed, self.num_decoder_layers)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_score_head_verb.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        init.constant_(self.enc_bbox_head_sub.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head_sub.layers[-1].bias, 0)

        for cls_, reg_, reg_sub_ in zip(self.dec_score_head, self.dec_bbox_head, self.dec_bbox_head_sub):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)
            init.constant_(reg_sub_.layers[-1].weight, 0)
            init.constant_(reg_sub_.layers[-1].bias, 0)
        
        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        init.xavier_uniform_(self.enc_output_sub[0].weight)
        # init.xavier_uniform_(self.enc_output_verb[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head_sub.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head_sub.layers[1].weight)
        init.xavier_uniform_(self.query_pos_head_verb.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head_verb.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return (feat_flatten, spatial_shapes, level_start_index)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(\
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None,
                           denoising_bbox_unact_sub=None,
                           topk_cat="object"):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

        output_memory = self.enc_output(memory)
        output_memory_sub = self.enc_output_sub(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_verb_class = self.enc_score_head_verb(output_memory_sub)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors
        enc_outputs_coord_unact_sub = self.enc_bbox_head_sub(output_memory_sub) + anchors

        if topk_cat == "verb":
            _, topk_ind = torch.topk(enc_outputs_verb_class.max(-1).values, self.num_queries, dim=1)
        elif topk_cat == "object":
            _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        else:
            raise ValueError("Not supported topk_cat: {}".format(topk_cat))
        
        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact.shape[-1]))
        reference_points_unact_sub = enc_outputs_coord_unact_sub.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_unact_sub.shape[-1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        enc_topk_bboxes_sub = F.sigmoid(reference_points_unact_sub)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)
            reference_points_unact_sub = torch.concat(
                [denoising_bbox_unact_sub, reference_points_unact_sub], 1)
        
        enc_topk_logits = enc_outputs_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1]))
        enc_topk_verb_class = enc_outputs_verb_class.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_verb_class.shape[-1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
            target_sub = self.tgt_embed_sub.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target_sub = output_memory_sub.gather(dim=1, \
                index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()
            target_sub = target_sub.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)
            target_sub = torch.concat([denoising_class, target_sub], 1)

        return target, target_sub, reference_points_unact.detach(), reference_points_unact_sub.detach(), enc_topk_bboxes, enc_topk_bboxes_sub, enc_topk_logits, enc_topk_verb_class

    def forward(self, feats, dn_args=None):

        # input projection and embedding
        (memory, spatial_shapes, level_start_index) = self._get_encoder_input(feats)
        
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_class, denoising_bbox_unact, denoising_bbox_unact_sub, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(dn_args[0], \
                    self.num_classes,
                    self.num_verb_classes,
                    self.num_queries,
                    self.label_enc,
                    None,
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, denoising_bbox_unact_sub, attn_mask, dn_meta = None, None, None, None, None

        target, target_sub, init_ref_points_unact, init_ref_points_unact_sub, enc_topk_bboxes, enc_topk_bboxes_sub, enc_topk_logits, enc_topk_verb_class = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact, denoising_bbox_unact_sub, topk_cat=self.topk_cat)
        
        # target, denoising_bbox_unact, attn_mask, dn_meta = \
        #     prepare_for_dn(dn_args, (target+target_sub)/2, init_ref_points_unact, memory.size(0), self.training, self.num_queries, self.num_classes, self.hidden_dim, self.label_enc, self.verb_enc)

        # decoder
        hs, out_bboxes, out_logits = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)
        
        sub_hs, out_bboxes_sub, _ = self.decoder_sub(
            target_sub,
            init_ref_points_unact_sub,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head_sub,
            None,
            self.query_pos_head_sub,
            attn_mask=attn_mask,
            with_score=False)

        target_verb = (hs + sub_hs) / 2.0
        inter_references = torch.cat([out_bboxes[-1], out_bboxes_sub[-1]], -1)
        coord = inter_references.clone().detach()
        coord = adaptive_shifted_MBR(coord)

        _, _, outputs_verb = self.decoder_verb(
            target_verb,
            coord,
            memory,
            spatial_shapes,
            level_start_index,
            None,
            self.verb_embed,
            self.query_pos_head_verb,
            attn_mask=attn_mask,
            with_score=True,
            with_bbox=False)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_bboxes_sub, out_bboxes_sub = torch.split(out_bboxes_sub, dn_meta['dn_num_split'], dim=2)

            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_outputs_verb, outputs_verb = torch.split(outputs_verb, dn_meta['dn_num_split'], dim=2)


        out = {'pred_logits': out_logits[-1], 'pred_obj_boxes': out_bboxes[-1], 'pred_sub_boxes': out_bboxes_sub[-1], 'pred_verb_logits': outputs_verb[-1]}

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1], out_bboxes_sub[:-1], outputs_verb[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes], [enc_topk_bboxes_sub], [enc_topk_verb_class]))
            
            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes, dn_out_bboxes_sub, dn_outputs_verb)
                out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_coord_sub, outputs_verb_class):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_obj_boxes': b, 'pred_sub_boxes': c, 'pred_verb_logits': d} for a, b, c, d in zip(outputs_class, outputs_coord, outputs_coord_sub, outputs_verb_class)]

