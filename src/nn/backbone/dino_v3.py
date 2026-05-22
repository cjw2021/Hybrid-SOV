import os
import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import OrderedDict

from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d

from src.core import register

from .convnext import ConvNeXt
from enum import Enum
from typing import List, Optional, Union
from urllib.parse import urlparse
from pathlib import Path

__all__ = ['DINOv3']

DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"

convnext_sizes = {
    "tiny": dict(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
    ),
    "small": dict(
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
    ),
    "base": dict(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
    ),
    "large": dict(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
    ),
}

class Weights(Enum):
    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"

def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("https", "file")

def convert_path_or_url_to_url(path: str) -> str:
    if is_url(path):
        return path
    return Path(path).expanduser().resolve().as_uri()


def _make_dinov3_convnext_model_url(
    *,
    compact_arch_name: str = "convnext_base",
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
):
    model_name = "dinov3"
    weights_name = weights.value.lower()
    hash_suffix = f"-{hash}" if hash else ""

    model_dir = f"{model_name}_{compact_arch_name}"
    model_filename = f"{model_name}_{compact_arch_name}_pretrain_{weights_name}{hash_suffix}.pth"
    return os.path.join(DINOV3_BASE_URL, model_dir, model_filename)

def _make_dinov3_convnext(
    in_chans: int = 3,
    depths: List[int] = [3, 3, 27, 3],
    dims: List[int] = [128, 256, 512, 1024],
    compact_arch_name: str = "convnext_base",
    drop_path_rate: float = 0.0,
    layer_scale_init_value: float = 1e-6,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
    **kwargs,
):
    model_kwargs = dict(
        in_chans=in_chans,
        depths=depths,
        dims=dims,
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
    )
    model_kwargs.update(**kwargs)
    model = ConvNeXt(**model_kwargs)
    if pretrained:
        url = _make_dinov3_convnext_model_url(
            compact_arch_name=compact_arch_name,
            weights=weights,
            hash=hash,
        )
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model

def dinov3_convnext_tiny(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "21b726bb"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    size_dict = convnext_sizes["tiny"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_tiny",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_small(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "296db49d"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    size_dict = convnext_sizes["small"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_small",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_base(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "801f2ba9"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    size_dict = convnext_sizes["base"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_base",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_large(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "61fa432d"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    size_dict = convnext_sizes["large"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_large",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


@register
class DINOv3(nn.Module):
    def __init__(
        self,
        model_type='dinov3_convnext_tiny'):
        super().__init__()

        if model_type == 'dinov3_convnext_tiny':
            self.model = dinov3_convnext_tiny(pretrained=True)
        elif model_type == 'dinov3_convnext_large':
            self.model = dinov3_convnext_large(pretrained=True)
        
        self.return_idx = [1, 2, 3]

        self._freeze_parameters(self)
            
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def forward(self, x):
        outs = []
        for idx, stage in enumerate(range(4)):
            x = self.model.downsample_layers[idx](x)
            x = self.model.stages[idx](x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


