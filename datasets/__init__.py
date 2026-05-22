import torch.utils.data

from .hico import build as build_hico
from .vcoco_self_object import build as build_vcoco_self_object


def build_dataset(image_set, args):
    if args.dataset_file == 'hico':
        return build_hico(image_set, args)
    if args.dataset_file == 'vcoco_self_object':
        return build_vcoco_self_object(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
