import logging
import torch

from .blip2_qformer import Blip2Qformer
from omegaconf import OmegaConf

from .blip_processors import BlipImageEvalProcessor, BlipCaptionProcessor


def load_preprocess(config):
    """
    Load preprocessor configs and construct preprocessors.

    If no preprocessor is specified, return BaseProcessor, which does not do any preprocessing.

    Args:
        config (dict): preprocessor configs.

    Returns:
        vis_processors (dict): preprocessors for visual inputs.
        txt_processors (dict): preprocessors for text inputs.

        Key is "train" or "eval" for processors used in training and evaluation respectively.
    """

    vis_processors = dict()
    txt_processors = dict()

    vis_processors["eval"] = BlipImageEvalProcessor(image_size=224)
    txt_processors["eval"] = BlipCaptionProcessor()

    return vis_processors, txt_processors

def load_blip2_and_preprocess(name, model_type, is_eval=False, device="cpu"):
    model_cls = Blip2Qformer()

    # load model
    model = model_cls.from_pretrained(model_type=model_type)

    if is_eval:
        model.eval()

    # load preprocess
    cfg = OmegaConf.load(model_cls.default_config_path(model_type))
    if cfg is not None:
        preprocess_cfg = cfg.preprocess

        vis_processors, txt_processors = load_preprocess(preprocess_cfg)
    else:
        vis_processors, txt_processors = None, None
        logging.info(
            f"""No default preprocess for model {name} ({model_type}).
                This can happen if the model is not finetuned on downstream datasets,
                or it is not intended for direct use without finetuning.
            """
        )

    if device == "cpu" or device == torch.device("cpu"):
        model = model.float()

    return model.to(device), vis_processors, txt_processors
