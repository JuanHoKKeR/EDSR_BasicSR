from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim
from .microscopy_metrics import calculate_psnr as calculate_psnr_micro, calculate_ssim as calculate_ssim_micro, calculate_msssim, calculate_mse

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_psnr_micro', 'calculate_ssim_micro', 'calculate_msssim', 'calculate_mse']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
