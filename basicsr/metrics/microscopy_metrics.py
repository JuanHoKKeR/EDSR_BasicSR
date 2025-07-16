import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as torch_ssim, ms_ssim
from basicsr.utils.registry import METRIC_REGISTRY
from basicsr.utils import img2tensor, tensor2img


@METRIC_REGISTRY.register()
def calculate_psnr_micro(img, img2, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    
    Args:
        img (ndarray): Images with range [0, 255] (numpy) or [0, 1] (tensor).
        img2 (ndarray): Images with range [0, 255] (numpy) or [0, 1] (tensor).
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    
    Returns:
        float: psnr result.
    """
    
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    # Handle crop_border
    crop_border = kwargs.get('crop_border', 0)
    if crop_border > 0:
        if isinstance(img, np.ndarray):
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        else:
            img = img[..., crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    
    # Convert numpy arrays to tensors if needed
    if isinstance(img, np.ndarray):
        # Assume numpy arrays are in [0, 255] range - convert to [0, 1]
        img = torch.from_numpy(img.astype(np.float32) / 255.0)
    if isinstance(img2, np.ndarray):
        # Assume numpy arrays are in [0, 255] range - convert to [0, 1]
        img2 = torch.from_numpy(img2.astype(np.float32) / 255.0)
    
    # Ensure tensors are in [0, 1] range
    img = torch.clamp(img, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # Flatten to calculate MSE across all pixels
    mse = torch.mean((img - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


@METRIC_REGISTRY.register()
def calculate_mse(img, img2, **kwargs):
    """Calculate MSE (Mean Squared Error).
    
    Args:
        img (ndarray): Images with range [0, 255] (numpy) or [0, 1] (tensor).
        img2 (ndarray): Images with range [0, 255] (numpy) or [0, 1] (tensor).
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the MSE calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    
    Returns:
        float: mse result.
    """
    
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    # Handle crop_border
    crop_border = kwargs.get('crop_border', 0)
    if crop_border > 0:
        if isinstance(img, np.ndarray):
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        else:
            img = img[..., crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    
    # Convert numpy arrays to tensors if needed
    if isinstance(img, np.ndarray):
        # Assume numpy arrays are in [0, 255] range - convert to [0, 1]
        img = torch.from_numpy(img.astype(np.float32) / 255.0)
    if isinstance(img2, np.ndarray):
        # Assume numpy arrays are in [0, 255] range - convert to [0, 1]
        img2 = torch.from_numpy(img2.astype(np.float32) / 255.0)
    
    # Ensure tensors are in [0, 1] range
    img = torch.clamp(img, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    return torch.mean((img - img2) ** 2).item()


@METRIC_REGISTRY.register()
def calculate_ssim_micro(img, img2, **kwargs):
    """Calculate SSIM (Structural Similarity Index).
    
    Args:
        img (ndarray): Images with range [0, 255] (numpy) or [0, 1] (tensor).
        img2 (ndarray): Images with range [0, 255] (numpy) or [0, 1] (tensor).
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    
    Returns:
        float: ssim result.
    """
    
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    # Handle crop_border
    crop_border = kwargs.get('crop_border', 0)
    if crop_border > 0:
        if isinstance(img, np.ndarray):
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        else:
            img = img[..., crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    
    # Convert numpy arrays to tensors if needed
    if isinstance(img, np.ndarray):
        # Assume numpy arrays are in [0, 255] range - convert to [0, 1]
        img = torch.from_numpy(img.astype(np.float32) / 255.0)
        # Convert HWC to CHW format for pytorch tensors
        if img.ndim == 3:
            img = img.permute(2, 0, 1)
    if isinstance(img2, np.ndarray):
        # Assume numpy arrays are in [0, 255] range - convert to [0, 1]
        img2 = torch.from_numpy(img2.astype(np.float32) / 255.0)
        # Convert HWC to CHW format for pytorch tensors
        if img2.ndim == 3:
            img2 = img2.permute(2, 0, 1)
    
    # Ensure tensors are in [0, 1] range
    img = torch.clamp(img, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # Add batch dimension if needed
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
    
    return torch_ssim(img, img2, data_range=1.0, size_average=True).item()


@METRIC_REGISTRY.register()
def calculate_msssim(img, img2, **kwargs):
    """Calculate MS-SSIM (Multi-Scale Structural Similarity Index).
    
    Args:
        img (ndarray): Images with range [0, 255] (numpy) or [0, 1] (tensor).
        img2 (ndarray): Images with range [0, 255] (numpy) or [0, 1] (tensor).
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the MS-SSIM calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    
    Returns:
        float: ms-ssim result.
    """
    
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    # Handle crop_border
    crop_border = kwargs.get('crop_border', 0)
    if crop_border > 0:
        if isinstance(img, np.ndarray):
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]
        else:
            img = img[..., crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]
    
    # Convert numpy arrays to tensors if needed
    if isinstance(img, np.ndarray):
        # Assume numpy arrays are in [0, 255] range - convert to [0, 1]
        img = torch.from_numpy(img.astype(np.float32) / 255.0)
        # Convert HWC to CHW format for pytorch tensors
        if img.ndim == 3:
            img = img.permute(2, 0, 1)
    if isinstance(img2, np.ndarray):
        # Assume numpy arrays are in [0, 255] range - convert to [0, 1]
        img2 = torch.from_numpy(img2.astype(np.float32) / 255.0)
        # Convert HWC to CHW format for pytorch tensors
        if img2.ndim == 3:
            img2 = img2.permute(2, 0, 1)
    
    # Ensure tensors are in [0, 1] range
    img = torch.clamp(img, 0, 1)
    img2 = torch.clamp(img2, 0, 1)
    
    # Add batch dimension if needed
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
    
    # Check if image is too small for MS-SSIM
    h, w = img.shape[2], img.shape[3]
    if h < 160 or w < 160:
        # Use regular SSIM for small images
        return torch_ssim(img, img2, data_range=1.0, size_average=True).item()
    
    return ms_ssim(img, img2, data_range=1.0, size_average=True).item()


class MicroscopyMetricsCalculator:
    """Calculadora de métricas para super-resolución de microscopía"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def calculate_all_metrics(self, pred, target, use_torch_ssim=True):
        """Calcula todas las métricas de una vez."""
        metrics = {}
        
        # Ensure images are in [0, 1] range
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # PSNR
        metrics['PSNR'] = calculate_psnr_micro(pred, target)
        
        # MSE
        metrics['MSE'] = calculate_mse(pred, target)
        
        # SSIM
        metrics['SSIM'] = calculate_ssim_micro(pred, target)
        
        # MS-SSIM
        metrics['MS-SSIM'] = calculate_msssim(pred, target)
        
        return metrics
    
    @staticmethod
    def calculate_metrics_batch(pred_batch, target_batch, device='cuda'):
        """Calcula métricas promedio para un batch de imágenes."""
        calculator = MicroscopyMetricsCalculator(device=device)
        batch_metrics = {'PSNR': [], 'MSE': [], 'SSIM': [], 'MS-SSIM': []}
        
        for i in range(pred_batch.shape[0]):
            pred = pred_batch[i:i+1]
            target = target_batch[i:i+1]
            metrics = calculator.calculate_all_metrics(pred, target)
            for key in batch_metrics:
                batch_metrics[key].append(metrics[key])
        
        # Calcular promedios
        avg_metrics = {}
        for key, values in batch_metrics.items():
            avg_metrics[key] = np.mean(values)
        
        return avg_metrics 