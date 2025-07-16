import torch
import cv2
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class SRModelMicroscopy(SRModel):
    """SR model for microscopy with W&B image logging."""

    def __init__(self, opt):
        super(SRModelMicroscopy, self).__init__(opt)
        # Check if W&B is available
        try:
            import wandb
            self.use_wandb = True
            self.wandb = wandb
        except ImportError:
            self.use_wandb = False
            logger = get_root_logger()
            logger.warning('Wandb not found. Image logging will be disabled.')
        
        # Import metrics for training batch calculation
        from basicsr.metrics.microscopy_metrics import calculate_psnr_micro, calculate_ssim_micro, calculate_msssim, calculate_mse
        self.calculate_psnr = calculate_psnr_micro
        self.calculate_ssim = calculate_ssim_micro
        self.calculate_msssim = calculate_msssim
        self.calculate_mse = calculate_mse

    def calculate_batch_metrics(self, pred, target):
        """Calculate metrics for a batch of images during training."""
        # Ensure images are in [0, 1] range (they should be already)
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # Calculate metrics for each image in the batch
        batch_metrics = {'psnr': [], 'ssim': [], 'msssim': [], 'mse': []}
        
        for i in range(pred.shape[0]):
            # Get single image (keep as tensor)
            pred_img = pred[i:i+1]  # Keep batch dimension
            target_img = target[i:i+1]  # Keep batch dimension
            
            # Calculate metrics directly with tensors (no conversion to numpy)
            batch_metrics['psnr'].append(self.calculate_psnr(pred_img, target_img))
            batch_metrics['ssim'].append(self.calculate_ssim(pred_img, target_img))
            batch_metrics['msssim'].append(self.calculate_msssim(pred_img, target_img))
            batch_metrics['mse'].append(self.calculate_mse(pred_img, target_img))
        
        # Return average metrics
        return {
            'psnr': np.mean(batch_metrics['psnr']),
            'ssim': np.mean(batch_metrics['ssim']),
            'msssim': np.mean(batch_metrics['msssim']),
            'mse': np.mean(batch_metrics['mse'])
        }

    def optimize_parameters(self, current_iter):
        """Override optimize_parameters to include metric calculation."""
        # Call parent method
        super().optimize_parameters(current_iter)
        
        # Calculate metrics only every 500 iterations to speed up training (reduced from 100)
        if current_iter % 500 == 0 and hasattr(self, 'output') and hasattr(self, 'gt'):
            batch_metrics = self.calculate_batch_metrics(self.output, self.gt)
            
            # Add metrics to log_dict for logging
            for metric_name, metric_value in batch_metrics.items():
                self.log_dict[f'train_{metric_name}'] = metric_value
            
            # Log to W&B if available
            if self.use_wandb:
                wandb_log_dict = {
                    'train/psnr': batch_metrics['psnr'],
                    'train/ssim': batch_metrics['ssim'],
                    'train/msssim': batch_metrics['msssim'],
                    'train/mse': batch_metrics['mse'],
                    'train/step': current_iter
                }
                self.wandb.log(wandb_log_dict)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        # For W&B image logging
        wandb_images = []
        individual_metrics = []

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                lq_img = tensor2img([visuals['lq']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                current_metrics = {}
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_value = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += metric_value
                    current_metrics[name] = metric_value
                
                # Store individual metrics for W&B
                individual_metrics.append(current_metrics)

            # W&B image logging (only first 3 images)
            if self.use_wandb and idx < 3 and 'gt' in visuals:
                # Convert BGR to RGB for W&B
                lq_img_rgb = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
                sr_img_rgb = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
                gt_img_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

                # Get current metrics for caption
                current_psnr = current_metrics.get('psnr', 0)
                current_ssim = current_metrics.get('ssim', 0)
                current_msssim = current_metrics.get('msssim', 0)
                current_mse = current_metrics.get('mse', 0)

                # Create W&B images
                wandb_images.extend([
                    self.wandb.Image(
                        lq_img_rgb,
                        caption=f"Input_{idx+1}: (LR) - {lq_img.shape[1]}x{lq_img.shape[0]}"
                    ),
                    self.wandb.Image(
                        sr_img_rgb,
                        caption=f"Predicted_{idx+1}: (SR) - PSNR: {current_psnr:.2f}dB, SSIM: {current_ssim:.4f} - {sr_img.shape[1]}x{sr_img.shape[0]}"
                    ),
                    self.wandb.Image(
                        gt_img_rgb,
                        caption=f"GT_{idx+1}: (HR) - {gt_img.shape[1]}x{gt_img.shape[0]}"
                    )
                ])

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

            # Log to W&B
            if self.use_wandb:
                # Log average metrics
                wandb_log_dict = {
                    'val/images': wandb_images,
                    'val/psnr': self.metric_results.get('psnr', 0),
                    'val/ssim': self.metric_results.get('ssim', 0),
                    'val/msssim': self.metric_results.get('msssim', 0),
                    'val/mse': self.metric_results.get('mse', 0),
                    'val/step': current_iter
                }
                
                # Log individual image metrics
                for i, metrics in enumerate(individual_metrics[:3]):  # Only first 3
                    for metric_name, metric_value in metrics.items():
                        wandb_log_dict[f'val/img_{i+1}_{metric_name}'] = metric_value

                self.wandb.log(wandb_log_dict) 