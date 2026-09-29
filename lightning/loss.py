import torch
import torch.nn as nn
from pytorch_msssim import MS_SSIM
from torch.nn import functional as F

from torch.cuda.amp import autocast

class Losses(nn.Module):
    def __init__(self, cfg=None):
        super(Losses, self).__init__()

        self.cfg = cfg
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

        self.ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3)

    def forward(self, batch, output, iter):

        scalar_stats = {}
        loss = 0

        B,V,H,W = batch['tar_rgb'].shape[:-1]

        tar_rgb = batch['tar_rgb'].permute(0,2,1,3,4).reshape(B,H,V*W,3)
        tar_rgb_view = batch['tar_rgb']

        use_conf = getattr(self.cfg, 'use_confidence', False) and output.get('confidence_active', True)
        if use_conf:
            conf_w = output['view_confidence'].detach() # Shape: (B, V)
            conf_w = conf_w / (conf_w.sum(dim=1, keepdim=True) + 1e-8) * V
        
        if 'image' in output:

            for prex in ['','_fine']:
                
                if prex=='_fine' and f'acc_map{prex}' not in output:
                    continue

                color_loss_all = (output[f'image{prex}']-tar_rgb)**2
                if use_conf:
                    color_loss_view = (output[f'image{prex}'].reshape(B, H, V, W, 3).permute(0, 2, 1, 3, 4) - tar_rgb_view)**2
                    color_loss_view = color_loss_view.mean(dim=(2, 3, 4)) # Shape: (B, V)
                    weighted_mse = (color_loss_view * conf_w).mean()
                    loss += weighted_mse
                    scalar_stats.update({f'mse{prex}': weighted_mse.detach()})
                else:
                    loss += color_loss_all.mean()
                    scalar_stats.update({f'mse{prex}': color_loss_all.mean().detach()})
                
                psnr = -10. * torch.log(color_loss_all.detach().mean()) / \
                    torch.log(torch.Tensor([10.]).to(color_loss_all.device))
                scalar_stats.update({f'psnr{prex}': psnr})

                with autocast(enabled=False): 
                    ssim_val = self.ssim(output[f'image{prex}'].permute(0,3,1,2), tar_rgb.permute(0,3,1,2))
                    scalar_stats.update({f'ssim{prex}': ssim_val.detach()})
                    loss += 0.5 * (1-ssim_val)
                
                if f'rend_dist{prex}' in output and iter>1000 and prex!='_fine':
                    distortion = output[f"rend_dist{prex}"].mean()
                    scalar_stats.update({f'distortion{prex}': distortion.detach()})
                    loss += distortion*1000
                    
                    rend_normal  = output[f'rend_normal{prex}']
                    depth_normal = output[f'depth_normal{prex}']
                    acc_map = output[f'acc_map{prex}'].detach()

                    normal_error = ((1 - (rend_normal * depth_normal).sum(dim=-1))*acc_map).mean() 
                    scalar_stats.update({f'normal{prex}': normal_error.detach()})
                    loss += normal_error*0.2

        scalar_stats['loss_total'] = loss.detach() if isinstance(loss, torch.Tensor) else torch.tensor(float(loss))
        return loss, scalar_stats

