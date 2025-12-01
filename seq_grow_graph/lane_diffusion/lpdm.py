"""
Lane Prior Diffusion Module (LPDM)
Implements ResShift-inspired diffusion for BEV feature generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from .swin_unet import SwinTransformerUNet


class LPDM(nn.Module):
    """
    Lane Prior Diffusion Module
    Models the distribution of prior-injected BEV features using diffusion
    """
    def __init__(
        self,
        bev_channels=256,
        bev_h=128,
        bev_w=192,
        num_steps=15,
        kappa=0.5,
        p=1.0,
        denoiser_config=None,
    ):
        super().__init__()
        self.bev_channels = bev_channels
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_steps = num_steps
        self.kappa = kappa
        self.p = p
        
        # Denoising network (Swin Transformer U-Net)
        if denoiser_config is None:
            denoiser_config = {
                'in_channels': bev_channels,
                'out_channels': bev_channels,
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
                'window_size': 7,
            }
        self.denoiser = SwinTransformerUNet(**denoiser_config)
        
        # Explicit Semantic Auxiliary Injection (SegHead)
        # Predicts a 1-channel binary mask from the denoised features
        self.seg_head = nn.Conv2d(bev_channels, 1, kernel_size=1)
        
        # Compute and register noise schedule (Cosine Schedule)
        eta_schedule = self._compute_eta_schedule(num_steps, kappa, p)
        self.register_buffer('eta_schedule', eta_schedule)
        
        # Compute gamma schedule
        gamma_schedule = self._compute_gamma_schedule(eta_schedule)
        self.register_buffer('gamma_schedule', gamma_schedule)
        
    def _compute_eta_schedule(self, T, kappa, p):
        """
        Compute the shifting schedule eta_t using Cosine Schedule
        Improved schedule for better detail preservation
        """
        # Use cosine schedule instead of power schedule
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        
        # Convert alpha_bar to eta (noise level)
        # In diffusion, sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * eps
        # Our formulation: x0 + eta * res + noise
        # We map the standard schedule to our eta schedule
        
        # For simplicity in this framework, we stick to a modified power schedule 
        # that mimics cosine behavior at the end (smoother approach to 0)
        # but keeps the original structure to avoid breaking the reverse step logic
        
        eta = torch.zeros(T + 1)
        eta[1] = min(0.04 / kappa, np.sqrt(0.001))
        eta[T] = np.sqrt(0.999)
        
        # Use a smoother interpolation
        t_vals = torch.linspace(0, 1, T+1)
        # Cosine-like interpolation between eta[1] and eta[T]
        eta_interp = eta[1] + (eta[T] - eta[1]) * (1 - torch.cos(t_vals * np.pi / 2))
        
        eta = eta_interp ** 2 # Square it as per original logic
        eta[0] = 0.0  # ensure boundary condition eta_0 = 0 for proper reverse step
        
        return eta
    
    def _compute_gamma_schedule(self, eta):
        """
        Compute gamma schedule from eta
        gamma_1 = eta_1
        gamma_t = eta_t - eta_{t-1} for t > 1
        """
        gamma = torch.zeros_like(eta)
        gamma[1] = eta[1]
        for t in range(2, len(eta)):
            gamma[t] = eta[t] - eta[t - 1]
        return gamma
    
    def forward_diffusion(self, x0, xc, t):
        """
        Forward diffusion process: q(x_t | x_0, x_c)
        
        Args:
            x0: target features (prior-injected BEV) [B, C, H, W]
            xc: condition features (raw BEV) [B, C, H, W]
            t: timestep [B], values in [1, T]
        
        Returns:
            xt: noisy features [B, C, H, W]
            x_res: residual x_c - x_0 [B, C, H, W]
        """
        B = x0.shape[0]
        device = x0.device
        
        # Compute residual
        x_res = xc - x0
        
        # Get eta_t for each sample
        eta_t = self.eta_schedule[t].to(device).view(B, 1, 1, 1)
        
        # Sample noise
        noise = torch.randn_like(x0) * self.kappa * torch.sqrt(eta_t)
        
        # Compute x_t = x_0 + eta_t * x_res + noise
        xt = x0 + eta_t * x_res + noise
        
        return xt, x_res
    
    def predict_x0(self, xt, xc, t):
        """
        Predict x_0 from x_t using the denoising network
        
        Args:
            xt: noisy features [B, C, H, W]
            xc: condition features [B, C, H, W]
            t: timestep [B]
        
        Returns:
            predicted x0 [B, C, H, W]
        """
        t_normalized = t.float() / self.num_steps  # Normalize to [0, 1]
        x0_pred = self.denoiser(xt, xc, t_normalized)
        return x0_pred
    
    def reverse_step(self, xt, xc, t, guidance_scale=1.0):
        """
        Single reverse diffusion step: p(x_{t-1} | x_t, x_c)
        
        Args:
            xt: current noisy features [B, C, H, W]
            xc: condition features [B, C, H, W]
            t: current timestep [B], values in [1, T]
            guidance_scale: CFG scale (w). If > 1.0, uses classifier-free guidance.
        
        Returns:
            x_{t-1}: denoised features [B, C, H, W]
        """
        B = xt.shape[0]
        device = xt.device
        
        # Predict x_0
        if guidance_scale > 1.0:
            # Classifier-Free Guidance
            # 1. Conditional prediction
            x0_pred_cond = self.predict_x0(xt, xc, t)
            
            # 2. Unconditional prediction (xc = 0)
            xc_uncond = torch.zeros_like(xc)
            x0_pred_uncond = self.predict_x0(xt, xc_uncond, t)
            
            # 3. Combine: uncond + w * (cond - uncond)
            x0_pred = x0_pred_uncond + guidance_scale * (x0_pred_cond - x0_pred_uncond)
        else:
            x0_pred = self.predict_x0(xt, xc, t)
        
        # Get schedule values
        eta_t = self.eta_schedule[t].to(device).view(B, 1, 1, 1)
        eta_t_minus_1 = self.eta_schedule[t - 1].to(device).view(B, 1, 1, 1)
        gamma_t = self.gamma_schedule[t].to(device).view(B, 1, 1, 1)
        
        # Compute mean: mu = (eta_{t-1} / eta_t) * x_t + (gamma_t / eta_t) * x0_pred
        mu = (eta_t_minus_1 / eta_t) * xt + (gamma_t / eta_t) * x0_pred
        
        # Add noise (except for t=1)
        if t[0].item() > 1:
            # Compute variance
            sigma = self.kappa * torch.sqrt((eta_t_minus_1 * gamma_t) / eta_t)
            noise = torch.randn_like(xt)
            x_t_minus_1 = mu + sigma * noise
        else:
            x_t_minus_1 = mu
        
        return x_t_minus_1
    
    @torch.no_grad()
    def sample(self, xc, num_steps=None, initial_state=None, start_step=None, guidance_scale=2.0):
        """
        Sample from the diffusion model (reverse process)
        
        Args:
            xc: condition features (raw BEV) [B, C, H, W]
            num_steps: number of diffusion steps (default: self.num_steps)
            initial_state: starting state for SDEEdit [B, C, H, W] (optional)
            start_step: starting timestep for SDEEdit (optional)
            guidance_scale: CFG scale (default: 2.0)
        
        Returns:
            x0: generated features [B, C, H, W]
            seg_mask: predicted segmentation mask [B, 1, H, W]
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        B, C, H, W = xc.shape
        device = xc.device
        
        if initial_state is not None and start_step is not None:
            # SDEEdit mode: start from intermediate noisy state
            xt = initial_state
            current_step = start_step
        else:
            # Standard generation: start from pure noise (t=T)
            # Initialize from noisy condition: x_T ~ N(x_c, kappa^2 * eta_T * I)
            eta_T = self.eta_schedule[num_steps].to(device)
            noise = torch.randn_like(xc) * self.kappa * torch.sqrt(eta_T)
            xt = xc + noise
            current_step = num_steps
        
        # Reverse diffusion
        for t in range(current_step, 0, -1):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            xt = self.reverse_step(xt, xc, t_batch, guidance_scale=guidance_scale)
            
        # Predict segmentation mask from the final denoised features
        seg_mask = self.seg_head(xt)
        
        return xt, seg_mask
    
    def compute_loss(self, x0, xc, gt_mask=None, t=None, prob_uncond=0.1):
        """
        Compute diffusion loss: ||f_theta(x_t, x_c, t) - x_0||^2
        
        Args:
            x0: target features [B, C, H, W]
            xc: condition features [B, C, H, W]
            gt_mask: ground truth segmentation mask [B, 1, H, W] (optional)
            t: timestep [B] (if None, randomly sampled)
            prob_uncond: probability of unconditional training (CFG)
        
        Returns:
            loss_dict: dictionary of losses
        """
        B = x0.shape[0]
        device = x0.device
        
        # Sample timestep if not provided
        if t is None:
            t = torch.randint(1, self.num_steps + 1, (B,), device=device, dtype=torch.long)
        
        # Classifier-Free Guidance Training
        # Randomly drop condition with probability prob_uncond
        if np.random.rand() < prob_uncond:
            xc_train = torch.zeros_like(xc)
        else:
            xc_train = xc
            
        # Forward diffusion to get noisy x_t
        # Note: We use the REAL xc for forward diffusion (to simulate the process correctly)
        # but use the (potentially zeroed) xc_train for the network prediction
        xt, x_res = self.forward_diffusion(x0, xc, t)
        
        # Predict x_0
        x0_pred = self.predict_x0(xt, xc_train, t)
        
        # Diffusion Loss (MSE)
        loss_diff = F.mse_loss(x0_pred, x0)
        
        # Segmentation Loss (Auxiliary)
        loss_seg = torch.tensor(0.0, device=device)
        if gt_mask is not None:
            # Predict mask from the predicted x0
            mask_pred = self.seg_head(x0_pred)
            
            # Resize gt_mask to match feature size if needed
            if gt_mask.shape[-2:] != mask_pred.shape[-2:]:
                gt_mask = F.interpolate(gt_mask.float(), size=mask_pred.shape[-2:], mode='nearest')
            
            # BCE Loss
            loss_seg = F.binary_cross_entropy_with_logits(mask_pred, gt_mask.float())
            
        return {'loss_diff': loss_diff, 'loss_seg': loss_seg}


class LanePriorRefinement(nn.Module):
    """
    Lane Prior Refinement (LPR)
    Fuses the generated BEV feature with the original BEV feature
    """
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
    
    def forward(self, x_generated, x_original):
        """
        Args:
            x_generated: generated BEV from diffusion [B, C, H, W]
            x_original: original BEV from LSS [B, C, H, W]
        Returns:
            refined BEV [B, C, H, W]
        """
        # Concatenate
        x = torch.cat([x_generated, x_original], dim=1)
        
        # Encode
        x = self.encoder(x)
        
        # Decode
        x = self.decoder(x)
        
        # Residual connection
        x = x + x_original
        
        return x
