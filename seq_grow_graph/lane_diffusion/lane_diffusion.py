"""
Complete LaneDiffusion Framework
Integrates LPIM, LPDM, and LPR for end-to-end training
"""

import torch
import torch.nn as nn
from .lpim import LPIM
from .lpdm import LPDM, LanePriorRefinement


class LaneDiffusion(nn.Module):
    """
    Complete LaneDiffusion framework
    Three-stage training:
        Stage I: Train LPIM
        Stage II: Train LPDM (freeze LPIM)
        Stage III: Train Decoder (freeze LPIM + LPDM)
    """
    def __init__(
        self,
        bev_channels=256,
        bev_h=128,
        bev_w=192,
        lpim_config=None,
        lpdm_config=None,
    ):
        super().__init__()
        
        # Default configs
        if lpim_config is None:
            lpim_config = {
                'bev_channels': bev_channels,
                'prior_dim': 256,
                'num_encoder_layers': 4,
                'num_heads': 8,
                'max_lanes': 50,
                'max_points_per_lane': 20,
            }
        
        if lpdm_config is None:
            lpdm_config = {
                'bev_channels': bev_channels,
                'bev_h': bev_h,
                'bev_w': bev_w,
                'num_steps': 15,
                'kappa': 0.5,
                'p': 1.0,
            }
        
        # Initialize modules
        self.lpim = LPIM(**lpim_config)
        self.lpdm = LPDM(**lpdm_config)
        self.lpr = LanePriorRefinement(in_channels=bev_channels, out_channels=bev_channels)
        
        # Training stage control
        self.current_stage = 'stage_i'  # 'stage_i', 'stage_ii', 'stage_iii', 'inference'
    
    def set_stage(self, stage):
        """
        Set training stage and freeze/unfreeze modules accordingly
        
        Args:
            stage: 'stage_i', 'stage_ii', 'stage_iii', or 'inference'
        """
        self.current_stage = stage
        
        if stage == 'stage_i':
            # Train LPIM only
            self.lpim.train()
            for param in self.lpim.parameters():
                param.requires_grad = True
            self.lpdm.eval()
            for param in self.lpdm.parameters():
                param.requires_grad = False
            self.lpr.eval()
            for param in self.lpr.parameters():
                param.requires_grad = False
                
        elif stage == 'stage_ii':
            # Train LPDM only (freeze LPIM)
            self.lpim.eval()
            for param in self.lpim.parameters():
                param.requires_grad = False
            self.lpdm.train()
            for param in self.lpdm.parameters():
                param.requires_grad = True
            self.lpr.eval()
            for param in self.lpr.parameters():
                param.requires_grad = False
                
        elif stage == 'stage_iii':
            # Freeze both LPIM and LPDM (decoder will be trained in main model)
            self.lpim.eval()
            for param in self.lpim.parameters():
                param.requires_grad = False
            self.lpdm.eval()
            for param in self.lpdm.parameters():
                param.requires_grad = False
            self.lpr.train()
            for param in self.lpr.parameters():
                param.requires_grad = True
                
        elif stage == 'inference':
            # Freeze all
            self.lpim.eval()
            for param in self.lpim.parameters():
                param.requires_grad = False
            self.lpdm.eval()
            for param in self.lpdm.parameters():
                param.requires_grad = False
            self.lpr.eval()
            for param in self.lpr.parameters():
                param.requires_grad = False
    
    def forward_stage_i(self, raw_bev, gt_centerlines):
        """
        Stage I: Train LPIM to generate prior-injected BEV
        
        Args:
            raw_bev: [B, C, H, W]
            gt_centerlines: List of GT centerlines
        
        Returns:
            prior_injected_bev: [B, C, H, W]
        """
        return self.lpim(raw_bev, gt_centerlines)
    
    def forward_stage_ii(self, raw_bev, gt_centerlines, gt_mask=None):
        """
        Stage II: Train LPDM to generate prior-injected BEV from raw BEV
        
        Args:
            raw_bev: [B, C, H, W]
            gt_centerlines: List of GT centerlines
            gt_mask: GT segmentation mask [B, 1, H, W] (optional)
        
        Returns:
            loss_dict: dictionary of losses
        """
        # Get target (x0) from frozen LPIM
        with torch.no_grad():
            x0 = self.lpim(raw_bev, gt_centerlines)
        
        # Compute diffusion loss (with CFG and SegHead)
        loss_dict = self.lpdm.compute_loss(x0, raw_bev, gt_mask=gt_mask, prob_uncond=0.1)
        
        return loss_dict
    
    def forward_stage_iii(self, raw_bev):
        """
        Stage III / Inference: Generate enhanced BEV using LPDM
        
        Args:
            raw_bev: [B, C, H, W]
        
        Returns:
            enhanced_bev: [B, C, H, W]
            seg_mask: [B, 1, H, W]
        """
        # Sample from LPDM
        with torch.no_grad():
            x_generated, seg_mask = self.lpdm.sample(raw_bev, guidance_scale=2.0)
        
        # Refine with LPR
        enhanced_bev = self.lpr(x_generated, raw_bev)
        
        return enhanced_bev, seg_mask
    
    def forward_cyclic(self, raw_bev, coarse_lines, refine_timestep=5):
        """
        Cyclic Refinement: Use coarse predictions to guide diffusion refinement
        
        Args:
            raw_bev: [B, C, H, W]
            coarse_lines: List of coarse predicted centerlines
            refine_timestep: Timestep to start refinement from (1-15)
            
        Returns:
            refined_bev: [B, C, H, W]
        """
        # 1. Pseudo-Prior Injection: Use coarse lines as "GT" for LPIM
        # Note: LPIM needs to be in eval mode but we need to run it
        # We assume self.lpim is already in eval mode during inference
        with torch.no_grad():
            prior_feat = self.lpim(raw_bev, coarse_lines)
            
        # 2. Add Noise (Forward Process) to t=refine_timestep
        # We need to construct a batch of timesteps
        B = raw_bev.shape[0]
        t_batch = torch.full((B,), refine_timestep, device=raw_bev.device, dtype=torch.long)
        
        # Use LPDM's forward_diffusion to get noisy state at t
        # x0 is prior_feat (from coarse lines)
        # xc is raw_bev
        # forward_diffusion returns (xt, x_res), we only need xt
        with torch.no_grad():
            noisy_feat, _ = self.lpdm.forward_diffusion(prior_feat, raw_bev, t_batch)
        
        # 3. Denoise (Reverse Process) from t=refine_timestep
        with torch.no_grad():
            refined_feat_raw, _ = self.lpdm.sample(
                xc=raw_bev, 
                num_steps=self.lpdm.num_steps,
                initial_state=noisy_feat,
                start_step=refine_timestep,
                guidance_scale=2.0
            )
            
        # 4. Refine with LPR
        refined_bev = self.lpr(refined_feat_raw, raw_bev)
        
        return refined_bev
    
    def forward(self, raw_bev, gt_centerlines=None, gt_mask=None):
        """
        Forward pass based on current stage
        
        Args:
            raw_bev: [B, C, H, W]
            gt_centerlines: List of GT centerlines (required for stage I and II)
            gt_mask: GT segmentation mask (required for stage II)
        
        Returns:
            output depends on stage
        """
        if self.current_stage == 'stage_i':
            return self.forward_stage_i(raw_bev, gt_centerlines)
        
        elif self.current_stage == 'stage_ii':
            return self.forward_stage_ii(raw_bev, gt_centerlines, gt_mask)
        
        elif self.current_stage in ['stage_iii', 'inference']:
            return self.forward_stage_iii(raw_bev)
        
        else:
            raise ValueError(f"Unknown stage: {self.current_stage}")
