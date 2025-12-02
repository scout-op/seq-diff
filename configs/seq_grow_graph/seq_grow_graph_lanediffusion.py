"""
Configuration for SeqGrowGraph with LaneDiffusion
Extends the default SeqGrowGraph config to include Lane Diffusion modules
"""

_base_ = ['./seq_grow_graph_default.py']

# Enable LaneDiffusion
model = dict(
    type='SeqGrowGraph',
    use_lane_diffusion=True,
    freeze_pretrain=True,
    
    # LaneDiffusion stage: 'stage_i', 'stage_ii', 'stage_iii', or 'inference'
    lane_diffusion_stage='inference',  # Change this for different training stages
    
    # Lane Diffusion configuration
    lane_diffusion_cfg=dict(
        # LPIM configuration
        lpim_config=dict(
            bev_channels=256,  # Will be overridden by lss_cfg['d_out']
            prior_dim=256,
            num_encoder_layers=4,
            num_heads=8,
            max_lanes=50,
            max_points_per_lane=20,
        ),
        
        # LPDM configuration  
        lpdm_config=dict(
            bev_channels=256,  # Will be overridden
            bev_h=128,  # Will be computed from grid_conf
            bev_w=192,  # Will be computed from grid_conf
            num_steps=15,  # Number of diffusion steps
            kappa=0.5,  # Noise variance parameter
            p=1.0,  # Shifting schedule growth rate
            
            # Swin Transformer U-Net configuration
            denoiser_config=dict(
                in_channels=256,
                out_channels=256,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
            ),
        ),
    ),
)

# Training configuration for different stages
# Uncomment the appropriate section based on your training stage

# ===== Stage I: Train LPIM =====
model['lane_diffusion_stage'] = 'stage_i'
load_from = '/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/lane2/work_dirs/seq_grow_graph_lanediffusion_s1_v7/epoch_19.pth'

# ===== Stage II: Train LPDM =====
# model['lane_diffusion_stage'] = 'stage_ii'
# # Load Stage I checkpoint here
# load_from = "/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/lane2/work_dirs/seq_grow_graph_lanediffusion_s2_v6/epoch_5.pth"

# ===== Stage III: Train Decoder with enhanced features =====
# model['lane_diffusion_stage'] = 'stage_iii'
# # Load Stage II checkpoint here (Update this path to your actual Stage II checkpoint)
# load_from = "/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/lane2/work_dirs/seq_grow_graph_lanediffusion_s3_v4/epoch_11.pth"

# ===== Inference =====
# model['lane_diffusion_stage'] = 'inference'
# load_from = "/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/lane2/work_dirs/seq_grow_graph_lanediffusion_s3_v4/epoch_10.pth"

# You may need to adjust batch size depending on GPU memory
train_dataloader = dict(
    batch_size=64,  # Reduced from 18 due to LaneDiffusion memory usage
)

val_dataloader = dict(
    batch_size=16,
)


test_dataloader = dict(
    batch_size=16,
)

# Optimizer - Restore learning rate for Stage III (fine-tuning decoder)
optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.01),  # Slightly lower than Stage I (2e-4 -> 1e-4)
    clip_grad=dict(max_norm=35, norm_type=2),  # Restore normal clipping
)


# work_dir = '/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/lane2/work_dirs/seq_grow_graph_lanediffusion_s3_v2'
# work_dir = '/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/lane2/work_dirs/seq_grow_graph_lanediffusion_infer'

# Work directory
work_dir = '/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/lane2/work_dirs/seq_grow_graph_lanediffusion_bl_add_stage1'

# DDP Settings
find_unused_parameters = True

# Increase NCCL timeout to 30 minutes (default is usually 30 min but watchdog might be shorter)
# This helps prevent "watchdog got stuck" errors
env_cfg = dict(
    dist_cfg=dict(backend='nccl', timeout=1800)  # 1800 seconds = 30 minutes
)
