
import torch
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from seq_grow_graph.lane_diffusion.lane_diffusion import LaneDiffusion

def test_forward_cyclic():
    print("Testing LaneDiffusion.forward_cyclic...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 2, 256, 32, 48
    
    # Initialize model
    model = LaneDiffusion(bev_channels=C, bev_h=H, bev_w=W).to(device)
    model.eval()
    
    # Dummy inputs
    raw_bev = torch.randn(B, C, H, W).to(device)
    
    # Mock coarse lines (list of tensors [N, 2])
    coarse_lines = []
    for i in range(B):
        # Random lines
        lines = []
        for j in range(3): # 3 lines per sample
            line = torch.rand(20, 2).to(device) * 100 # Scale to grid
            lines.append(line)
        coarse_lines.append(lines)
        
    # Test forward_cyclic
    try:
        refined_bev = model.forward_cyclic(raw_bev, coarse_lines, refine_timestep=5)
        
        print(f"Input shape: {raw_bev.shape}")
        print(f"Output shape: {refined_bev.shape}")
        
        assert refined_bev.shape == raw_bev.shape
        print("✅ forward_cyclic shape check passed")
        
        # Check if output is different from input (it should be, due to diffusion)
        diff = (refined_bev - raw_bev).abs().mean()
        print(f"Difference from raw_bev: {diff.item()}")
        assert diff > 0
        print("✅ Output is different from input")
        
    except Exception as e:
        print(f"❌ forward_cyclic failed: {e}")
        import traceback
        traceback.print_exc()

def test_seg_head_and_cfg():
    print("\nTesting SegHead and CFG...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, C, H, W = 2, 256, 32, 48
    
    # Initialize model
    model = LaneDiffusion(bev_channels=C, bev_h=H, bev_w=W).to(device)
    model.eval()
    
    raw_bev = torch.randn(B, C, H, W).to(device)
    
    # Test forward_stage_iii (Inference with SegHead)
    try:
        enhanced_bev, seg_mask = model.forward_stage_iii(raw_bev)
        
        print(f"SegMask shape: {seg_mask.shape}")
        assert seg_mask.shape == (B, 1, H, W)
        print("✅ SegHead output shape check passed")
        
    except Exception as e:
        print(f"❌ forward_stage_iii failed: {e}")
        import traceback
        traceback.print_exc()

    # Test compute_loss with Mask and CFG
    print("\nTesting compute_loss with Mask and CFG...")
    model.train()
    model.set_stage('stage_ii')
    
    gt_centerlines = [] # Empty for simplicity
    for i in range(B): gt_centerlines.append([])
        
    gt_mask = torch.randint(0, 2, (B, 1, H, W)).float().to(device)
    
    try:
        loss_dict = model.forward_stage_ii(raw_bev, gt_centerlines, gt_mask=gt_mask)
        print(f"Loss dict: {loss_dict}")
        
        assert 'loss_diff' in loss_dict
        assert 'loss_seg' in loss_dict
        print("✅ Loss computation check passed")
        
    except Exception as e:
        print(f"❌ forward_stage_ii failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_forward_cyclic()
    test_seg_head_and_cfg()
