import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.structures import InstanceData
import numpy as np

from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures.ops import bbox3d2result
from .grid_mask import GridMask
from .LiftSplatShoot import LiftSplatShootEgo
from .core import seq2nodelist, seq2bznodelist, seq2plbznodelist, av2seq2bznodelist
from .core import EvalSeq2Graph_with_start as EvalSeq2Graph

from .encode_centerline import convert_coeff_coord
from .bz_roadnet_reach_dist_eval import get_geom, get_range
from .lane_diffusion import LaneDiffusion


@MODELS.register_module()
class SeqGrowGraph(MVXTwoStageDetector):
    """Petr3D. nan for all token except label"""
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 lss_cfg=None,
                 grid_conf=None,
                 bz_grid_conf=None,
                 data_aug_conf=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 vis_cfg=None,
                 freeze_pretrain=True,
                 bev_scale=1.0,
                 epsilon=2,
                 max_box_num=700, #>=660+2
                 init_cfg=None,
                 data_preprocessor=None,front_camera_only=False,vis_dir="original",
                 use_lane_diffusion=False,
                 lane_diffusion_cfg=None,
                 lane_diffusion_stage='inference',
                 ):
        super(SeqGrowGraph, self).__init__(pts_voxel_layer, pts_middle_encoder,
                                                        pts_fusion_layer, img_backbone, pts_backbone,
                                                        img_neck, pts_neck, pts_bbox_head, img_roi_head,
                                                        img_rpn_head, train_cfg, test_cfg, init_cfg,
                                                        data_preprocessor)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.front_camera_only=front_camera_only
        self.vis_dir=vis_dir
        # data_aug_conf = {
        #     'final_dim': (128, 352),
        #     'H': 900, 'W': 1600,
        # }
        # self.up = Up(512, 256, scale_factor=2)
        # view_transformers = []
        # view_transformers.append(
        #     LiftSplatShoot(grid_conf, data_aug_conf, downsample=16, d_in=256, d_out=256, return_bev=True))
        # self.view_transformers = nn.ModuleList(view_transformers)
        # self.view_transformers = LiftSplatShoot(grid_conf, data_aug_conf, downsample=16, d_in=256, d_out=256, return_bev=True)
        self.view_transformers = LiftSplatShootEgo(grid_conf, data_aug_conf, return_bev=True, **lss_cfg)
        self.downsample = lss_cfg['downsample']
        self.final_dim = data_aug_conf['final_dim']

        self.split_connect=571
        self.split_node=572
        self.start = 574
        self.end = 573
        self.summary_split = 570
        self.split_lines=569
        

      
        
        # self.box_range = 200
        # self.coeff_range = 200
        # self.num_classes=4
        # self.category_start = 200
        # self.connect_start = 250 
        self.coeff_start = 350 
        self.idx_start=250
        self.no_known = 575  # n/a and padding share the same label to be eliminated from loss calculation
        self.num_center_classes = 576 
        # self.noise_connect = 572 
        self.noise_label = 569
        # self.noise_coeff = 570
        
        
        self.vis_cfg = vis_cfg
        self.bev_scale = bev_scale
        self.epsilon = epsilon
        self.max_box_num = max_box_num #!暂时没用到

        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf

        self.dx, bx, nx, self.pc_range, ego_points = get_geom(grid_conf)
        self.bz_dx, bz_bx, bz_nx, self.bz_pc_range = get_range(bz_grid_conf)

        # Lane Diffusion integration
        self.use_lane_diffusion = use_lane_diffusion
        if use_lane_diffusion:
            # Get BEV dimensions from grid_conf
            bev_h = int((grid_conf['ybound'][1] - grid_conf['ybound'][0]) / grid_conf['ybound'][2])
            bev_w = int((grid_conf['xbound'][1] - grid_conf['xbound'][0]) / grid_conf['xbound'][2])
            
            if lane_diffusion_cfg is None:
                lane_diffusion_cfg = {}
            
            lane_diffusion_cfg.update({
                'bev_channels': lss_cfg.get('d_out', 256),
                'bev_h': bev_h,
                'bev_w': bev_w,
            })

            # ensure LPIM aware of coordinate range
            lpim_cfg = lane_diffusion_cfg.get('lpim_config', {})
            if grid_conf is not None:
                lpim_cfg.setdefault('grid_conf', grid_conf)
            lane_diffusion_cfg['lpim_config'] = lpim_cfg
            
            self.lane_diffusion = LaneDiffusion(**lane_diffusion_cfg)
            self.lane_diffusion.set_stage(lane_diffusion_stage)
            
            # Channel adapter: BEV (256) + SegMask (1) -> BEV (256)
            # This is needed when seg_mask is concatenated to bev_feats
            bev_channels = lss_cfg.get('d_out', 256)
            self.bev_mask_adapter = nn.Conv2d(bev_channels + 1, bev_channels, kernel_size=1)
            
            # Stage II specific freezing logic
            if lane_diffusion_stage == 'stage_ii':
                print("🔒 Stage II: Freezing Backbone, Neck, LSS, and Decoder. Training LPDM only.")
                # Freeze everything else
                for p in self.parameters():
                    p.requires_grad = False
                # Unfreeze LPDM (set_stage handles this, but we ensure it here)
                for p in self.lane_diffusion.lpdm.parameters():
                    p.requires_grad = True
                # Ensure LPIM and LPR are frozen
                for p in self.lane_diffusion.lpim.parameters():
                    p.requires_grad = False
                for p in self.lane_diffusion.lpr.parameters():
                    p.requires_grad = False
            
            # Stage III specific freezing logic
            elif lane_diffusion_stage == 'stage_iii':
                print("🔒 Stage III: Freezing LPIM and LPDM. Training Decoder with enhanced features.")
                # Freeze LPIM and LPDM weights only
                for p in self.lane_diffusion.lpim.parameters():
                    p.requires_grad = False
                for p in self.lane_diffusion.lpdm.parameters():
                    p.requires_grad = False
                # Allow Lane Prior Refinement to keep learning
                for p in self.lane_diffusion.lpr.parameters():
                    p.requires_grad = True
                # Unfreeze everything else (Backbone, Neck, Decoder, etc.)
                for module_name, module_param in self.named_parameters():
                    if not module_name.startswith("lane_diffusion."):
                        module_param.requires_grad = True
        else:
            self.lane_diffusion = None
            self.bev_mask_adapter = None

        if freeze_pretrain:
            self.freeze_pretrain()
    
    def freeze_pretrain(self):
        print('[SeqGrowGraph] Freezing img_backbone/img_neck/view_transformers (pretrained layers)')
        for m in self.img_backbone.parameters():
            m.requires_grad=False
        for m in self.img_neck.parameters():
            m.requires_grad=False
        for m in self.view_transformers.parameters():
            m.requires_grad=False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        # print(img[0].size())
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, img, img_metas, gt_centerlines=None, skip_diffusion=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        largest_feat_shape = img_feats[0].shape[3]
        down_level = int(np.log2(self.downsample // (self.final_dim[0] // largest_feat_shape)))
        bev_feats = self.view_transformers(img_feats[down_level], img_metas)
        
        seg_mask = None
        
        # Apply LaneDiffusion if enabled
        if self.use_lane_diffusion and self.lane_diffusion is not None and not skip_diffusion:
            stage = self.lane_diffusion.current_stage
            
            if stage == 'stage_i':
                # Train LPIM: need GT centerlines
                if gt_centerlines is not None:
                    bev_feats = self.lane_diffusion(bev_feats, gt_centerlines)
            
            elif stage == 'stage_ii':
                # Train LPDM: returns loss, not features
                # This will be handled in forward_pts_train
                pass
            
            elif stage in ['stage_iii', 'inference']:
                # Use diffusion to enhance features
                # Now returns (features, seg_mask)
                bev_feats, seg_mask = self.lane_diffusion(bev_feats)
        
        return bev_feats, seg_mask

    def forward_pts_train(self,
                          bev_feats,
                          gt_lines_sequences,
                          img_metas,
                          num_coeff,summary_subgraphs, seg_mask=None):
        """Forward function for point cloud branch."""
        device = bev_feats[0].device

        input_seqs = []

        max_len = max([len(target) for target in gt_lines_sequences])

        coeff_dim = num_coeff * 2
        

        input_seqs=[]
        for gt_lines_sequence in gt_lines_sequences:
            input_seq= [self.start]+gt_lines_sequence+[self.end]+[self.no_known]*(max_len-len(gt_lines_sequence))
            input_seq=torch.tensor(input_seq, device=device).long()
            input_seqs.append(input_seq.unsqueeze(0))

            
 
        input_seqs = torch.cat(input_seqs , dim=0)  # [8,501]
 
        # If seg_mask is available (from LaneDiffusion), concatenate it to BEV features
        if seg_mask is not None and self.bev_mask_adapter is not None:
            # Concatenate seg_mask to bev_feats
            # bev_feats: [B, C, H, W], seg_mask: [B, 1, H, W]
            decoder_input = torch.cat([bev_feats, seg_mask], dim=1)  # [B, C+1, H, W]
            # Adapt back to original channel count
            decoder_input = self.bev_mask_adapter(decoder_input)  # [B, C, H, W]
        else:
            decoder_input = bev_feats
            
        outputs = self.pts_bbox_head(decoder_input, input_seqs, img_metas)[-1, :, :-1, :]

       
        clause_length = 4 + coeff_dim
        n_control = img_metas[0]['n_control']
        
        
        # =============================================
        # 训练可视化
        
   
        # for bi in range(outputs.shape[0]):
        #     try:
        #         pred_line_seq = outputs[bi]
        #         pred_line_seq = pred_line_seq.argmax(-1)
        #         if self.end in pred_line_seq:
        #             stop_idx = (pred_line_seq == self.end).nonzero(as_tuple=True)[0][0]
        #         else:
        #             stop_idx = len(pred_line_seq)
        #         # if self.summary_split in pred_line_seq:
        #         #     start_idx=(pred_line_seq == self.summary_split).nonzero(as_tuple=True)[0][0]
        #         # else:
        #         #     start_idx=-1
        #         # pred_line_seq = pred_line_seq[start_idx+1:stop_idx]
        #         pred_line_seq = pred_line_seq[:stop_idx]
        #         
        #         pred_graph = EvalSeq2Graph(img_metas[bi]['token'],pred_line_seq.detach().cpu().numpy().tolist(),front_camera_only=self.front_camera_only,pc_range=self.pc_range,dx=self.dx,bz_pc_range=self.bz_pc_range,bz_dx=self.bz_dx)
        #         pred_graph.visualization([200, 200], os.path.join(self.vis_dir,'train'), 'n', 'n')
        #     except:
        #         import traceback
        #         traceback.print_exc()
        #     break

        # =============================================
        
        outputs = outputs.reshape(-1, self.num_center_classes)  # [602, 2003] last layer
        input_seqs=input_seqs[:,1:]
        input_seqs=input_seqs.flatten()
        gt_seqs_pad=input_seqs[input_seqs!=self.no_known]
        outputs=outputs[input_seqs!=self.no_known]

        losses = self.pts_bbox_head.loss_by_feat_seq(outputs, gt_seqs_pad)

        return losses
    
    def loss(self,
             inputs=None,
             data_samples=None,**kwargs):

        img = inputs['img']
        img_metas = [ds.metainfo for ds in data_samples]

        # Extract GT centerlines for LaneDiffusion if needed
        gt_centerlines = None
        gt_mask = None # For SegHead training
        
        if self.use_lane_diffusion and self.lane_diffusion is not None:
            stage = self.lane_diffusion.current_stage
            if stage in ['stage_i', 'stage_ii']:
                # Need to prepare GT centerlines
                gt_centerlines = self._prepare_gt_centerlines(img_metas)

        # Extract BEV features (with or without LaneDiffusion)
        bev_feats, seg_mask = self.extract_feat(img=img, img_metas=img_metas, gt_centerlines=gt_centerlines)
        
        if self.bev_scale != 1.0:
            b, c, h, w = bev_feats.shape
            bev_feats = F.interpolate(bev_feats, (int(h * self.bev_scale), int(w * self.bev_scale)))
            if seg_mask is not None:
                seg_mask = F.interpolate(seg_mask, (int(h * self.bev_scale), int(w * self.bev_scale)))
        
        losses = dict()
        
        # Handle Stage II (LPDM training) separately
        if self.use_lane_diffusion and self.lane_diffusion is not None:
            if self.lane_diffusion.current_stage == 'stage_ii':
                # Compute diffusion loss
                # Need to re-extract raw BEV (without diffusion)
                img_feats = self.extract_img_feat(img, img_metas)
                largest_feat_shape = img_feats[0].shape[3]
                down_level = int(np.log2(self.downsample // (self.final_dim[0] // largest_feat_shape)))
                raw_bev = self.view_transformers(img_feats[down_level], img_metas)
                gt_mask = self._prepare_gt_mask(img_metas, img_feats_shape=raw_bev.shape[-2:], device=raw_bev.device)
                
                # Forward Stage II with GT Mask and CFG
                diffusion_loss_dict = self.lane_diffusion.forward_stage_ii(raw_bev, gt_centerlines, gt_mask=gt_mask)
                losses.update(diffusion_loss_dict)
                
                # Still compute decoder loss for stage II
                # (or you can skip it depending on your training strategy)
        
        # Compute decoder loss
        gt_lines_sequences = [img_meta['centerline_sequence'] for img_meta in img_metas]
        summary_subgraphs=[img_meta['summary_subgraph']  if 'summary_subgraph' in  img_meta else [] for img_meta in img_metas]
    
        n_control = img_metas[0]['n_control']
        num_coeff = n_control - 2
        losses_pts = self.forward_pts_train(bev_feats,gt_lines_sequences ,
                                            img_metas, num_coeff,summary_subgraphs, seg_mask=seg_mask)
        losses.update(losses_pts)
        return losses
    
    def _extract_lane_list(self, img_meta):
        """Return list of lane polylines for a sample."""
        lane_list = []
        center_lines_meta = img_meta.get('center_lines_meta', None)
        if center_lines_meta:
            for lane in center_lines_meta:
                lane_arr = np.asarray(lane)
                if lane_arr.ndim == 2 and lane_arr.shape[0] >= 2:
                    lane_list.append(lane_arr[:, :2])
        if not lane_list:
            center_lines = img_meta.get('center_lines', None)
            if center_lines is not None:
                if isinstance(center_lines, dict):
                    lane_source = center_lines.get('centerlines', None)
                else:
                    lane_source = getattr(center_lines, 'centerlines', None)
                if lane_source is not None:
                    for lane in lane_source:
                        lane_arr = np.asarray(lane)
                        if lane_arr.ndim == 2 and lane_arr.shape[0] >= 2:
                            lane_list.append(lane_arr[:, :2])
        if not lane_list and 'centerline_coord' in img_meta:
            coords_field = img_meta['centerline_coord']
            if isinstance(coords_field, (list, tuple)):
                candidates = coords_field
            else:
                candidates = [coords_field]
            for lane in candidates:
                lane_arr = np.asarray(lane)
                if lane_arr.ndim == 2 and lane_arr.shape[0] >= 2:
                    lane_list.append(lane_arr[:, :2])
        return lane_list

    def _prepare_gt_mask(self, img_metas, img_feats_shape, device=None):
        """
        Prepare GT segmentation mask for SegHead training
        """
        B = len(img_metas)
        H, W = img_feats_shape
        if device is None:
            device = next(self.parameters()).device
        gt_masks = torch.zeros(B, 1, H, W, device=device)
        
        # Grid parameters from grid_conf (NOT bz_grid_conf!)
        # BEV features use grid_conf dimensions
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        dx, dy = self.dx[0], self.dx[1]
        
        # BEV feature dimensions should match:
        # H = (y_max - y_min) / dy = (32 - (-32)) / 0.5 = 128
        # W = (x_max - x_min) / dx = (48 - (-48)) / 0.5 = 192
        
        for i, img_meta in enumerate(img_metas):
            lanes = self._extract_lane_list(img_meta)
            if not lanes:
                continue

            canvas = np.zeros((H, W), dtype=np.uint8)

            for lane in lanes:
                lane = np.asarray(lane)
                if lane.ndim != 2 or lane.shape[0] < 2:
                    continue
                pts = np.zeros_like(lane)
                pts[:, 0] = (lane[:, 0] - x_min) / dx
                pts[:, 1] = (lane[:, 1] - y_min) / dy
                pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
                pts_int = pts.astype(np.int32)
                cv2.polylines(canvas, [pts_int], isClosed=False, color=1, thickness=2)

            gt_masks[i, 0] = torch.from_numpy(canvas).to(device=device, dtype=torch.float32)
            
        return gt_masks

    def _prepare_gt_centerlines(self, img_metas):
        """
        Extract and prepare GT centerlines from img_metas
        
        Args:
            img_metas: list of metadata dicts
        
        Returns:
            List of centerline coordinates for each sample
        """
        gt_centerlines = []
        for img_meta in img_metas:
            lanes = self._extract_lane_list(img_meta)
            gt_centerlines.append(lanes)
        return gt_centerlines
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_imgs = batch_inputs_dict['img']
        return self.simple_test(batch_input_metas, batch_input_imgs)

    def simple_test_pts(self, pts_feats, img_metas, seg_mask=None):
        """Test function of point cloud branch."""
        n_control = img_metas[0]['n_control']
        num_coeff = n_control - 2
        clause_length = 4 + num_coeff * 2
        
        # If seg_mask is available, concatenate it
        if seg_mask is not None and self.bev_mask_adapter is not None:
            decoder_input = torch.cat([pts_feats, seg_mask], dim=1)  # [B, C+1, H, W]
            decoder_input = self.bev_mask_adapter(decoder_input)  # [B, C, H, W]
        else:
            decoder_input = pts_feats

        device = pts_feats[0].device
        input_seqs = (torch.ones(pts_feats.shape[0], 1).to(device) * self.start).long()
        outs = self.pts_bbox_head(decoder_input, input_seqs, img_metas)
        output_seqs, values = outs
        line_results = []
        for bi in range(output_seqs.shape[0]):
            pred_line_seq = output_seqs[bi]
            pred_line_seq = pred_line_seq[1:]
            if self.end in pred_line_seq:
                stop_idx = (pred_line_seq == self.end).nonzero(as_tuple=True)[0][0]
            else:
                stop_idx = len(pred_line_seq)
                
            pred_line_seq = pred_line_seq[:stop_idx]
            
     
            line_results.append(dict(
                line_seqs = pred_line_seq.detach().cpu().numpy(),
         
            ))
        return line_results

    def simple_test_cyclic(self, img, img_metas, refine_timestep=5):
        """
        Cyclic Refinement Inference:
        1. Raw BEV -> Standard Diffusion -> Coarse Prediction
        2. Coarse Prediction -> Coarse Lines (Graph Parsing)
        3. Raw BEV + Coarse Lines -> Cyclic Refinement (SDEEdit) -> Refined BEV
        4. Refined BEV -> Final Prediction
        """
        import bezier
        
        # 1. Get Raw BEV (skip diffusion)
        # extract_feat returns (bev, mask). We only need bev.
        raw_bev, _ = self.extract_feat(img, img_metas, skip_diffusion=True)
        
        # 2. Initial Coarse Prediction (Standard Generation)
        # We use the LaneDiffusion model to get an initial enhanced BEV
        ld_out = self.lane_diffusion(raw_bev)
        if isinstance(ld_out, tuple):
            enhanced_bev_initial, seg_mask_initial = ld_out
        else:
            enhanced_bev_initial, seg_mask_initial = ld_out, None
             
        coarse_results = self.simple_test_pts(enhanced_bev_initial, img_metas, seg_mask=seg_mask_initial)
        
        # 3. Parse Coarse Lines
        coarse_lines_batch = []
        
        # Precompute grid parameters for coordinate conversion
        # bz_pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        x_min, y_min = self.bz_pc_range[0], self.bz_pc_range[1]
        dx, dy = self.bz_dx[0], self.bz_dx[1]
        
        for i, res in enumerate(coarse_results):
            token = img_metas[i]['token']
            seq = res['line_seqs']
            
            try:
                # Parse sequence to graph
                graph = EvalSeq2Graph(token, seq, self.pc_range, self.dx, self.bz_pc_range, self.bz_dx)
                
                lines = []
                # Traverse graph to extract lines
                for node in graph.graph_nodelist:
                    if node is None: continue
                    for child, coeff in node.childs:
                        # Control points in grid coordinates
                        p0 = node.coord
                        p1 = coeff
                        p2 = child.coord
                        
                        # Bezier curve interpolation
                        nodes = np.asfortranarray([p0, p1, p2]).T
                        curve = bezier.Curve(nodes, degree=2)
                        s_vals = np.linspace(0.0, 0.99, 20) # 20 points per line, avoid 1.0 to prevent overlap issues? or 1.0 is fine
                        points_grid = curve.evaluate_multi(s_vals).T # [20, 2]
                        
                        # Convert to meters
                        points_meters = points_grid * np.array([dx, dy]) + np.array([x_min, y_min])
                        
                        lines.append(torch.tensor(points_meters, dtype=torch.float32))
                
                if len(lines) == 0:
                    lines.append(torch.zeros(20, 2))
                
                # Stack lines: [M, 20, 2]
                lines_tensor = torch.stack(lines).to(raw_bev.device)
                coarse_lines_batch.append(lines_tensor)
                
            except Exception as e:
                # print(f"Error parsing seq for token {token}: {e}")
                # Fallback
                coarse_lines_batch.append(torch.zeros(1, 20, 2).to(raw_bev.device))

        # 4. Cyclic Refinement
        # Note: forward_cyclic returns refined_bev
        refined_bev = self.lane_diffusion.forward_cyclic(raw_bev, coarse_lines_batch, refine_timestep)
        
        # 5. Final Prediction
        # We might want to use the seg_mask from the initial pass or predict a new one?
        # forward_cyclic currently only returns refined_bev.
        # We can pass None for seg_mask or reuse the initial one if we think it's good enough.
        # Or we can modify forward_cyclic to return a mask too (if it runs the full model).
        # But forward_cyclic uses SDEEdit which might not output a mask explicitly unless we ask for it.
        # For now, let's pass None or the initial mask. 
        # Passing initial mask might be safer to avoid shape mismatch if adapter is used.
        
        final_results = self.simple_test_pts(refined_bev, img_metas, seg_mask=seg_mask_initial)
        
        return final_results

    def simple_test(self, img_metas, img=None):
        """Test function without augmentaiton."""
        gt_centerlines = None
        if self.use_lane_diffusion and self.lane_diffusion is not None:
            if self.lane_diffusion.current_stage == 'stage_i':
                gt_centerlines = self._prepare_gt_centerlines(img_metas)

        bev_feats, seg_mask = self.extract_feat(
            img=img, img_metas=img_metas, gt_centerlines=gt_centerlines)
        
        # Check for cyclic refinement
        use_cyclic = False
        if self.use_lane_diffusion and self.lane_diffusion is not None:
            if self.lane_diffusion.current_stage == 'inference':
                use_cyclic = True
        
        if use_cyclic:
            line_results = self.simple_test_cyclic(img, img_metas, refine_timestep=5)
        else:
            line_results = self.simple_test_pts(bev_feats, img_metas, seg_mask=seg_mask)
            
        bbox_list = [dict() for i in range(len(img_metas))]
        i=0
        for result_dict, line_result, img_meta in zip(bbox_list, line_results, img_metas):
            
            result_dict['line_results'] = line_result
            result_dict['token'] = img_meta['token']
            if i==0:
                try:
                    pred_graph = EvalSeq2Graph(img_meta['token'],line_result["line_seqs"],front_camera_only=self.front_camera_only,pc_range=self.pc_range,dx=self.dx,bz_pc_range=self.bz_pc_range,bz_dx=self.bz_dx)
                    pred_graph.visualization([200, 200], os.path.join(self.vis_dir,'test'), 'n', 'n')
                except:
                    import traceback
                    traceback.print_exc()
            i+=1
                

        return bbox_list
