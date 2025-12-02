import os
from typing import List, Optional, Tuple

import bezier
import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules
from projects.SeqGrowGraph.seq_grow_graph.core.centerline.structures import (
    EvalSeq2Graph_with_start as EvalSeq2Graph,
)


class ScriptArgs:
    """Hard-coded parameters for quick inspection."""

    # TODO: 修改为你当前使用的配置与权重
    # config = 'projects/SeqGrowGraph/configs/seq_grow_graph/seq_grow_graph_lanediffusion.py'
    config = 'projects/SeqGrowGraph/configs/seq_grow_graph/seq_grow_graph_default.py'
    # checkpoint = '/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/lane2/work_dirs/seq_grow_graph_lanediffusion_s2_v6/epoch_75.pth'
    checkpoint = '/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/lane2/work_dirs/seq_grow_graph_lanediffusion_s1_v4/epoch_24.pth'
    out_dir = 'vis_stage2_all_9'
    num_samples = 10
    start_index = 0
    random_seed = 0


def get_args():
    return ScriptArgs()


def feature_to_rgb(feature: torch.Tensor) -> np.ndarray:
    """Convert BEV feature map (C, H, W) to pseudo RGB image via PCA."""
    feature = feature.detach().float()
    c, h, w = feature.shape
    flattened = feature.reshape(c, -1).permute(1, 0)
    flattened = flattened - flattened.mean(dim=0, keepdim=True)
    if flattened.numel() == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    q = min(3, flattened.shape[1])
    if q == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)

    try:
        _, _, v = torch.pca_lowrank(flattened, q=q, center=False)
        proj = flattened @ v[:, :q]
    except RuntimeError:
        proj = flattened[:, :q]

    if proj.shape[1] < 3:
        pad = torch.zeros((proj.shape[0], 3 - proj.shape[1]),
                          device=proj.device,
                          dtype=proj.dtype)
        proj = torch.cat([proj, pad], dim=1)

    proj_min, proj_max = proj.min(), proj.max()
    if (proj_max - proj_min) < 1e-6:
        proj = torch.zeros_like(proj)
    else:
        proj = (proj - proj_min) / (proj_max - proj_min)
    proj = proj.view(h, w, 3)
    rgb = (proj.cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def mask_to_bgr(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.squeeze().detach().cpu().numpy()
    mask_np = np.clip(mask_np, 0, 1)
    mask_u8 = (mask_np * 255).astype(np.uint8)
    return cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)


def format_img_tensor(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 5:
        return img
    if img.dim() == 4:
        return img.unsqueeze(0)
    if img.dim() == 3:
        return img.unsqueeze(0).unsqueeze(0)
    raise ValueError(f'Unsupported image shape: {tuple(img.shape)}')


def prepare_raw_and_refined(model,
                            img: torch.Tensor,
                            img_metas: list,
                            device: torch.device):
    img = format_img_tensor(img).to(device)
    with torch.no_grad():
        raw_bev, _ = model.extract_feat(
            img=img, img_metas=img_metas, skip_diffusion=True)
        refined_feat, pred_mask = model.lane_diffusion.lpdm.sample(raw_bev)
    return raw_bev, refined_feat, pred_mask
# def prepare_raw_and_refined(model,
#                             img: torch.Tensor,
#                             img_metas: list,
#                             device: torch.device):
#     img = format_img_tensor(img).to(device)
#     with torch.no_grad():
#         # 1. 正常提取原始特征
#         raw_bev, _ = model.extract_feat(
#             img=img, img_metas=img_metas, skip_diffusion=True)
        
#         # =======================================================
#         # ⚠️【在这里插入代码】进行“失明测试”
#         # =======================================================
#         print("⚠️ WARNING: BLINDNESS TEST - Forcing Raw BEV to ZEROS!")

def build_graph(seq: Optional[np.ndarray], token: str, model) -> Optional[EvalSeq2Graph]:
    if seq is None:
        return None
    if isinstance(seq, torch.Tensor):
        seq = seq.detach().cpu().numpy()
    seq_list = np.asarray(seq).astype(np.int64).tolist()
    if len(seq_list) == 0:
        return None
    return EvalSeq2Graph(
        token,
        seq_list,
        model.pc_range,
        model.dx,
        model.bz_pc_range,
        model.bz_dx,
    )


def draw_topology(graph: Optional[EvalSeq2Graph],
                  canvas_size: Tuple[int, int] = (200, 200),
                  scale: int = 5,
                  node_color: Tuple[int, int, int] = (0, 255, 0),
                  edge_color: Tuple[int, int, int] = (0, 170, 255)) -> np.ndarray:
    width = int(canvas_size[0] * scale)
    height = int(canvas_size[1] * scale)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if graph is None:
        return canvas

    for node in graph.graph_nodelist:
        if node is None:
            continue
        coord = np.clip(np.round(node.coord * scale).astype(int),
                        [0, 0], [width - 1, height - 1])
        cv2.circle(canvas, tuple(coord), max(2, int(scale * 0.8)),
                   color=node_color, thickness=-1)
        for child, coeff in node.childs:
            if child is None:
                continue
            child_coord = np.clip(np.round(child.coord * scale).astype(int),
                                  [0, 0], [width - 1, height - 1])
            control = np.clip(np.round(coeff * scale).astype(int),
                              [0, 0], [width - 1, height - 1])
            fin_res = np.stack((coord, control, child_coord))
            curve = bezier.Curve(fin_res.T, degree=2)
            pts = curve.evaluate_multi(np.linspace(0.0, 1.0, 50)).T.astype(int)
            cv2.polylines(canvas, [pts], False, edge_color, thickness=2)
    return canvas


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def pad_to_same_size(images: List[np.ndarray]) -> List[np.ndarray]:
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        if img.shape[0] == max_h and img.shape[1] == max_w:
            padded.append(img)
        else:
            padded.append(cv2.resize(img, (max_w, max_h)))
    return padded


def main():
    args = get_args()
    register_all_modules()

    cfg = Config.fromfile(args.config)
    dataset = DATASETS.build(cfg.val_dataloader['dataset'])

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if hasattr(model, 'lane_diffusion') and model.lane_diffusion is not None:
        model.lane_diffusion.set_stage('stage_iii')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    num_total = len(dataset)
    indices = list(range(num_total))
    if args.random_seed is not None:
        rng = np.random.default_rng(args.random_seed)
        rng.shuffle(indices)
    selected = indices[:args.num_samples]

    for data_idx in selected:
        data = dataset[data_idx]
        inputs = data['inputs']
        data_sample = data['data_samples']
        img = inputs['img']
        img_metas = [data_sample.metainfo]

        raw_bev, refined_bev, pred_mask = prepare_raw_and_refined(
            model, img, img_metas, device)
        gt_mask = model._prepare_gt_mask(
            img_metas,
            img_feats_shape=raw_bev.shape[-2:],
            device=device)

        line_results = model.simple_test_pts(
            refined_bev, img_metas, seg_mask=pred_mask)
        pred_seq = line_results[0]['line_seqs']
        gt_seq = data_sample.metainfo.get('centerline_sequence')
        token = data_sample.metainfo.get('token', f'sample_{data_idx:04d}')
        graph_pred = build_graph(pred_seq, token, model)
        graph_gt = build_graph(gt_seq, token, model)

        vis_raw = feature_to_rgb(raw_bev[0])
        vis_refined = feature_to_rgb(refined_bev[0])
        pred_prob = torch.sigmoid(pred_mask[0])
        vis_pred_mask = cv2.applyColorMap(
            (pred_prob.squeeze().detach().cpu().numpy() * 255).astype(np.uint8),
            cv2.COLORMAP_JET)
        vis_gt_mask = mask_to_bgr(gt_mask[0])
        vis_pred_topo = draw_topology(graph_pred,
                                      node_color=(0, 102, 255),
                                      edge_color=(255, 193, 7))
        vis_gt_topo = draw_topology(graph_gt,
                                    node_color=(0, 255, 0),
                                    edge_color=(0, 170, 0))

        panels = [
            add_label(vis_raw, 'Raw Feature'),
            add_label(vis_refined, 'Refined Feature (PCA)'),
            add_label(vis_pred_mask, 'Pred Mask'),
            add_label(vis_gt_mask, 'GT Mask'),
            add_label(vis_pred_topo, 'Pred Topology'),
            add_label(vis_gt_topo, 'GT Topology'),
        ]
        panels = pad_to_same_size(panels)

        rows, cols = 2, 3
        tile_h, tile_w = panels[0].shape[:2]
        canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for idx_panel, panel in enumerate(panels):
            r = idx_panel // cols
            c = idx_panel % cols
            canvas[r * tile_h:(r + 1) * tile_h,
                   c * tile_w:(c + 1) * tile_w] = panel

        save_path = os.path.join(args.out_dir, f'sample_{data_idx:04d}.png')
        cv2.imwrite(save_path, canvas)
        print(f'Saved visualization to {save_path}')


if __name__ == '__main__':
    main()
