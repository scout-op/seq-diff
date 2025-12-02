import os
from typing import List, Tuple

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules


class ScriptArgs:
    # TODO: 修改为你的 Stage I 配置与权重
    config = 'projects/SeqGrowGraph/configs/seq_grow_graph/seq_grow_graph_lanediffusion.py'
    checkpoint = '/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/lane2/work_dirs/seq_grow_graph_lanediffusion_s1_v7/epoch_2.pth'
    out_dir = 'vis_stage1_lpim_4_2'
    num_samples = 10
    random_seed = 0
    visualize_topology = False


def get_args():
    return ScriptArgs()


def format_img(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 5:
        return img
    if img.dim() == 4:
        return img.unsqueeze(0)
    if img.dim() == 3:
        return img.unsqueeze(0).unsqueeze(0)
    raise ValueError(f'Unexpected img shape: {tuple(img.shape)}')


def feature_to_rgb(feat: torch.Tensor) -> np.ndarray:
    feat = feat.detach().float()
    c, h, w = feat.shape
    flat = feat.reshape(c, -1).permute(1, 0)
    flat = flat - flat.mean(dim=0, keepdim=True)
    if flat.numel() == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    q = min(3, flat.shape[1])
    _, _, v = torch.pca_lowrank(flat, q=q, center=False)
    proj = flat @ v[:, :q]
    if proj.shape[1] < 3:
        pad = torch.zeros((proj.shape[0], 3 - proj.shape[1]),
                          device=proj.device,
                          dtype=proj.dtype)
        proj = torch.cat([proj, pad], dim=1)
    proj = proj.view(h, w, 3)
    proj_min, proj_max = proj.min(), proj.max()
    if (proj_max - proj_min) < 1e-6:
        proj = torch.zeros_like(proj)
    else:
        proj = (proj - proj_min) / (proj_max - proj_min)
    return (proj.cpu().numpy() * 255).astype(np.uint8)


def prepare_lpim_output(model, img, img_metas, device):
    img = format_img(img).to(device)
    with torch.no_grad():
        raw_bev, _ = model.extract_feat(
            img=img, img_metas=img_metas, skip_diffusion=True)
        gt_centerlines = model._prepare_gt_centerlines(img_metas)
        lpim_feat = model.lane_diffusion.lpim(raw_bev, gt_centerlines)
    return raw_bev, lpim_feat


def add_text(img: np.ndarray, text: str) -> np.ndarray:
    canvas = img.copy()
    cv2.putText(canvas, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def pad_to_same(images: List[np.ndarray]) -> List[np.ndarray]:
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        if img.shape[:2] == (max_h, max_w):
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
    if model.lane_diffusion is not None:
        model.lane_diffusion.set_stage('stage_i')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(args.random_seed)
    rng.shuffle(indices)
    selected = indices[:args.num_samples]

    for idx in selected:
        data = dataset[idx]
        inputs = data['inputs']
        data_sample = data['data_samples']
        img = inputs['img']
        img_metas = [data_sample.metainfo]

        raw_bev, lpim_bev = prepare_lpim_output(model, img, img_metas, device)

        raw_vis = feature_to_rgb(raw_bev[0])
        lpim_vis = feature_to_rgb(lpim_bev[0])

        enlarged = []
        for vis in [raw_vis, lpim_vis]:
            vis_large = cv2.resize(vis, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            enlarged.append(vis_large)

        panels = pad_to_same([
            add_text(enlarged[0], 'Raw BEV (LSS)'),
            add_text(enlarged[1], 'LPIM Injected BEV')
        ])

        tile_h, tile_w = panels[0].shape[:2]
        canvas = np.zeros((tile_h, tile_w * 2, 3), dtype=np.uint8)
        canvas[:, :tile_w] = panels[0]
        canvas[:, tile_w:] = panels[1]

        save_path = os.path.join(args.out_dir, f'sample_{idx:04d}.png')
        cv2.imwrite(save_path, canvas)
        print(f'Saved {save_path}')


if __name__ == '__main__':
    main()
