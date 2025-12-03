import os
import numpy as np
import cv2
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
import bezier
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules
from projects.SeqGrowGraph.seq_grow_graph.core.centerline.structures import (
    EvalSeq2Graph_with_start as EvalSeq2Graph,
)


class ScriptArgs:
    config = 'projects/SeqGrowGraph/configs/seq_grow_graph/seq_grow_graph_lanediffusion.py'
    checkpoint = '/mnt/tf-mdriver-jfs/exps/lixiangjie/roadnet/data_copy/lane2/work_dirs/seq_grow_graph_lanediffusion_s1_v7/epoch_100.pth'
    out_dir = 'vis_stage1_lpim_topology——3'
    num_samples = 20
    random_seed = 0


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


def draw_centerlines(feat_vis: np.ndarray,
                     graph: EvalSeq2Graph,
                     color=(255, 255, 255)) -> np.ndarray:
    canvas = feat_vis.copy()
    h, w = canvas.shape[:2]
    for node in graph.graph_nodelist:
        if node is None:
            continue
        x = int(np.clip(np.round(node.coord[0]), 0, w - 1))
        y = int(np.clip(np.round(node.coord[1]), 0, h - 1))
        cv2.circle(canvas, (x, y), 2, color, -1)
        for child, coeff in node.childs:
            if child is None:
                continue
            child_x = int(np.clip(np.round(child.coord[0]), 0, w - 1))
            child_y = int(np.clip(np.round(child.coord[1]), 0, h - 1))
            ctrl_x = float(np.clip(coeff[0], 0, w - 1))
            ctrl_y = float(np.clip(coeff[1], 0, h - 1))
            nodes = np.asfortranarray([[x, ctrl_x, child_x],
                                       [y, ctrl_y, child_y]])
            curve = bezier.Curve(nodes, degree=2)
            poly = curve.evaluate_multi(np.linspace(0.0, 1.0, 50)).T
            for i in range(len(poly) - 1):
                p0 = tuple(np.clip(np.round(poly[i]), [0, 0], [w - 1, h - 1]).astype(int))
                p1 = tuple(np.clip(np.round(poly[i + 1]), [0, 0], [w - 1, h - 1]).astype(int))
                cv2.line(canvas, p0, p1, color, 1)
    return canvas


def build_graph(seq, token, model):
    if seq is None:
        return None
    if torch.is_tensor(seq):
        seq = seq.cpu().numpy()
    seq_list = np.asarray(seq).astype(np.int64).tolist()
    if not seq_list:
        return None
    return EvalSeq2Graph(
        token,
        seq_list,
        model.pc_range,
        model.dx,
        model.bz_pc_range,
        model.bz_dx,
    )


def prepare_lpim_output(model, img, img_metas, device):
    img = format_img(img).to(device)
    with torch.no_grad():
        raw_bev, _ = model.extract_feat(
            img=img, img_metas=img_metas, skip_diffusion=True)
        gt_centerlines = model._prepare_gt_centerlines(img_metas)
        lpim_bev = model.lane_diffusion.lpim(raw_bev, gt_centerlines)
    return raw_bev, lpim_bev, gt_centerlines


def pad_to_same(images):
    max_h = max(img.shape[0] for img in images)
    max_w = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        if img.shape[:2] == (max_h, max_w):
            padded.append(img)
        else:
            padded.append(cv2.resize(img, (max_w, max_h)))
    return padded


def add_text(img, text):
    canvas = img.copy()
    cv2.putText(canvas, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


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
    rng = np.random.default_rng(args.random_seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    selected = indices[:args.num_samples]

    for idx in selected:
        data = dataset[idx]
        inputs = data['inputs']
        data_sample = data['data_samples']
        img = inputs['img']
        img_metas = [data_sample.metainfo]

        raw_bev, lpim_bev, _ = prepare_lpim_output(model, img, img_metas, device)
        seq_gt = data_sample.metainfo.get('centerline_sequence')
        token = data_sample.metainfo.get('token', f'sample_{idx:04d}')
        graph_gt = build_graph(seq_gt, token, model)

        raw_vis = feature_to_rgb(raw_bev[0])
        lpim_vis = feature_to_rgb(lpim_bev[0])

        if graph_gt is not None:
            raw_vis = draw_centerlines(raw_vis, graph_gt, color=(0, 255, 0))
            lpim_vis = draw_centerlines(lpim_vis, graph_gt, color=(0, 255, 0))

        diff_map = torch.abs(lpim_bev[0] - raw_bev[0])
        diff_vis = feature_to_rgb(diff_map)

        raw_vis = add_text(cv2.resize(raw_vis, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC),
                           'Raw BEV + GT')
        lpim_vis = add_text(cv2.resize(lpim_vis, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC),
                            'LPIM BEV + GT')
        diff_vis = add_text(cv2.resize(diff_vis, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC),
                             '|LPIM - Raw|')

        panels = pad_to_same([raw_vis, lpim_vis, diff_vis])
        tile_h, tile_w = panels[0].shape[:2]
        canvas = np.zeros((tile_h, tile_w * 3, 3), dtype=np.uint8)
        for i, panel in enumerate(panels):
            canvas[:, i * tile_w:(i + 1) * tile_w] = panel

        save_path = os.path.join(args.out_dir, f'sample_{idx:04d}.png')
        cv2.imwrite(save_path, canvas)
        print(f'Saved {save_path}')


if __name__ == '__main__':
    main()
