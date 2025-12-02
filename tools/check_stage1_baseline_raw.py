import os
import numpy as np
import cv2
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules


class ScriptArgs:
    config = '/data/roadnet_data/lane2/mmdetection3d/projects/SeqGrowGraph/configs/road_seg/lss_roadseg_48x32_b4x8_resnet_adam_24e_default.py'
    checkpoint = '/data/roadnet_data/lane2/mmdetection3d/ckpts/lss_roadseg_48x32_b4x8_resnet_adam_24e_default.pth'
    out_dir = 'vis_stage1_baseline_raw_4'
    num_samples = 10
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


def get_raw_bev(model, img_tensor, img_metas):
    try:
        out = model.extract_feat(img=img_tensor, img_metas=img_metas, skip_diffusion=True)
    except TypeError:
        try:
            out = model.extract_feat(img=img_tensor, img_metas=img_metas)
        except TypeError:
            out = model.extract_feat(img_tensor, img_metas)

    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)):
        for item in out:
            if torch.is_tensor(item) and item.dim() == 4:
                return item
    raise RuntimeError('Unable to locate BEV tensor from extract_feat output.')


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


def main():
    args = get_args()
    register_all_modules()

    cfg = Config.fromfile(args.config)
    dataset = DATASETS.build(cfg.val_dataloader['dataset'])

    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
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

        img_tensor = format_img(img).to(device)
        with torch.no_grad():
            raw_bev = get_raw_bev(model, img_tensor, img_metas)

        raw_vis = feature_to_rgb(raw_bev[0])
        raw_vis = cv2.resize(raw_vis, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        cv2.putText(raw_vis, 'Raw BEV Feature (Baseline Stage I)', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        save_path = os.path.join(args.out_dir, f'sample_{idx:04d}.png')
        cv2.imwrite(save_path, raw_vis)
        print(f'Saved {save_path}')


if __name__ == '__main__':
    main()
