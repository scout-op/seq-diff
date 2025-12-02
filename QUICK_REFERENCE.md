# LaneDiffusion 快速参考

## 🚀 快速开始

```bash
# 1. 安装依赖 (如果需要)
conda activate SeqGrowGraph
pip install einops  # 如果缺少

# 2. Stage I - 训练 LPIM
# 编辑 configs/seq_grow_graph/seq_grow_graph_lanediffusion.py
# 设置: model['lane_diffusion_stage'] = 'stage_i'
./tools/dist_train.sh configs/seq_grow_graph/seq_grow_graph_lanediffusion.py 8

# 3. Stage II - 训练 LPDM
# 设置: model['lane_diffusion_stage'] = 'stage_ii'
# 设置: load_from = "work_dirs/.../epoch_XX.pth"
./tools/dist_train.sh configs/seq_grow_graph/seq_grow_graph_lanediffusion.py 8

# 4. Stage III - 微调 Decoder
# 设置: model['lane_diffusion_stage'] = 'stage_iii'
# 设置: load_from = "work_dirs/.../epoch_XX.pth"
./tools/dist_train.sh configs/seq_grow_graph/seq_grow_graph_lanediffusion.py 8

# 5. 测试
# 设置: model['lane_diffusion_stage'] = 'inference'
./tools/dist_test.sh configs/seq_grow_graph/seq_grow_graph_lanediffusion.py \
  work_dirs/.../epoch_XX.pth 8
```

## 📊 核心参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_steps` | 15 | 扩散步数 (推理时) |
| `kappa` | 0.5 | 噪声方差系数 |
| `p` | 1.0 | Shifting schedule 增长率 |
| `prior_dim` | 256 | Prior 特征维度 |
| `num_encoder_layers` | 4 | LPIM Transformer 层数 |
| `max_lanes` | 50 | 最大车道数 |
| `max_points_per_lane` | 20 | 每条车道采样点数 |
| `embed_dim` | 96 | Swin U-Net 嵌入维度 |

## 🔧 常见调整

### 降低显存使用
```python
# 在配置文件中:
train_dataloader = dict(batch_size=4)  # 降低 batch size
model = dict(
    lane_diffusion_cfg=dict(
        lpdm_config=dict(
            denoiser_config=dict(
                embed_dim=48,  # 降低维度
                depths=[2, 2, 2, 2],  # 减少层数
            ),
        ),
    ),
)
```

### 加快采样速度
```python
lpdm_config=dict(
    num_steps=10,  # 减少采样步数 (15 → 10)
)
```

### 提升质量
```python
lpdm_config=dict(
    num_steps=30,  # 增加采样步数
    kappa=0.3,  # 降低噪声
)
```

## 📁 关键文件位置

```
SeqGrowGraph/
├── seq_grow_graph/
│   ├── lane_diffusion/
│   │   ├── swin_unet.py       # ← Denoiser 网络
│   │   ├── lpim.py             # ← Prior 注入
│   │   ├── lpdm.py             # ← 核心扩散模块
│   │   └── lane_diffusion.py  # ← 完整框架
│   └── seq_grow_graph.py       # ← 主模型 (已修改)
├── configs/seq_grow_graph/
│   └── seq_grow_graph_lanediffusion.py  # ← 训练配置
└── LANEDIFFUSION_IMPLEMENTATION.md      # ← 完整文档
```

## 🐛 调试技巧

```bash
# 检查模块导入
python -c "from seq_grow_graph.lane_diffusion import LaneDiffusion; print('OK')"

# 小规模测试
# 修改配置: train_dataloader = dict(batch_size=1)
# num_epochs = 1

# 可视化 BEV 特征
# 在 extract_feat() 中添加:
# import cv2
# import numpy as np
# feat_vis = bev_feats.mean(1)[0].detach().cpu().numpy()
# cv2.imwrite('bev_feat.png', (feat_vis*255).astype(np.uint8))
```

## 📈 评估指标

```bash
# 运行评估
python seq_grow_graph/nus_metric_new.py \
  --result_path work_dirs/seq_grow_graph_lanediffusion/results.pkl
```

## ⚡ 性能优化

| 技巧 | 实现方法 |
|------|----------|
| 混合精度 | `--amp` (如果支持) |
| Gradient Checkpointing | `with_cp=True` in Swin blocks |
| 分布式训练 | 使用 `dist_train.sh` 而非 `train.py` |
| 预加载数据 | `num_workers=10` in dataloader |

## 🎯 预期结果

训练完成后，你应该在 nuScenes val 上看到:
- GEO F1: ~59% (baseline ~55%)
- TOPO F1: ~47% (baseline ~42%)
- APLS: ~37% (baseline ~31%)

如果结果偏低，检查:
1. Stage I 是否收敛
2. GT centerlines 是否正确加载
3. BEV 特征维度是否匹配
