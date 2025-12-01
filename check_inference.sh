#!/bin/bash
# 推理前检查脚本

echo "========================================="
echo "LaneDiffusion Inference Checklist"
echo "========================================="
echo ""

# 1. 检查配置文件
echo "1. 检查配置文件中的 stage 设置..."
grep "lane_diffusion_stage.*=.*'inference'" configs/seq_grow_graph/seq_grow_graph_lanediffusion.py
if [ $? -eq 0 ]; then
    echo "   ✓ Stage 设置为 inference"
else
    echo "   ✗ 请在配置文件中设置: model['lane_diffusion_stage'] = 'inference'"
fi
echo ""

# 2. 检查权重文件
echo "2. 检查 Stage III 训练的权重文件..."
if [ -d "work_dirs/seq_grow_graph_lanediffusion_s3" ]; then
    echo "   ✓ 找到 Stage III 工作目录"
    echo "   可用的 checkpoint:"
    ls -lh work_dirs/seq_grow_graph_lanediffusion_s3/*.pth 2>/dev/null | tail -5
else
    echo "   ✗ 未找到 Stage III 工作目录"
fi
echo ""

# 3. 提示推理命令
echo "3. 推理命令示例:"
echo ""
echo "   方式一：使用最新的 checkpoint"
echo "   ./tools/dist_test.sh \\"
echo "       configs/seq_grow_graph/seq_grow_graph_lanediffusion.py \\"
echo "       work_dirs/seq_grow_graph_lanediffusion_s3/latest.pth \\"
echo "       8"
echo ""
echo "   方式二：使用指定 epoch 的 checkpoint"
echo "   ./tools/dist_test.sh \\"
echo "       configs/seq_grow_graph/seq_grow_graph_lanediffusion.py \\"
echo "       work_dirs/seq_grow_graph_lanediffusion_s3/epoch_XX.pth \\"
echo "       8"
echo ""
echo "========================================="
