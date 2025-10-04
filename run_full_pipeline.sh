#!/bin/bash

# 完整流程示例
# 演示Step 1 -> Step 2 -> Step 3的完整流程

set -e

echo "=========================================="
echo "完整流程示例 (Step 1 -> 2 -> 3)"
echo "=========================================="
echo ""
echo "注意: 请确保已经放置了foundation和sft模型"
read -p "是否继续? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Step 1: Ensemble测试
echo ""
echo "=========================================="
echo "Step 1: Ensemble 快速测试"
echo "=========================================="

python src/step1_ensemble/inference.py \
    --config configs/config.yaml \
    --prompt "请按照SOP流程处理客户退款申请"

echo ""
read -p "Step 1完成，是否继续Step 2? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Step 2: BTM训练
echo ""
echo "=========================================="
echo "Step 2: BTM Router训练"
echo "=========================================="

python src/step2_btm/train.py \
    --config configs/config.yaml \
    --output_dir models/checkpoints/btm

# BTM推理测试
echo ""
echo "测试BTM推理..."
python src/step2_btm/inference.py \
    --config configs/config.yaml \
    --router_checkpoint models/checkpoints/btm/best_router.pt \
    --prompt "请按照SOP流程处理客户退款申请" \
    --show_router_stats

echo ""
read -p "Step 2完成，是否继续Step 3? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Step 3: 蒸馏训练
echo ""
echo "=========================================="
echo "Step 3: 蒸馏训练"
echo "=========================================="

python src/step3_distill/train.py \
    --config configs/config.yaml \
    --teacher_type btm \
    --btm_router models/checkpoints/btm/best_router.pt \
    --student_init foundation \
    --output_dir models/checkpoints/distill

echo ""
echo "=========================================="
echo "完整流程完成!"
echo "=========================================="
echo ""
echo "最终评估对比:"
python evaluate.py \
    --config configs/config.yaml \
    --models foundation sft btm distill \
    --btm_router models/checkpoints/btm/best_router.pt \
    --distill_model models/checkpoints/distill/best_student \
    --output_file outputs/final_evaluation.json

echo ""
echo "所有结果已保存到 outputs/ 目录"
echo "查看详细结果: cat outputs/final_evaluation.json"
