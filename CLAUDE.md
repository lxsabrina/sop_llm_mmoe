# SOP LLM MMoE 项目说明

## 项目概述

本项目实现了基于混合专家模型(MoE)的SOP LLM融合方案，通过智能路由策略融合Foundation模型和SFT模型。

## 核心架构

### 1. Simple Ensemble (Step 1)
基础融合策略，用于快速验证性能上限。

### 2. BTM - Branch-Train-Merge (Step 2)
- Token-level routing: 为每个token决定使用哪个expert
- Layer-level routing: 为每一层决定融合策略
- 可训练的router网络

### 3. Hybrid Router (新增)
智能路由策略，结合单一agent和BTM的优势：
- 高置信度(≥0.85): 使用单一agent (1x推理成本)
- 低置信度(<0.85): 使用BTM融合 (2x推理成本)
- 平均推理成本: ~1.2x

**关键文件**：
- `src/step2_btm/query_classifier.py`: Query分类器
- `src/step2_btm/hybrid_router.py`: 混合路由器
- `src/step2_btm/train_classifier.py`: 分类器训练
- `src/step2_btm/hybrid_inference.py`: 混合推理脚本

### 4. Knowledge Distillation (Step 3)
将BTM能力蒸馏到单一模型，实现1x推理成本。

## 使用场景

### 快速原型验证
```bash
# 使用rule-based分类器，无需训练
python src/step2_btm/hybrid_inference.py \
  --config configs/config.yaml \
  --classifier_type rule_based \
  --prompt "你的query"
```

### 训练learned分类器
```bash
# 创建示例训练数据
python src/step2_btm/train_classifier.py --create_sample

# 训练分类器
python src/step2_btm/train_classifier.py \
  --train_file data/classifier/train.jsonl \
  --epochs 5
```

### 生产部署
```bash
# 启用BTM fallback的混合路由
python src/step2_btm/hybrid_inference.py \
  --classifier_type learned \
  --classifier_checkpoint models/checkpoints/classifier/best_classifier.pt \
  --enable_btm \
  --btm_router models/checkpoints/btm/best_router.pt \
  --eval_file data/eval/eval.jsonl \
  --show_routing_info
```

## 配置说明

### Hybrid配置 (`configs/config.yaml`)
```yaml
hybrid:
  classifier:
    type: "rule_based"  # 或 "learned"
    encoder_model: "Qwen/Qwen2.5-0.5B"
    checkpoint_dir: "./models/checkpoints/classifier"

  routing:
    confidence_threshold: 0.85
    enable_btm_fallback: true
```

## 性能对比

| 方案 | 推理成本 | 性能 | 训练成本 |
|------|---------|------|---------|
| Foundation Only | 1x | 基准 | 0 |
| SFT Only | 1x | SOP优 | 0 |
| Ensemble | 2x | 最优 | 0 |
| BTM | 2x | 近最优 | 低 |
| Hybrid Router | ~1.2x | 近最优 | 极低 |
| Distilled | 1x | 优 | 高 |

## 技术细节

### Query分类器
- **Rule-based**: 基于关键词匹配，零训练成本
- **Learned**: 基于轻量级encoder，需少量标注数据

### 路由策略
1. Query → 分类器 → (类别, 置信度)
2. 置信度 ≥ threshold → 单一agent
3. 置信度 < threshold → BTM融合

### 可扩展性
- 支持多个SFT专家（客服、销售、技术等）
- 分类器可扩展为多类别分类
- Router可配置不同策略

## 开发指南

### 添加新的SFT专家
1. 在配置文件中添加模型路径
2. 扩展分类器支持新类别
3. 更新HybridRouter的路由逻辑

### 自定义Router策略
编辑`src/step2_btm/hybrid_router.py`中的`decide_routing`方法。

### 训练数据格式
```json
{"query": "查询文本", "label": 0}  // 0=reasoning, 1=sop
```

## 常见问题

### Q: Rule-based分类器准确率不够？
A: 训练learned分类器，准备至少200-500条标注数据。

### Q: BTM fallback显存不足？
A: 设置`enable_btm_fallback: false`，低置信度query使用foundation。

### Q: 如何调整成本/性能平衡？
A: 调整`confidence_threshold`，降低阈值提高性能但增加成本。

## 后续优化方向

1. 多专家扩展：支持3+个SFT专家
2. 动态阈值：根据query复杂度自适应调整
3. 混合策略：部分token用expert A，部分用expert B
4. 在线学习：根据用户反馈持续优化分类器
