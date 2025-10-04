# SOP LLM MMoE - 混合专家模型融合方案

基于Qwen3-8B的Foundation模型和SFT模型融合项目，通过三个阶段逐步实现模型融合和优化。

## 项目背景

本项目解决SFT模型过拟合、泛化能力不足的问题，通过融合Foundation模型的推理能力和SFT模型的SOP遵循能力，实现最佳性能。

## 项目结构

```
sop_llm_mmoe/
├── configs/
│   ├── config.yaml                    # 主配置文件
│   └── llamafactory_distill.yaml      # LlamaFactory蒸馏配置
├── models/
│   ├── foundation/                    # 【请放置】Qwen3-8B Foundation模型
│   ├── sft/                          # 【请放置】Qwen3-8B SFT模型
│   └── checkpoints/
│       ├── btm/                      # BTM router checkpoints
│       └── distill/                  # 蒸馏后的student模型
├── data/
│   ├── train/
│   │   └── train.jsonl               # 训练数据
│   └── eval/
│       ├── eval.jsonl                # 通用评估数据
│       ├── sop_strict.jsonl          # SOP严格遵循测试
│       ├── reasoning.jsonl           # 推理能力测试
│       └── mixed.jsonl               # 混合场景测试
├── src/
│   ├── utils/                        # 通用工具
│   ├── step1_ensemble/               # Step 1: 简单Ensemble
│   ├── step2_btm/                    # Step 2: BTM训练
│   └── step3_distill/                # Step 3: 蒸馏训练
├── outputs/                          # 输出结果
├── logs/                             # 训练日志
├── evaluate.py                       # 统一评估脚本
└── README.md

```

## 环境准备

### 1. 安装依赖

```bash
pip install torch transformers accelerate peft tqdm pyyaml
pip install tensorboard  # 可选，用于可视化

# 如果使用LlamaFactory
pip install llamafactory-cli
```

### 2. 准备模型

将你的模型放到对应目录：

```bash
# Foundation模型
models/foundation/
  ├── config.json
  ├── model.safetensors (或pytorch_model.bin)
  ├── tokenizer.json
  └── ...

# SFT模型
models/sft/
  ├── config.json
  ├── model.safetensors
  ├── tokenizer.json
  └── ...
```

### 3. 准备数据

数据格式为JSONL，每行一个JSON对象：

```json
{"input": "用户输入", "output": "期望输出"}
```

示例：

```bash
# 创建示例数据
python -c "from src.utils.data_utils import create_sample_data; create_sample_data()"
```

## 使用流程

### Step 1: Simple Ensemble (1天)

**目标**: 快速验证两个模型融合的性能上限

```bash
# 测试Ensemble
python src/step1_ensemble/inference.py \
  --config configs/config.yaml \
  --prompt "请按照SOP流程处理客户退款申请"

# 批量评估
python src/step1_ensemble/inference.py \
  --config configs/config.yaml \
  --eval_file data/eval/eval.jsonl \
  --output_file outputs/ensemble_results.jsonl
```

**参数说明**:
- `--foundation_weight`: Foundation模型权重 (默认0.5)
- `--sft_weight`: SFT模型权重 (默认0.5)
- `--prompt`: 单个prompt测试
- `--eval_file`: 批量评估文件

**预期结果**:
- 验证ensemble上限性能
- 分析两个模型各自的优势场景

---

### Step 2: BTM训练 (3-5天)

**目标**: 训练router，智能融合两个模型

#### 2.1 训练Router

```bash
# 训练BTM router
python src/step2_btm/train.py \
  --config configs/config.yaml \
  --output_dir models/checkpoints/btm
```

**配置参数** (在`configs/config.yaml`中修改):
```yaml
btm:
  num_epochs: 3
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  router_type: "layer_level"  # token_level或layer_level
  load_balance_weight: 0.01
  sparsity_weight: 0.001
```

**训练监控**:
```bash
# 查看日志
tail -f logs/btm_training.log

# TensorBoard (可选)
tensorboard --logdir outputs/tensorboard
```

#### 2.2 BTM推理

```bash
# 使用训练好的router推理
python src/step2_btm/inference.py \
  --config configs/config.yaml \
  --router_checkpoint models/checkpoints/btm/best_router.pt \
  --prompt "解释量子纠缠现象" \
  --show_router_stats

# 批量评估
python src/step2_btm/inference.py \
  --config configs/config.yaml \
  --router_checkpoint models/checkpoints/btm/best_router.pt \
  --eval_file data/eval/eval.jsonl \
  --output_file outputs/btm_results.jsonl \
  --show_router_stats
```

**Router统计**:
显示每个query使用Foundation vs SFT的比例，帮助理解routing策略。

---

### Step 3: 蒸馏训练 (1-2周)

**目标**: 将BTM能力蒸馏到单一模型，降低推理成本

#### 3.1 使用自定义trainer蒸馏

```bash
# 蒸馏训练
python src/step3_distill/train.py \
  --config configs/config.yaml \
  --teacher_type btm \
  --btm_router models/checkpoints/btm/best_router.pt \
  --student_init foundation \
  --output_dir models/checkpoints/distill
```

**参数说明**:
- `--teacher_type`: Teacher类型 (btm或ensemble)
- `--btm_router`: BTM router checkpoint路径
- `--student_init`: Student初始化 (foundation/sft/自定义路径)

**配置参数**:
```yaml
distill:
  temperature: 2.0                # 蒸馏温度
  distill_loss_weight: 0.7       # 蒸馏loss权重
  task_loss_weight: 0.3          # 任务loss权重
  learning_rate: 5e-6
  num_epochs: 2
  student_init: "foundation"
```

#### 3.2 使用LlamaFactory蒸馏 (可选)

如果你熟悉LlamaFactory，可以用它做baseline SFT：

```bash
# 1. 准备数据集配置 (data/dataset_info.json)
{
  "distill_train": {
    "file_name": "train/train.jsonl",
    "columns": {
      "prompt": "input",
      "response": "output"
    }
  }
}

# 2. 运行LlamaFactory训练
llamafactory-cli train configs/llamafactory_distill.yaml
```

---

## 评估对比

统一评估所有模型：

```bash
python evaluate.py \
  --config configs/config.yaml \
  --models foundation sft btm distill \
  --btm_router models/checkpoints/btm/best_router.pt \
  --distill_model models/checkpoints/distill/best_student \
  --output_file outputs/evaluation_results.json
```

**评估指标**:
- Perplexity: 困惑度
- Accuracy: 准确率
- 分类评估: SOP遵循 vs 推理能力 vs 混合场景

**输出示例**:
```
评估结果对比
================================================================================

sop_strict:
模型             Perplexity      Accuracy
--------------------------------------------------
foundation      25.30           65.00%
sft             18.50           95.00%
btm             19.20           92.00%
distill         20.10           90.00%

reasoning:
模型             Perplexity      Accuracy
--------------------------------------------------
foundation      22.10           88.00%
sft             28.40           70.00%
btm             23.50           86.00%
distill         24.20           85.00%
```

---

## 配置说明

### 主配置文件: `configs/config.yaml`

关键配置项：

```yaml
# 模型路径
models:
  foundation:
    path: "./models/foundation"
  sft:
    path: "./models/sft"

# 数据
data:
  train_file: "./data/train/train.jsonl"
  eval_file: "./data/eval/eval.jsonl"

# Ensemble权重
ensemble:
  foundation_weight: 0.5
  sft_weight: 0.5

# BTM
btm:
  router_type: "layer_level"
  learning_rate: 1e-4
  num_epochs: 3

# Distillation
distill:
  temperature: 2.0
  distill_loss_weight: 0.7
  task_loss_weight: 0.3
  learning_rate: 5e-6
  num_epochs: 2
```

---

## 显存需求

| 阶段 | 训练显存 | 推理显存 | 备注 |
|------|---------|---------|------|
| Step 1 Ensemble | 0 | 2×16GB = 32GB | 并行加载两个模型 |
| Step 2 BTM训练 | 2×16GB + 2GB | 2×16GB | Router很小 |
| Step 2 BTM推理 | 0 | 2×16GB | 并行推理 |
| Step 3 蒸馏训练 | 2×16GB + 16GB | 16GB | 2个teacher + 1个student |
| Step 3 蒸馏推理 | 0 | 16GB | 单模型推理 |

**优化建议**:
- 使用`bf16`或`fp16`减少显存
- 使用`gradient_checkpointing`
- 使用`load_in_8bit=True`量化
- 多GPU分布式训练

---

## 常见问题

### Q1: 模型路径不存在

```bash
# 检查配置
python src/utils/config_loader.py
```

### Q2: 显存不足

方案：
1. 减小batch_size
2. 增加gradient_accumulation_steps
3. 使用8bit量化
4. 使用更小的模型测试流程

### Q3: BTM训练loss不下降

检查：
1. 数据质量是否足够好
2. learning_rate是否合适 (尝试1e-5到1e-3)
3. load_balance_weight是否过大 (导致router不学习)

### Q4: 蒸馏后性能下降明显

建议：
1. 增加训练数据量
2. 调整temperature (1.5-3.0)
3. 调整loss权重比例
4. 使用更好的student初始化

---

## 进阶使用

### 1. 自定义Router架构

编辑`src/step2_btm/model.py`，修改`TokenRouter`类：

```python
class TokenRouter(nn.Module):
    def __init__(self, ...):
        # 自定义网络结构
        self.router_net = nn.Sequential(
            # 你的架构
        )
```

### 2. 添加新的评估指标

编辑`evaluate.py`，在`ModelEvaluator`类中添加新方法：

```python
def evaluate_custom_metric(self, ...):
    # 实现你的指标
    pass
```

### 3. Layer-wise不同策略

修改BTM配置，为不同层设置不同的routing策略（需要修改代码）。

---

## 性能对比参考

基于Qwen3-8B的预期性能（仅供参考）：

| 模型 | SOP准确率 | 推理能力 | 推理成本 | 训练成本 |
|------|---------|---------|---------|---------|
| Foundation | 65% | 88% | 1x | 0 |
| SFT | 95% | 70% | 1x | 0 |
| Ensemble | 96% | 90% | 2x | 0 |
| BTM | 92% | 86% | 2x | 低 |
| Distill | 90% | 85% | 1x | 高 |

**推荐路径**:
1. 快速验证 → Ensemble
2. 生产部署（成本不敏感）→ BTM
3. 生产部署（成本敏感）→ Distill

---

## 下一步

1. **数据准备**: 准备高质量的真实case数据（至少1K条）
2. **Step 1验证**: 运行Ensemble，确认融合效果
3. **Step 2训练**: 训练BTM router（2-3天）
4. **Step 3蒸馏**: 如果BTM效果好，做蒸馏（1-2周）
5. **生产部署**: 根据成本选择BTM或Distill部署

---

## 引用

如果使用了本项目，请引用相关论文：

- Branch-Train-Merge: Li et al., 2022
- Model Distillation: Hinton et al., 2015
- Task Arithmetic: Ilharco et al., 2023

---

## 联系

如有问题，请查看`logs/`目录下的日志，或提issue。

**Good Luck!** 🚀
