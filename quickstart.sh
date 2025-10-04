#!/bin/bash

# 快速启动脚本
# 用于初始化项目和运行示例

set -e

echo "=========================================="
echo "SOP LLM MMoE 快速启动"
echo "=========================================="

# 1. 检查Python环境
echo ""
echo "步骤 1/5: 检查Python环境..."
python --version || {
    echo "错误: 未找到Python"
    exit 1
}

# 2. 安装依赖
echo ""
echo "步骤 2/5: 安装依赖..."
read -p "是否安装依赖包? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements.txt
fi

# 3. 创建示例数据
echo ""
echo "步骤 3/5: 创建示例数据..."
python -c "from src.utils.data_utils import create_sample_data; create_sample_data()"
echo "示例数据已创建在 data/ 目录"

# 4. 验证配置
echo ""
echo "步骤 4/5: 验证配置..."
python src/utils/config_loader.py

# 5. 检查模型
echo ""
echo "步骤 5/5: 检查模型..."
if [ ! -d "models/foundation" ]; then
    echo "警告: Foundation模型不存在"
    echo "请将Qwen3-8B Foundation模型放到 models/foundation/ 目录"
fi

if [ ! -d "models/sft" ]; then
    echo "警告: SFT模型不存在"
    echo "请将SFT模型放到 models/sft/ 目录"
fi

echo ""
echo "=========================================="
echo "初始化完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 放置模型到 models/foundation/ 和 models/sft/"
echo "2. 准备训练数据到 data/train/train.jsonl"
echo "3. 运行 Step 1: python src/step1_ensemble/inference.py --prompt '你的测试问题'"
echo ""
echo "详细使用说明请查看 README.md"
echo ""
