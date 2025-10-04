# 贡献指南

感谢你对SOP LLM MMoE项目的关注！

## 如何贡献

### 报告Bug

如果你发现了bug，请创建issue并包含：
- 详细的错误描述
- 复现步骤
- 环境信息（Python版本、GPU型号、CUDA版本等）
- 错误日志

### 提交代码

1. Fork本仓库
2. 创建feature分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -m 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建Pull Request

### 代码规范

- 遵循PEP 8 Python代码规范
- 添加必要的文档字符串
- 更新相关文档
- 添加测试（如果适用）

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/lxsabrina/sop_llm_mmoe.git
cd sop_llm_mmoe

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装开发依赖
pip install black flake8 pytest
```

### 测试

运行测试前请确保：
- 已放置模型到`models/`目录
- 已准备测试数据到`data/`目录

```bash
# 运行代码风格检查
black src/ --check
flake8 src/

# 运行单元测试（如果有）
pytest tests/
```

### 提交信息规范

请使用清晰的提交信息：

```
<type>: <subject>

<body>

<footer>
```

类型（type）：
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `test`: 测试相关
- `chore`: 构建/工具相关

示例：
```
feat: Add support for custom router architecture

- Allow users to specify custom router network
- Add configuration options in config.yaml
- Update documentation

Closes #123
```

## 开发路线图

当前优先级：

### 高优先级
- [ ] 添加更多评估指标（BLEU, ROUGE等）
- [ ] 优化显存使用
- [ ] 添加推理加速选项

### 中优先级
- [ ] 支持更多模型架构（Llama, ChatGLM等）
- [ ] Web UI界面
- [ ] Docker部署支持

### 低优先级
- [ ] 自动超参数调优
- [ ] 分布式训练优化
- [ ] 模型量化支持

## 问题和讨论

如有任何问题，欢迎：
- 创建Issue讨论
- 在Discussions发起讨论
- 通过邮件联系维护者

## 行为准则

请保持友好和专业的态度。我们致力于创建一个包容和欢迎的环境。

## 许可证

通过提交代码，你同意你的贡献将在与本项目相同的许可证下发布。
