# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project documentation and contribution guidelines
- MIT License
- Changelog

## [0.1.0] - 2024-10-04

### Added
- Initial implementation of hybrid MoE architecture
- Step 1: Simple Ensemble inference
  - Weighted logits fusion
  - Configurable expert weights
  - Batch processing support
- Step 2: BTM (Branch-Train-Merge)
  - Token-level and layer-level routing
  - Trainable router with load balancing
  - Router checkpoint save/load
  - Inference with router statistics
- Step 3: Knowledge Distillation
  - Teacher-student training framework
  - KL divergence distillation loss
  - Configurable temperature and loss weights
  - LlamaFactory integration support
- Unified evaluation framework
  - Perplexity and accuracy metrics
  - Category-based evaluation
  - Model comparison tools
- Utility modules
  - Config loader with YAML support
  - Model loader for dual model setup
  - Data utilities for JSONL processing
- Configuration files
  - Main config.yaml
  - LlamaFactory distillation config
- Documentation
  - Comprehensive README
  - Quick start script
  - Full pipeline script
- Dependencies specification (requirements.txt)

### Technical Details
- Based on Qwen3-8B models
- PyTorch + Transformers implementation
- Support for bf16/fp16 training
- Gradient checkpointing support
- Multi-GPU compatible

[Unreleased]: https://github.com/lxsabrina/sop_llm_mmoe/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lxsabrina/sop_llm_mmoe/releases/tag/v0.1.0
