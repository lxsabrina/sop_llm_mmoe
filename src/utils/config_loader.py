import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置文件有效性"""
    required_keys = ['models', 'data', 'inference']

    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置文件缺少必需字段: {key}")

    # 验证模型路径
    foundation_path = Path(config['models']['foundation']['path'])
    sft_path = Path(config['models']['sft']['path'])

    if not foundation_path.exists():
        print(f"警告: Foundation模型路径不存在: {foundation_path}")
        print("请将Qwen3-8B foundation model放到models/foundation目录")

    if not sft_path.exists():
        print(f"警告: SFT模型路径不存在: {sft_path}")
        print("请将SFT model放到models/sft目录")

    return True


if __name__ == "__main__":
    config = load_config()
    validate_config(config)
    print("配置文件加载成功")
    print(yaml.dump(config, allow_unicode=True, default_flow_style=False))
