import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
    trust_remote_code: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载模型和tokenizer

    Args:
        model_path: 模型路径
        device: 设备
        torch_dtype: 数据类型
        load_in_8bit: 是否使用8bit量化
        trust_remote_code: 是否信任远程代码(Qwen需要)

    Returns:
        model, tokenizer
    """
    logger.info(f"正在加载模型: {model_path}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side='left'
    )

    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=trust_remote_code,
        load_in_8bit=load_in_8bit
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()
    logger.info(f"模型加载完成，设备: {device}")

    return model, tokenizer


def load_dual_models(
    foundation_path: str,
    sft_path: str,
    device_map: Optional[dict] = None
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """
    加载foundation和sft两个模型

    Args:
        foundation_path: Foundation模型路径
        sft_path: SFT模型路径
        device_map: 设备映射，例如 {"foundation": "cuda:0", "sft": "cuda:1"}

    Returns:
        foundation_model, sft_model, tokenizer
    """
    if device_map is None:
        # 默认: 如果有多GPU，分别加载到不同GPU
        if torch.cuda.device_count() >= 2:
            device_map = {"foundation": "cuda:0", "sft": "cuda:1"}
        else:
            device_map = {"foundation": "cuda", "sft": "cuda"}

    logger.info(f"设备映射: {device_map}")

    # 加载foundation model
    foundation_model, tokenizer = load_model_and_tokenizer(
        foundation_path,
        device=device_map.get("foundation", "cuda")
    )

    # 加载sft model (使用相同tokenizer)
    sft_model, _ = load_model_and_tokenizer(
        sft_path,
        device=device_map.get("sft", "cuda")
    )

    logger.info("双模型加载完成")

    return foundation_model, sft_model, tokenizer


def estimate_memory_usage(model_path: str) -> dict:
    """
    估算模型显存占用

    Args:
        model_path: 模型路径

    Returns:
        内存使用信息
    """
    # 简单估算: Qwen3-8B约16GB (bf16)
    config_path = f"{model_path}/config.json"
    try:
        import json
        with open(config_path) as f:
            config = json.load(f)

        num_params = config.get('num_parameters', 8e9)  # 默认8B
        # bf16: 2 bytes per param
        memory_gb = (num_params * 2) / 1e9

        return {
            "num_parameters": num_params,
            "memory_bf16_gb": memory_gb,
            "memory_fp32_gb": memory_gb * 2,
            "memory_int8_gb": memory_gb / 2
        }
    except Exception as e:
        logger.warning(f"无法估算内存: {e}")
        return {"memory_bf16_gb": 16.0}  # 默认估算


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    print("测试模型内存估算:")
    mem_info = estimate_memory_usage("./models/foundation")
    print(mem_info)
