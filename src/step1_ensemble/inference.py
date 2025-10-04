"""
Step 1: Simple Ensemble 推理
功能: 验证两个模型融合的性能上限
策略: 加权平均 logits
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import load_config, load_dual_models, create_dataloader

logger = logging.getLogger(__name__)


class SimpleEnsemble:
    """简单ensemble推理器"""

    def __init__(
        self,
        foundation_model,
        sft_model,
        tokenizer,
        foundation_weight: float = 0.5,
        sft_weight: float = 0.5
    ):
        """
        Args:
            foundation_model: Foundation模型
            sft_model: SFT模型
            tokenizer: Tokenizer
            foundation_weight: Foundation模型权重
            sft_weight: SFT模型权重
        """
        self.foundation_model = foundation_model
        self.sft_model = sft_model
        self.tokenizer = tokenizer

        # 归一化权重
        total = foundation_weight + sft_weight
        self.foundation_weight = foundation_weight / total
        self.sft_weight = sft_weight / total

        logger.info(f"Ensemble权重 - Foundation: {self.foundation_weight:.2f}, SFT: {self.sft_weight:.2f}")

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> List[str]:
        """
        Ensemble生成

        Args:
            input_ids: 输入token ids
            attention_mask: Attention mask
            max_new_tokens: 最大生成长度
            temperature: 温度
            top_p: Top-p采样
            do_sample: 是否采样

        Returns:
            生成的文本列表
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # 初始化生成序列
        generated_ids = input_ids.clone()
        past_key_values_foundation = None
        past_key_values_sft = None

        for step in range(max_new_tokens):
            # 并行前向传播两个模型
            # Foundation model
            outputs_foundation = self.foundation_model(
                input_ids=generated_ids if past_key_values_foundation is None else generated_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values_foundation,
                use_cache=True
            )
            logits_foundation = outputs_foundation.logits[:, -1, :]  # [batch, vocab]
            past_key_values_foundation = outputs_foundation.past_key_values

            # SFT model
            outputs_sft = self.sft_model(
                input_ids=generated_ids if past_key_values_sft is None else generated_ids[:, -1:],
                attention_mask=attention_mask,
                past_key_values=past_key_values_sft,
                use_cache=True
            )
            logits_sft = outputs_sft.logits[:, -1, :]  # [batch, vocab]
            past_key_values_sft = outputs_sft.past_key_values

            # 融合logits (加权平均)
            logits_ensemble = self.foundation_weight * logits_foundation + \
                            self.sft_weight * logits_sft

            # 应用temperature
            logits_ensemble = logits_ensemble / temperature

            # 采样
            if do_sample:
                # Top-p sampling
                sorted_logits, sorted_indices = torch.sort(logits_ensemble, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率超过top_p的tokens
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits_ensemble[indices_to_remove] = float('-inf')

                # 采样
                probs = F.softmax(logits_ensemble, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = torch.argmax(logits_ensemble, dim=-1, keepdim=True)

            # 拼接
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # 更新attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
            ], dim=-1)

            # 检查是否所有序列都生成了eos
            if (next_token == self.tokenizer.eos_token_id).all():
                break

        # Decode
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return outputs

    @torch.no_grad()
    def generate_simple(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> List[str]:
        """
        简化版生成接口（使用HF generate方法的ensemble变种）

        Args:
            prompts: 输入文本列表
            max_new_tokens: 最大生成长度
            temperature: 温度
            top_p: Top-p采样
            do_sample: 是否采样

        Returns:
            生成的文本列表
        """
        # 这个方法使用更简单的策略：分别生成然后选择
        # 适合快速验证

        logger.info(f"生成中... (batch_size={len(prompts)})")

        # Tokenize
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.foundation_model.device)
        attention_mask = inputs["attention_mask"].to(self.foundation_model.device)

        # 分别生成
        outputs_foundation = self.foundation_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id
        )

        outputs_sft = self.sft_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id
        )

        # Decode
        texts_foundation = self.tokenizer.batch_decode(outputs_foundation, skip_special_tokens=True)
        texts_sft = self.tokenizer.batch_decode(outputs_sft, skip_special_tokens=True)

        # 简单策略: 返回两个输出 (后续可以改进为logits融合)
        results = []
        for i, prompt in enumerate(prompts):
            results.append({
                "prompt": prompt,
                "foundation_output": texts_foundation[i],
                "sft_output": texts_sft[i],
                # 简单选择：如果权重>0.5用哪个
                "ensemble_output": texts_sft[i] if self.sft_weight > 0.5 else texts_foundation[i]
            })

        return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Step 1: Simple Ensemble推理")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("--foundation_weight", type=float, default=None, help="Foundation模型权重")
    parser.add_argument("--sft_weight", type=float, default=None, help="SFT模型权重")
    parser.add_argument("--prompt", type=str, default=None, help="单个prompt测试")
    parser.add_argument("--eval_file", type=str, default=None, help="评估数据文件")
    parser.add_argument("--output_file", type=str, default="outputs/ensemble_results.jsonl", help="输出文件")

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 加载配置
    config = load_config(args.config)

    # 加载模型
    logger.info("=" * 50)
    logger.info("Step 1: Simple Ensemble 推理")
    logger.info("=" * 50)

    foundation_path = config['models']['foundation']['path']
    sft_path = config['models']['sft']['path']

    foundation_model, sft_model, tokenizer = load_dual_models(
        foundation_path,
        sft_path
    )

    # 创建ensemble
    foundation_weight = args.foundation_weight or config['ensemble']['foundation_weight']
    sft_weight = args.sft_weight or config['ensemble']['sft_weight']

    ensemble = SimpleEnsemble(
        foundation_model,
        sft_model,
        tokenizer,
        foundation_weight=foundation_weight,
        sft_weight=sft_weight
    )

    # 测试
    if args.prompt:
        # 单个prompt测试
        logger.info(f"\n测试Prompt: {args.prompt}")
        results = ensemble.generate_simple([args.prompt])

        print("\n" + "=" * 50)
        print("生成结果:")
        print("=" * 50)
        print(f"\nFoundation输出:\n{results[0]['foundation_output']}")
        print(f"\nSFT输出:\n{results[0]['sft_output']}")
        print(f"\nEnsemble输出:\n{results[0]['ensemble_output']}")

    elif args.eval_file:
        # 批量评估
        from src.utils import load_jsonl, save_jsonl

        eval_data = load_jsonl(args.eval_file)
        prompts = [item['input'] for item in eval_data]

        logger.info(f"评估数据: {len(prompts)}条")

        all_results = []
        batch_size = config['inference']['batch_size']

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = ensemble.generate_simple(batch_prompts)
            all_results.extend(batch_results)

            logger.info(f"进度: {i + len(batch_prompts)}/{len(prompts)}")

        # 保存结果
        import os
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        save_jsonl(all_results, args.output_file)
        logger.info(f"结果已保存到: {args.output_file}")

    else:
        # 默认测试
        test_prompts = [
            "请按照SOP流程处理客户退款申请",
            "解释一下什么是机器学习？"
        ]

        logger.info("\n运行默认测试...")
        results = ensemble.generate_simple(test_prompts)

        for i, result in enumerate(results):
            print(f"\n{'=' * 50}")
            print(f"测试 {i + 1}")
            print(f"{'=' * 50}")
            print(f"Prompt: {result['prompt']}")
            print(f"\nFoundation: {result['foundation_output']}")
            print(f"\nSFT: {result['sft_output']}")
            print(f"\nEnsemble: {result['ensemble_output']}")

    logger.info("\n完成!")


if __name__ == "__main__":
    main()
