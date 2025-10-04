"""
Step 2: BTM 推理脚本
使用训练好的router进行推理
"""
import torch
import sys
from pathlib import Path
import logging
from typing import List

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import load_config, load_dual_models, load_jsonl, save_jsonl
from src.step2_btm.model import BTMModel

logger = logging.getLogger(__name__)


class BTMInference:
    """BTM推理器"""

    def __init__(
        self,
        btm_model: BTMModel,
        tokenizer,
        device: str = "cuda"
    ):
        """
        Args:
            btm_model: BTM模型
            tokenizer: Tokenizer
            device: 设备
        """
        self.model = btm_model
        self.tokenizer = tokenizer
        self.device = device

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        return_router_stats: bool = False
    ) -> List[dict]:
        """
        批量生成

        Args:
            prompts: 输入prompts
            max_new_tokens: 最大生成长度
            temperature: 温度
            top_p: Top-p采样
            do_sample: 是否采样
            return_router_stats: 是否返回router统计信息

        Returns:
            结果列表
        """
        results = []

        for prompt in prompts:
            # 构建messages
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = self.tokenizer(input_text, return_tensors="pt")
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            # 生成
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )

            # Decode
            output_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )

            result = {
                "prompt": prompt,
                "output": output_text
            }

            # Router统计
            if return_router_stats:
                # 运行一次forward获取router信息
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_router_logits=True
                )

                gate_weights = outputs['gate_weights']  # [batch, seq_len, num_experts]

                # 统计
                foundation_ratio = gate_weights[:, :, 0].mean().item()
                sft_ratio = gate_weights[:, :, 1].mean().item()

                result['router_stats'] = {
                    'foundation_ratio': foundation_ratio,
                    'sft_ratio': sft_ratio
                }

            results.append(result)

        return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Step 2: BTM推理")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件")
    parser.add_argument("--router_checkpoint", type=str, required=True, help="Router checkpoint路径")
    parser.add_argument("--prompt", type=str, default=None, help="单个prompt测试")
    parser.add_argument("--eval_file", type=str, default=None, help="评估文件")
    parser.add_argument("--output_file", type=str, default="outputs/btm_results.jsonl", help="输出文件")
    parser.add_argument("--show_router_stats", action="store_true", help="显示router统计")

    args = parser.parse_args()

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 加载配置
    config = load_config(args.config)

    # 加载模型
    logger.info("加载模型...")
    foundation_path = config['models']['foundation']['path']
    sft_path = config['models']['sft']['path']

    foundation_model, sft_model, tokenizer = load_dual_models(
        foundation_path,
        sft_path
    )

    # 创建BTM模型
    hidden_dim = foundation_model.config.hidden_size
    num_layers = foundation_model.config.num_hidden_layers

    btm_model = BTMModel(
        foundation_model=foundation_model,
        sft_model=sft_model,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        router_hidden_dim=config['btm']['router_hidden_dim'],
        router_type=config['btm']['router_type']
    )

    # 加载router checkpoint
    logger.info(f"加载router: {args.router_checkpoint}")
    btm_model.load_router(args.router_checkpoint)

    # 创建推理器
    inference = BTMInference(
        btm_model=btm_model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # 推理
    if args.prompt:
        # 单个prompt
        logger.info(f"测试Prompt: {args.prompt}")
        results = inference.generate(
            prompts=[args.prompt],
            return_router_stats=args.show_router_stats
        )

        print("\n" + "=" * 50)
        print("BTM生成结果:")
        print("=" * 50)
        print(f"\n{results[0]['output']}")

        if args.show_router_stats:
            stats = results[0]['router_stats']
            print(f"\nRouter统计:")
            print(f"  Foundation: {stats['foundation_ratio']:.2%}")
            print(f"  SFT: {stats['sft_ratio']:.2%}")

    elif args.eval_file:
        # 批量评估
        eval_data = load_jsonl(args.eval_file)
        prompts = [item['input'] for item in eval_data]

        logger.info(f"评估数据: {len(prompts)}条")

        results = inference.generate(
            prompts=prompts,
            return_router_stats=args.show_router_stats
        )

        # 保存
        import os
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        save_jsonl(results, args.output_file)

        logger.info(f"结果已保存: {args.output_file}")

        # 统计
        if args.show_router_stats:
            avg_foundation = sum(r['router_stats']['foundation_ratio'] for r in results) / len(results)
            avg_sft = sum(r['router_stats']['sft_ratio'] for r in results) / len(results)

            print(f"\n平均Router使用率:")
            print(f"  Foundation: {avg_foundation:.2%}")
            print(f"  SFT: {avg_sft:.2%}")

    else:
        # 默认测试
        test_prompts = [
            "请按照SOP流程处理客户退款申请",
            "解释一下量子力学的基本原理"
        ]

        results = inference.generate(
            prompts=test_prompts,
            return_router_stats=True
        )

        for i, result in enumerate(results):
            print(f"\n{'=' * 50}")
            print(f"测试 {i + 1}")
            print(f"{'=' * 50}")
            print(f"Prompt: {result['prompt']}")
            print(f"\nOutput: {result['output']}")

            if 'router_stats' in result:
                stats = result['router_stats']
                print(f"\nRouter: Foundation {stats['foundation_ratio']:.2%} | SFT {stats['sft_ratio']:.2%}")

    logger.info("完成!")


if __name__ == "__main__":
    main()
