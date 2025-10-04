"""
统一评估脚本
评估各个阶段的模型性能
"""
import torch
import sys
from pathlib import Path
import logging
from typing import List, Dict
import json
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, load_jsonl, save_jsonl

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""

    def __init__(self, model, tokenizer, device="cuda"):
        """
        Args:
            model: 待评估模型
            tokenizer: Tokenizer
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.model.eval()

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """批量生成"""
        results = []

        for prompt in prompts:
            # 构建input
            messages = [{"role": "user", "content": prompt}]
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(input_text, return_tensors="pt")
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            # 生成
            try:
                # 检查模型是否有generate方法
                if hasattr(self.model, 'generate'):
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                else:
                    # BTM模型
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )

                # Decode
                output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append(output_text)

            except Exception as e:
                logger.error(f"生成失败: {e}")
                results.append(f"[生成错误: {str(e)}]")

        return results

    def evaluate_perplexity(self, test_data: List[Dict]) -> float:
        """计算困惑度"""
        total_loss = 0
        total_tokens = 0

        for item in test_data:
            # 构建input + output
            messages = [
                {"role": "user", "content": item['input']},
                {"role": "assistant", "content": item['output']}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False
            )

            inputs = self.tokenizer(text, return_tensors="pt")
            input_ids = inputs['input_ids'].to(self.device)

            # 计算loss
            try:
                outputs = self.model(input_ids=input_ids, labels=input_ids)

                if isinstance(outputs, dict):
                    loss = outputs['loss']
                else:
                    loss = outputs.loss

                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)

            except Exception as e:
                logger.warning(f"计算perplexity失败: {e}")
                continue

        if total_tokens == 0:
            return float('inf')

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity

    def evaluate_accuracy(self, test_data: List[Dict], generated_outputs: List[str]) -> Dict:
        """
        评估准确率
        这里简化为检查关键词匹配
        实际应用中可以使用更复杂的指标
        """
        correct = 0
        total = len(test_data)

        for i, item in enumerate(test_data):
            expected = item['output'].lower()
            generated = generated_outputs[i].lower()

            # 简单匹配: 检查是否包含关键内容
            # 可以改进为BLEU/ROUGE等指标
            if expected[:50] in generated or generated[:50] in expected:
                correct += 1

        accuracy = correct / total if total > 0 else 0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }


def evaluate_model(
    model_name: str,
    model,
    tokenizer,
    test_categories: List[Dict],
    config: Dict
) -> Dict:
    """
    评估单个模型

    Args:
        model_name: 模型名称
        model: 模型
        tokenizer: Tokenizer
        test_categories: 测试类别列表
        config: 配置

    Returns:
        评估结果
    """
    logger.info(f"\n{'=' * 50}")
    logger.info(f"评估模型: {model_name}")
    logger.info(f"{'=' * 50}")

    evaluator = ModelEvaluator(model, tokenizer)
    results = {
        "model_name": model_name,
        "categories": {}
    }

    for category in test_categories:
        cat_name = category['name']
        cat_file = category['file']

        logger.info(f"\n测试类别: {cat_name}")

        # 加载测试数据
        try:
            test_data = load_jsonl(cat_file)
        except FileNotFoundError:
            logger.warning(f"测试文件不存在: {cat_file}")
            continue

        if len(test_data) == 0:
            logger.warning(f"测试数据为空: {cat_file}")
            continue

        prompts = [item['input'] for item in test_data]

        # 生成
        logger.info(f"生成中... ({len(prompts)}条)")
        generated_outputs = evaluator.generate_batch(prompts)

        # 计算指标
        # 1. 困惑度
        logger.info("计算困惑度...")
        perplexity = evaluator.evaluate_perplexity(test_data)

        # 2. 准确率
        logger.info("计算准确率...")
        accuracy_metrics = evaluator.evaluate_accuracy(test_data, generated_outputs)

        # 保存结果
        cat_results = {
            "perplexity": perplexity,
            "accuracy": accuracy_metrics['accuracy'],
            "num_samples": len(test_data),
            "samples": []
        }

        # 保存前3个样本
        for i in range(min(3, len(test_data))):
            cat_results['samples'].append({
                "input": test_data[i]['input'],
                "expected": test_data[i]['output'],
                "generated": generated_outputs[i]
            })

        results['categories'][cat_name] = cat_results

        logger.info(f"  Perplexity: {perplexity:.2f}")
        logger.info(f"  Accuracy: {accuracy_metrics['accuracy']:.2%}")

    return results


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="统一评估脚本")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件")
    parser.add_argument("--models", type=str, nargs='+', required=True,
                       help="要评估的模型 (foundation, sft, ensemble, btm, distill)")
    parser.add_argument("--btm_router", type=str, default=None, help="BTM router路径")
    parser.add_argument("--distill_model", type=str, default=None, help="Distill模型路径")
    parser.add_argument("--output_file", type=str, default="outputs/evaluation_results.json", help="输出文件")

    args = parser.parse_args()

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 加载配置
    config = load_config(args.config)

    # 测试类别
    test_categories = config['evaluation']['test_categories']

    # 加载模型并评估
    all_results = {}

    for model_name in args.models:
        logger.info(f"\n准备评估: {model_name}")

        if model_name == "foundation":
            from src.utils import load_model_and_tokenizer
            model, tokenizer = load_model_and_tokenizer(
                config['models']['foundation']['path']
            )

        elif model_name == "sft":
            from src.utils import load_model_and_tokenizer
            model, tokenizer = load_model_and_tokenizer(
                config['models']['sft']['path']
            )

        elif model_name == "ensemble":
            from src.step1_ensemble.inference import SimpleEnsemble
            from src.utils import load_dual_models

            foundation_model, sft_model, tokenizer = load_dual_models(
                config['models']['foundation']['path'],
                config['models']['sft']['path']
            )

            # Ensemble作为"模型"
            # 注意: 这里需要适配SimpleEnsemble的接口
            logger.warning("Ensemble模型评估需要特殊处理，暂时跳过")
            continue

        elif model_name == "btm":
            if not args.btm_router:
                logger.error("BTM评估需要指定--btm_router参数")
                continue

            from src.utils import load_dual_models
            from src.step2_btm.model import BTMModel

            foundation_model, sft_model, tokenizer = load_dual_models(
                config['models']['foundation']['path'],
                config['models']['sft']['path']
            )

            hidden_dim = foundation_model.config.hidden_size
            num_layers = foundation_model.config.num_hidden_layers

            model = BTMModel(
                foundation_model=foundation_model,
                sft_model=sft_model,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                router_hidden_dim=config['btm']['router_hidden_dim'],
                router_type=config['btm']['router_type']
            )

            model.load_router(args.btm_router)

        elif model_name == "distill":
            if not args.distill_model:
                logger.error("Distill评估需要指定--distill_model参数")
                continue

            from src.utils import load_model_and_tokenizer
            model, tokenizer = load_model_and_tokenizer(args.distill_model)

        else:
            logger.error(f"未知模型类型: {model_name}")
            continue

        # 评估
        results = evaluate_model(
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            test_categories=test_categories,
            config=config
        )

        all_results[model_name] = results

        # 清理显存
        del model
        torch.cuda.empty_cache()

    # 保存结果
    import os
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n评估结果已保存: {args.output_file}")

    # 打印对比表格
    print("\n" + "=" * 80)
    print("评估结果对比")
    print("=" * 80)

    # 提取指标
    for cat in test_categories:
        cat_name = cat['name']
        print(f"\n{cat_name}:")
        print(f"{'模型':<15} {'Perplexity':<15} {'Accuracy':<15}")
        print("-" * 50)

        for model_name, results in all_results.items():
            if cat_name in results['categories']:
                cat_result = results['categories'][cat_name]
                ppl = cat_result['perplexity']
                acc = cat_result['accuracy']
                print(f"{model_name:<15} {ppl:<15.2f} {acc:<15.2%}")

    logger.info("\n完成!")


if __name__ == "__main__":
    main()
