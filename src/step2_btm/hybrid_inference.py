"""
混合路由推理脚本
"""
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import load_config, load_dual_models, load_jsonl, save_jsonl
from src.step2_btm.hybrid_router import HybridRouter
from src.step2_btm.query_classifier import QueryClassifier

logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="混合路由推理")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--classifier_type", default="rule_based", choices=["rule_based", "learned"])
    parser.add_argument("--confidence_threshold", type=float, default=0.85)
    parser.add_argument("--enable_btm", action="store_true")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--show_routing_info", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    
    config = load_config(args.config)
    foundation_model, sft_model, tokenizer = load_dual_models(
        config['models']['foundation']['path'],
        config['models']['sft']['path']
    )

    hybrid_router = HybridRouter(
        foundation_model=foundation_model,
        sft_model=sft_model,
        tokenizer=tokenizer,
        classifier_type=args.classifier_type,
        confidence_threshold=args.confidence_threshold,
        enable_btm_fallback=args.enable_btm
    )

    if args.prompt:
        results = hybrid_router.generate(
            queries=[args.prompt],
            return_routing_info=True
        )
        print("\n结果:")
        print(results[0]['output'])
        if 'routing_decision' in results[0]:
            print(f"\n路由: {results[0]['routing_decision']}")
        hybrid_router.print_stats()


if __name__ == "__main__":
    main()
