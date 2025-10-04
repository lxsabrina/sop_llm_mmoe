"""
混合路由器 - 结合单一agent和BTM的智能路由
策略:
- 高置信度 → 单一agent (1x成本)
- 低置信度 → BTM融合 (2x成本)
"""
import torch
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

from .query_classifier import QueryClassifier, SimpleRuleBasedClassifier
from .model import BTMModel

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """路由策略"""
    FOUNDATION_ONLY = "foundation"
    SFT_ONLY = "sft"
    BTM_FUSION = "btm"


@dataclass
class RoutingDecision:
    """路由决策"""
    strategy: RoutingStrategy
    query_type: str
    confidence: float
    reasoning: str


@dataclass
class RoutingStats:
    """路由统计"""
    total_queries: int = 0
    foundation_only: int = 0
    sft_only: int = 0
    btm_fusion: int = 0

    @property
    def avg_cost_multiplier(self) -> float:
        """平均推理成本倍数"""
        if self.total_queries == 0:
            return 1.0
        total_cost = (
            self.foundation_only * 1.0 +
            self.sft_only * 1.0 +
            self.btm_fusion * 2.0
        )
        return total_cost / self.total_queries

    def __str__(self):
        if self.total_queries == 0:
            return "暂无统计数据"
        return f"""
路由统计:
  总查询数: {self.total_queries}
  Foundation Only: {self.foundation_only} ({self.foundation_only/self.total_queries*100:.1f}%)
  SFT Only: {self.sft_only} ({self.sft_only/self.total_queries*100:.1f}%)
  BTM Fusion: {self.btm_fusion} ({self.btm_fusion/self.total_queries*100:.1f}%)
  平均推理成本: {self.avg_cost_multiplier:.2f}x
"""


class HybridRouter:
    """混合路由器"""

    def __init__(
        self,
        foundation_model,
        sft_model,
        tokenizer,
        btm_model: Optional[BTMModel] = None,
        classifier: Optional[QueryClassifier] = None,
        classifier_type: str = "rule_based",
        confidence_threshold: float = 0.85,
        enable_btm_fallback: bool = True
    ):
        self.foundation_model = foundation_model
        self.sft_model = sft_model
        self.tokenizer = tokenizer
        self.btm_model = btm_model
        self.confidence_threshold = confidence_threshold
        self.enable_btm_fallback = enable_btm_fallback and btm_model is not None

        # 分类器
        if classifier is not None:
            self.classifier = classifier
            self.classifier_type = "learned"
        elif classifier_type == "rule_based":
            self.classifier = SimpleRuleBasedClassifier()
            self.classifier_type = "rule_based"
        else:
            raise ValueError(f"需要提供classifier或使用rule_based")

        self.stats = RoutingStats()

        logger.info(
            f"HybridRouter初始化: "
            f"classifier={self.classifier_type}, "
            f"threshold={confidence_threshold}, "
            f"btm_fallback={self.enable_btm_fallback}"
        )

    def classify_query(self, queries: List[str]) -> Dict[str, torch.Tensor]:
        """分类query"""
        if self.classifier_type == "rule_based":
            return self.classifier.predict(queries)
        else:
            inputs = self.tokenizer(
                queries,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.foundation_model.device) for k, v in inputs.items()}
            return self.classifier.predict(inputs["input_ids"], inputs["attention_mask"])

    def decide_routing(self, query: str, predicted_class: int, confidence: float) -> RoutingDecision:
        """决定路由策略"""
        query_type = "reasoning" if predicted_class == 0 else "sop"

        if confidence >= self.confidence_threshold:
            if predicted_class == 0:
                return RoutingDecision(
                    strategy=RoutingStrategy.FOUNDATION_ONLY,
                    query_type=query_type,
                    confidence=confidence,
                    reasoning=f"高置信度({confidence:.3f})推理任务，使用Foundation"
                )
            else:
                return RoutingDecision(
                    strategy=RoutingStrategy.SFT_ONLY,
                    query_type=query_type,
                    confidence=confidence,
                    reasoning=f"高置信度({confidence:.3f})SOP任务，使用SFT"
                )
        else:
            if self.enable_btm_fallback:
                return RoutingDecision(
                    strategy=RoutingStrategy.BTM_FUSION,
                    query_type=query_type,
                    confidence=confidence,
                    reasoning=f"低置信度({confidence:.3f})，使用BTM融合"
                )
            else:
                return RoutingDecision(
                    strategy=RoutingStrategy.FOUNDATION_ONLY,
                    query_type=query_type,
                    confidence=confidence,
                    reasoning=f"低置信度({confidence:.3f})且无BTM，默认使用Foundation"
                )

    @torch.no_grad()
    def generate(
        self,
        queries: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        return_routing_info: bool = False
    ) -> List[Dict]:
        """混合路由生成"""
        classification_results = self.classify_query(queries)
        results = []

        for i, query in enumerate(queries):
            predicted_class = classification_results["predicted_class"][i].item()
            confidence = classification_results["confidence"][i].item()

            decision = self.decide_routing(query, predicted_class, confidence)

            # 更新统计
            self.stats.total_queries += 1
            if decision.strategy == RoutingStrategy.FOUNDATION_ONLY:
                self.stats.foundation_only += 1
            elif decision.strategy == RoutingStrategy.SFT_ONLY:
                self.stats.sft_only += 1
            elif decision.strategy == RoutingStrategy.BTM_FUSION:
                self.stats.btm_fusion += 1

            output = self._generate_single(
                query, decision.strategy, max_new_tokens,
                temperature, top_p, do_sample
            )

            result = {"query": query, "output": output}
            if return_routing_info:
                result["routing_decision"] = {
                    "strategy": decision.strategy.value,
                    "query_type": decision.query_type,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning
                }
            results.append(result)

        return results

    def _generate_single(
        self, query: str, strategy: RoutingStrategy,
        max_new_tokens: int, temperature: float,
        top_p: float, do_sample: bool
    ) -> str:
        """单个query的生成"""
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.foundation_model.device)
        attention_mask = inputs["attention_mask"].to(self.foundation_model.device)

        if strategy == RoutingStrategy.FOUNDATION_ONLY:
            output_ids = self.foundation_model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        elif strategy == RoutingStrategy.SFT_ONLY:
            output_ids = self.sft_model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )
        elif strategy == RoutingStrategy.BTM_FUSION:
            output_ids = self.btm_model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, do_sample=do_sample
            )
        else:
            raise ValueError(f"未知策略: {strategy}")

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def get_stats(self) -> RoutingStats:
        return self.stats

    def reset_stats(self):
        self.stats = RoutingStats()
        logger.info("路由统计已重置")

    def print_stats(self):
        print("\n" + "=" * 50)
        print(self.stats)
        print("=" * 50)
