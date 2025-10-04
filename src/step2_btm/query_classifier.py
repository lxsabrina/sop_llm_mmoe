"""
Query分类器 - 判断query类型和置信度
用于混合路由策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class QueryClassifier(nn.Module):
    """
    轻量级Query分类器
    功能: 判断query属于reasoning还是sop，并给出置信度
    """

    def __init__(
        self,
        encoder_model_name: str = "Qwen/Qwen2.5-0.5B",
        num_classes: int = 2,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            encoder_model_name: 编码器模型名称（使用轻量级模型）
            num_classes: 类别数（reasoning=0, sop=1）
            hidden_dim: 分类头hidden dimension
            dropout: Dropout比例
        """
        super().__init__()

        # 加载轻量级encoder
        logger.info(f"加载分类器encoder: {encoder_model_name}")
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        self.encoder_dim = self.encoder.config.hidden_size

        # 冻结部分层以减少训练成本
        # 只训练最后2层
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 解冻最后2层
        for layer in self.encoder.encoder.layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # 分类头
        self.classifier_head = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.num_classes = num_classes
        logger.info(f"QueryClassifier初始化完成: {num_classes}类分类")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            logits: [batch, num_classes] - 分类logits
            probs: [batch, num_classes] - 分类概率
        """
        # Encode
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 使用[CLS] token的表示 (或者平均池化)
        # Qwen没有[CLS]，使用最后一个token或平均池化
        if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
            pooled = encoder_outputs.pooler_output
        else:
            # 平均池化
            last_hidden = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden]
            # 考虑attention_mask的平均
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_hidden = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask

        # 分类
        logits = self.classifier_head(pooled)  # [batch, num_classes]
        probs = F.softmax(logits, dim=-1)

        return logits, probs

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, any]:
        """
        推理接口

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            Dict包含:
                - predicted_class: [batch] - 预测类别 (0=reasoning, 1=sop)
                - confidence: [batch] - 置信度 (0-1)
                - probs: [batch, num_classes] - 各类别概率
        """
        self.eval()
        with torch.no_grad():
            logits, probs = self.forward(input_ids, attention_mask)

            # 预测类别
            predicted_class = torch.argmax(probs, dim=-1)  # [batch]

            # 置信度 = 最大概率
            confidence, _ = torch.max(probs, dim=-1)  # [batch]

            return {
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probs": probs
            }

    def save(self, save_path: str):
        """保存分类器"""
        torch.save({
            'classifier_state_dict': self.state_dict(),
            'encoder_dim': self.encoder_dim,
            'num_classes': self.num_classes
        }, save_path)
        logger.info(f"QueryClassifier已保存到: {save_path}")

    def load(self, load_path: str):
        """加载分类器"""
        checkpoint = torch.load(load_path, map_location='cpu')
        self.load_state_dict(checkpoint['classifier_state_dict'])
        logger.info(f"QueryClassifier已加载: {load_path}")


class SimpleRuleBasedClassifier:
    """
    简单的基于规则的分类器
    用于快速原型验证，不需要训练
    """

    def __init__(self):
        self.sop_keywords = [
            "sop", "流程", "步骤", "规范", "操作指南",
            "客服", "退款", "申请", "处理", "工单",
            "标准", "要求", "政策", "按照"
        ]

        self.reasoning_keywords = [
            "解释", "为什么", "原理", "如何理解",
            "分析", "推理", "证明", "计算",
            "什么是", "概念", "定义"
        ]

    def predict(
        self,
        queries: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        基于关键词匹配的简单分类

        Args:
            queries: 查询文本列表

        Returns:
            Dict包含predicted_class, confidence, probs
        """
        batch_size = len(queries)
        predicted_classes = []
        confidences = []

        for query in queries:
            query_lower = query.lower()

            # 统计关键词匹配
            sop_score = sum(1 for kw in self.sop_keywords if kw in query_lower)
            reasoning_score = sum(1 for kw in self.reasoning_keywords if kw in query_lower)

            # 判断
            if sop_score > reasoning_score:
                predicted_class = 1  # SOP
                total = sop_score + reasoning_score
                confidence = sop_score / total if total > 0 else 0.5
            elif reasoning_score > sop_score:
                predicted_class = 0  # Reasoning
                total = sop_score + reasoning_score
                confidence = reasoning_score / total if total > 0 else 0.5
            else:
                # 无法判断，低置信度，默认reasoning
                predicted_class = 0
                confidence = 0.3

            # 限制置信度范围
            confidence = min(max(confidence, 0.3), 0.95)

            predicted_classes.append(predicted_class)
            confidences.append(confidence)

        # 构造probs
        probs = torch.zeros(batch_size, 2)
        for i, (cls, conf) in enumerate(zip(predicted_classes, confidences)):
            if cls == 0:
                probs[i, 0] = conf
                probs[i, 1] = 1 - conf
            else:
                probs[i, 0] = 1 - conf
                probs[i, 1] = conf

        return {
            "predicted_class": torch.tensor(predicted_classes),
            "confidence": torch.tensor(confidences),
            "probs": probs
        }


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    # 测试SimpleRuleBasedClassifier
    print("=" * 50)
    print("测试SimpleRuleBasedClassifier")
    print("=" * 50)

    classifier = SimpleRuleBasedClassifier()

    test_queries = [
        "请按照SOP流程处理客户退款申请",
        "解释一下什么是量子纠缠？",
        "客服工单标准处理流程",
        "为什么神经网络能够学习？",
        "今天天气怎么样"  # 边界case
    ]

    results = classifier.predict(test_queries)

    for i, query in enumerate(test_queries):
        cls = results['predicted_class'][i].item()
        conf = results['confidence'][i].item()
        cls_name = "Reasoning" if cls == 0 else "SOP"

        print(f"\nQuery: {query}")
        print(f"预测类别: {cls_name}")
        print(f"置信度: {conf:.3f}")
        print(f"概率分布: Reasoning={results['probs'][i, 0]:.3f}, SOP={results['probs'][i, 1]:.3f}")

    print("\n测试完成!")
