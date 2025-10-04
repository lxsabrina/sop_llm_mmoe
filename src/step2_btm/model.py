"""
Step 2: BTM (Branch-Train-Merge) 模型架构
核心: Token-level routing，融合两个模型的hidden states
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class TokenRouter(nn.Module):
    """Token-level router，为每个token决定使用哪个expert"""

    def __init__(
        self,
        hidden_dim: int,
        router_hidden_dim: int = 512,
        num_experts: int = 2,
        gate_activation: str = "softmax"
    ):
        """
        Args:
            hidden_dim: 模型hidden dimension
            router_hidden_dim: Router中间层维度
            num_experts: Expert数量 (foundation + sft = 2)
            gate_activation: Gate激活函数 (softmax, sigmoid, sparsemax)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.gate_activation = gate_activation

        # Router网络
        self.router_net = nn.Sequential(
            nn.Linear(hidden_dim, router_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(router_hidden_dim, num_experts)
        )

        logger.info(f"TokenRouter初始化: hidden_dim={hidden_dim}, experts={num_experts}")

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            gate_weights: [batch, seq_len, num_experts] - 归一化的权重
            gate_logits: [batch, seq_len, num_experts] - 原始logits (用于loss计算)
        """
        # 计算gate logits
        gate_logits = self.router_net(hidden_states)  # [batch, seq_len, num_experts]

        # 激活函数
        if self.gate_activation == "softmax":
            gate_weights = F.softmax(gate_logits, dim=-1)
        elif self.gate_activation == "sigmoid":
            gate_weights = torch.sigmoid(gate_logits)
            # 归一化
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"不支持的激活函数: {self.gate_activation}")

        return gate_weights, gate_logits


class LayerRouter(nn.Module):
    """Layer-level router，为每一层决定融合策略"""

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_experts: int = 2
    ):
        """
        Args:
            num_layers: 模型层数
            hidden_dim: Hidden dimension
            num_experts: Expert数量
        """
        super().__init__()

        self.num_layers = num_layers
        self.num_experts = num_experts

        # 每层一个router
        self.layer_routers = nn.ModuleList([
            TokenRouter(hidden_dim, num_experts=num_experts)
            for _ in range(num_layers)
        ])

        logger.info(f"LayerRouter初始化: {num_layers}层")

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: 当前层的hidden states
            layer_idx: 层索引

        Returns:
            gate_weights, gate_logits
        """
        return self.layer_routers[layer_idx](hidden_states)


class BTMModel(nn.Module):
    """
    BTM (Branch-Train-Merge) 模型
    融合foundation和sft两个模型
    """

    def __init__(
        self,
        foundation_model,
        sft_model,
        hidden_dim: int = 4096,
        num_layers: int = 32,
        router_hidden_dim: int = 512,
        router_type: str = "layer_level"
    ):
        """
        Args:
            foundation_model: Foundation模型 (冻结)
            sft_model: SFT模型 (冻结)
            hidden_dim: Hidden dimension (Qwen3-8B: 4096)
            num_layers: 层数 (Qwen3-8B: 32)
            router_hidden_dim: Router中间维度
            router_type: Router类型 (token_level, layer_level)
        """
        super().__init__()

        self.foundation_model = foundation_model
        self.sft_model = sft_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.router_type = router_type

        # 冻结两个expert模型
        for param in self.foundation_model.parameters():
            param.requires_grad = False
        for param in self.sft_model.parameters():
            param.requires_grad = False

        logger.info("Foundation和SFT模型已冻结")

        # Router
        if router_type == "layer_level":
            self.router = LayerRouter(
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                num_experts=2
            )
        elif router_type == "token_level":
            self.router = TokenRouter(
                hidden_dim=hidden_dim,
                router_hidden_dim=router_hidden_dim,
                num_experts=2
            )
        else:
            raise ValueError(f"不支持的router类型: {router_type}")

        # 输出head (使用foundation model的lm_head)
        self.lm_head = foundation_model.lm_head

        logger.info(f"BTM模型初始化完成: router_type={router_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_router_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len] - 用于计算loss
            output_router_logits: 是否输出router logits

        Returns:
            Dict包含:
                - logits: [batch, seq_len, vocab_size]
                - loss: 标量 (如果提供labels)
                - gate_weights: [batch, seq_len, num_experts] (如果output_router_logits=True)
                - load_balance_loss: 标量
        """
        batch_size, seq_len = input_ids.shape

        # 并行前向传播两个expert
        with torch.no_grad():
            outputs_foundation = self.foundation_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            outputs_sft = self.sft_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        # 获取所有层的hidden states
        # hidden_states: tuple of (num_layers+1) tensors, each [batch, seq_len, hidden_dim]
        hidden_states_foundation = outputs_foundation.hidden_states  # (layer0, layer1, ..., layer32)
        hidden_states_sft = outputs_sft.hidden_states

        # Layer-by-layer融合
        all_gate_weights = []
        all_gate_logits = []

        merged_hidden = None
        for layer_idx in range(self.num_layers):
            # 获取当前层的hidden states (注意: hidden_states[0]是embedding)
            h_foundation = hidden_states_foundation[layer_idx + 1]
            h_sft = hidden_states_sft[layer_idx + 1]

            # Router决定权重
            if self.router_type == "layer_level":
                # 使用上一层的merged hidden作为router输入
                router_input = merged_hidden if merged_hidden is not None else hidden_states_foundation[layer_idx]
                gate_weights, gate_logits = self.router(router_input, layer_idx)
            else:
                # Token-level: 使用当前层的平均hidden
                router_input = (h_foundation + h_sft) / 2
                gate_weights, gate_logits = self.router(router_input)

            # 融合: weighted sum
            # gate_weights: [batch, seq_len, 2]
            merged_hidden = gate_weights[:, :, 0:1] * h_foundation + \
                           gate_weights[:, :, 1:2] * h_sft

            all_gate_weights.append(gate_weights)
            all_gate_logits.append(gate_logits)

        # 最后一层的hidden states通过lm_head
        logits = self.lm_head(merged_hidden)  # [batch, seq_len, vocab_size]

        # 计算loss
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        # Load balance loss
        # 目标: 让两个expert被均衡使用
        gate_weights_mean = torch.stack(all_gate_weights).mean(dim=[0, 1, 2])  # [num_experts]
        load_balance_loss = torch.var(gate_weights_mean)  # 方差越小越平衡

        # Sparsity loss (可选)
        # 鼓励gate做出明确选择
        gate_weights_concat = torch.stack(all_gate_weights).mean(dim=0)  # [batch, seq_len, num_experts]
        sparsity_loss = -torch.sum(gate_weights_concat * torch.log(gate_weights_concat + 1e-8))

        # 返回结果
        result = {
            "logits": logits,
            "load_balance_loss": load_balance_loss,
            "sparsity_loss": sparsity_loss
        }

        if loss is not None:
            result["loss"] = loss

        if output_router_logits:
            result["gate_weights"] = gate_weights_concat
            result["all_gate_weights"] = all_gate_weights

        return result

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        生成文本 (autoregressive)

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            max_new_tokens: 最大生成token数
            temperature: 温度
            top_p: Top-p采样
            do_sample: 是否采样

        Returns:
            generated_ids: [batch, seq_len + max_new_tokens]
        """
        self.eval()

        batch_size = input_ids.shape[0]
        device = input_ids.device

        generated_ids = input_ids.clone()

        with torch.no_grad():
            for step in range(max_new_tokens):
                # 前向传播 (使用当前已生成的序列)
                outputs = self.forward(
                    input_ids=generated_ids,
                    attention_mask=attention_mask
                )

                # 获取最后一个token的logits
                next_token_logits = outputs["logits"][:, -1, :] / temperature

                # Top-p采样
                if do_sample:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # 拼接
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)

                # 更新attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                    ], dim=-1)

                # 检查eos
                if (next_token == self.foundation_model.config.eos_token_id).all():
                    break

        return generated_ids

    def save_router(self, save_path: str):
        """只保存router参数"""
        torch.save({
            'router_state_dict': self.router.state_dict(),
            'router_type': self.router_type,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        }, save_path)
        logger.info(f"Router参数已保存到: {save_path}")

    def load_router(self, load_path: str):
        """加载router参数"""
        checkpoint = torch.load(load_path)
        self.router.load_state_dict(checkpoint['router_state_dict'])
        logger.info(f"Router参数已加载: {load_path}")


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)

    # 创建dummy models
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
    config.num_hidden_layers = 4  # 简化测试

    print("创建测试模型...")
    # 这里只是结构测试，实际使用时会加载真实模型
    # foundation_model = AutoModelForCausalLM.from_config(config)
    # sft_model = AutoModelForCausalLM.from_config(config)

    print("BTM模型架构测试完成")
