"""
Step 3: Distillation 训练
将BTM或Ensemble的能力蒸馏到单一模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Optional
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import load_config, load_dual_models, load_model_and_tokenizer, create_dataloader
from src.step2_btm.model import BTMModel

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """蒸馏训练器"""

    def __init__(
        self,
        student_model,
        teacher_model,  # BTM model或ensemble
        tokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        config: Dict,
        output_dir: str,
        teacher_type: str = "btm"  # btm or ensemble
    ):
        """
        Args:
            student_model: Student模型 (可训练)
            teacher_model: Teacher模型 (冻结)
            tokenizer: Tokenizer
            train_dataloader: 训练数据
            eval_dataloader: 评估数据
            config: 配置
            output_dir: 输出目录
            teacher_type: Teacher类型
        """
        self.student = student_model
        self.teacher = teacher_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.output_dir = output_dir
        self.teacher_type = teacher_type

        # 蒸馏参数
        distill_config = config['distill']
        self.temperature = distill_config['temperature']
        self.distill_loss_weight = distill_config['distill_loss_weight']
        self.task_loss_weight = distill_config['task_loss_weight']
        self.num_epochs = distill_config['num_epochs']
        self.gradient_accumulation_steps = distill_config['gradient_accumulation_steps']
        self.save_steps = distill_config['save_steps']
        self.eval_steps = distill_config['eval_steps']

        # 冻结teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # Optimizer (student全参数训练)
        self.optimizer = AdamW(
            self.student.parameters(),
            lr=distill_config['learning_rate']
        )

        # Scheduler
        total_steps = len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        warmup_steps = int(total_steps * distill_config['warmup_ratio'])

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float('inf')

        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Distillation Trainer初始化完成")
        logger.info(f"Teacher类型: {teacher_type}")
        logger.info(f"总训练步数: {total_steps}")
        logger.info(f"Student可训练参数: {sum(p.numel() for p in self.student.parameters() if p.requires_grad)}")

    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算蒸馏loss (KL divergence)

        Args:
            student_logits: [batch, seq_len, vocab_size]
            teacher_logits: [batch, seq_len, vocab_size]

        Returns:
            蒸馏loss
        """
        # Softmax with temperature
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL divergence
        kl_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        return kl_loss

    def train_step(self, batch) -> Dict[str, float]:
        """单步训练"""
        input_ids = batch['input_ids'].to(self.student.device)
        attention_mask = batch['attention_mask'].to(self.student.device)
        labels = input_ids.clone()

        # Teacher前向 (no grad)
        with torch.no_grad():
            if self.teacher_type == "btm":
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs['logits']
            else:
                # Ensemble: 需要实现
                teacher_outputs = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits

        # Student前向
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        student_logits = student_outputs.logits

        # 计算losses
        # 1. Task loss (CE with true labels)
        task_loss = student_outputs.loss

        # 2. Distillation loss (KL with teacher)
        distill_loss = self.compute_distillation_loss(student_logits, teacher_logits)

        # 总loss
        total_loss = self.task_loss_weight * task_loss + \
                    self.distill_loss_weight * distill_loss

        return {
            'total_loss': total_loss,
            'task_loss': task_loss.item(),
            'distill_loss': distill_loss.item()
        }

    def train(self):
        """训练主循环"""
        logger.info("=" * 50)
        logger.info("开始蒸馏训练")
        logger.info("=" * 50)

        self.student.train()

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            epoch_loss = 0
            epoch_task_loss = 0
            epoch_distill_loss = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                # 训练步
                losses = self.train_step(batch)

                # 反向传播
                total_loss = losses['total_loss'] / self.gradient_accumulation_steps
                total_loss.backward()

                # 累积统计
                epoch_loss += total_loss.item()
                epoch_task_loss += losses['task_loss']
                epoch_distill_loss += losses['distill_loss']

                # 更新参数
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # 日志
                    progress_bar.set_postfix({
                        'loss': f'{epoch_loss / (step + 1):.4f}',
                        'task': f'{epoch_task_loss / (step + 1):.4f}',
                        'distill': f'{epoch_distill_loss / (step + 1):.4f}',
                        'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                    })

                    # 评估
                    if self.global_step % self.eval_steps == 0:
                        eval_loss = self.evaluate()
                        logger.info(f"Step {self.global_step} - Eval Loss: {eval_loss:.4f}")

                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint('best_student')
                            logger.info(f"保存最佳模型 (loss={eval_loss:.4f})")

                        self.student.train()

                    # 定期保存
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint(f'student_step_{self.global_step}')

            # Epoch结束
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            logger.info(f"Epoch {epoch + 1} 完成 - Avg Loss: {avg_epoch_loss:.4f}")

        # 训练结束
        logger.info("\n蒸馏训练完成!")
        self.save_checkpoint('final_student')

    @torch.no_grad()
    def evaluate(self) -> float:
        """评估"""
        self.student.eval()

        total_loss = 0
        total_steps = 0

        for batch in self.eval_dataloader:
            input_ids = batch['input_ids'].to(self.student.device)
            attention_mask = batch['attention_mask'].to(self.student.device)
            labels = input_ids.clone()

            outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        return avg_loss

    def save_checkpoint(self, name: str):
        """保存checkpoint"""
        save_path = os.path.join(self.output_dir, name)

        # 保存整个student model
        self.student.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        logger.info(f"Student模型已保存: {save_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Step 3: Distillation训练")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件")
    parser.add_argument("--teacher_type", type=str, default="btm", choices=["btm", "ensemble"], help="Teacher类型")
    parser.add_argument("--btm_router", type=str, default=None, help="BTM router checkpoint (如果teacher_type=btm)")
    parser.add_argument("--student_init", type=str, default=None, help="Student初始化 (foundation/sft/path)")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")

    args = parser.parse_args()

    # 日志
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/distillation.log')
        ]
    )

    # 加载配置
    config = load_config(args.config)

    output_dir = args.output_dir or config['distill']['checkpoint_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 加载模型
    foundation_path = config['models']['foundation']['path']
    sft_path = config['models']['sft']['path']

    logger.info("加载teacher模型...")
    if args.teacher_type == "btm":
        # 加载BTM teacher
        foundation_model, sft_model, tokenizer = load_dual_models(
            foundation_path,
            sft_path
        )

        hidden_dim = foundation_model.config.hidden_size
        num_layers = foundation_model.config.num_hidden_layers

        teacher_model = BTMModel(
            foundation_model=foundation_model,
            sft_model=sft_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            router_hidden_dim=config['btm']['router_hidden_dim'],
            router_type=config['btm']['router_type']
        )

        # 加载router
        if args.btm_router:
            teacher_model.load_router(args.btm_router)
            logger.info(f"BTM router已加载: {args.btm_router}")
        else:
            logger.warning("未指定BTM router，使用随机初始化的router")

    else:
        # TODO: Ensemble teacher
        raise NotImplementedError("Ensemble teacher暂未实现")

    # 加载student模型
    logger.info("加载student模型...")
    student_init = args.student_init or config['distill']['student_init']

    if student_init == "foundation":
        student_path = foundation_path
    elif student_init == "sft":
        student_path = sft_path
    else:
        student_path = student_init

    student_model, tokenizer = load_model_and_tokenizer(student_path)
    logger.info(f"Student初始化自: {student_path}")

    # 加载数据
    logger.info("加载数据...")
    train_dataloader = create_dataloader(
        data_path=config['data']['train_file'],
        tokenizer=tokenizer,
        batch_size=config['distill']['batch_size'],
        max_length=config['inference']['max_length'],
        shuffle=True
    )

    eval_dataloader = create_dataloader(
        data_path=config['data']['eval_file'],
        tokenizer=tokenizer,
        batch_size=config['distill']['batch_size'],
        max_length=config['inference']['max_length'],
        shuffle=False
    )

    logger.info(f"训练数据: {len(train_dataloader.dataset)}条")
    logger.info(f"评估数据: {len(eval_dataloader.dataset)}条")

    # 创建trainer
    trainer = DistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        output_dir=output_dir,
        teacher_type=args.teacher_type
    )

    # 开始训练
    trainer.train()

    logger.info("全部完成!")


if __name__ == "__main__":
    main()
