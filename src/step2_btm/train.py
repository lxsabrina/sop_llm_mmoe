"""
Step 2: BTM 训练脚本
只训练router，两个expert模型保持冻结
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import sys
import os
from pathlib import Path
from typing import Dict
import json

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import load_config, load_dual_models, create_dataloader
from src.step2_btm.model import BTMModel

logger = logging.getLogger(__name__)


class BTMTrainer:
    """BTM训练器"""

    def __init__(
        self,
        btm_model: BTMModel,
        tokenizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        config: Dict,
        output_dir: str
    ):
        """
        Args:
            btm_model: BTM模型
            tokenizer: Tokenizer
            train_dataloader: 训练数据
            eval_dataloader: 评估数据
            config: 配置字典
            output_dir: 输出目录
        """
        self.model = btm_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.output_dir = output_dir

        # 训练参数
        btm_config = config['btm']
        self.num_epochs = btm_config['num_epochs']
        self.gradient_accumulation_steps = btm_config['gradient_accumulation_steps']
        self.save_steps = btm_config['save_steps']
        self.eval_steps = btm_config['eval_steps']
        self.logging_steps = btm_config['logging_steps']

        # Loss权重
        self.load_balance_weight = btm_config['load_balance_weight']
        self.sparsity_weight = btm_config['sparsity_weight']

        # Optimizer (只优化router参数)
        self.optimizer = AdamW(
            self.model.router.parameters(),
            lr=btm_config['learning_rate']
        )

        # Scheduler
        total_steps = len(train_dataloader) * self.num_epochs // self.gradient_accumulation_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=btm_config['warmup_steps'],
            num_training_steps=total_steps
        )

        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float('inf')

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"BTM Trainer初始化完成")
        logger.info(f"总训练步数: {total_steps}")
        logger.info(f"可训练参数: {sum(p.numel() for p in self.model.router.parameters())}")

    def train(self):
        """训练主循环"""
        logger.info("=" * 50)
        logger.info("开始训练 BTM Router")
        logger.info("=" * 50)

        self.model.train()
        # 确保expert模型保持eval模式
        self.model.foundation_model.eval()
        self.model.sft_model.eval()

        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            epoch_loss = 0
            epoch_task_loss = 0
            epoch_lb_loss = 0
            epoch_sp_loss = 0

            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                # 准备数据
                input_ids = batch['input_ids'].to(self.model.foundation_model.device)
                attention_mask = batch['attention_mask'].to(self.model.foundation_model.device)

                # 构建labels (使用input_ids作为labels)
                labels = input_ids.clone()

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                # 计算总loss
                task_loss = outputs['loss']
                load_balance_loss = outputs['load_balance_loss']
                sparsity_loss = outputs['sparsity_loss']

                total_loss = task_loss + \
                           self.load_balance_weight * load_balance_loss + \
                           self.sparsity_weight * sparsity_loss

                # 反向传播
                total_loss = total_loss / self.gradient_accumulation_steps
                total_loss.backward()

                # 累积统计
                epoch_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_lb_loss += load_balance_loss.item()
                epoch_sp_loss += sparsity_loss.item()

                # 更新参数
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.router.parameters(), 1.0)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # 日志
                    if self.global_step % self.logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'task': f'{epoch_task_loss / (step + 1):.4f}',
                            'lb': f'{epoch_lb_loss / (step + 1):.4f}',
                            'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
                        })

                    # 评估
                    if self.global_step % self.eval_steps == 0:
                        eval_loss = self.evaluate()
                        logger.info(f"Step {self.global_step} - Eval Loss: {eval_loss:.4f}")

                        # 保存最佳模型
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self.save_checkpoint('best_router.pt')
                            logger.info(f"保存最佳模型 (loss={eval_loss:.4f})")

                        self.model.train()
                        self.model.foundation_model.eval()
                        self.model.sft_model.eval()

                    # 定期保存
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint(f'router_step_{self.global_step}.pt')

            # Epoch结束
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            logger.info(f"Epoch {epoch + 1} 完成 - Avg Loss: {avg_epoch_loss:.4f}")

        # 训练结束
        logger.info("\n训练完成!")
        self.save_checkpoint('final_router.pt')

    @torch.no_grad()
    def evaluate(self) -> float:
        """评估"""
        self.model.eval()

        total_loss = 0
        total_steps = 0

        for batch in self.eval_dataloader:
            input_ids = batch['input_ids'].to(self.model.foundation_model.device)
            attention_mask = batch['attention_mask'].to(self.model.foundation_model.device)
            labels = input_ids.clone()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs['loss'].item()
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        return avg_loss

    def save_checkpoint(self, filename: str):
        """保存checkpoint"""
        save_path = os.path.join(self.output_dir, filename)
        self.model.save_router(save_path)

        # 保存训练状态
        state_path = save_path.replace('.pt', '_state.pt')
        torch.save({
            'global_step': self.global_step,
            'best_eval_loss': self.best_eval_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict()
        }, state_path)

        logger.info(f"Checkpoint已保存: {save_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Step 2: BTM训练")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="配置文件")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint")

    args = parser.parse_args()

    # 日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/btm_training.log')
        ]
    )

    # 加载配置
    config = load_config(args.config)

    # 输出目录
    output_dir = args.output_dir or config['btm']['checkpoint_dir']
    os.makedirs(output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 加载模型
    logger.info("加载模型...")
    foundation_path = config['models']['foundation']['path']
    sft_path = config['models']['sft']['path']

    foundation_model, sft_model, tokenizer = load_dual_models(
        foundation_path,
        sft_path
    )

    # 创建BTM模型
    logger.info("创建BTM模型...")

    # 获取模型配置
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

    # 加载数据
    logger.info("加载数据...")
    train_dataloader = create_dataloader(
        data_path=config['data']['train_file'],
        tokenizer=tokenizer,
        batch_size=config['btm']['batch_size'],
        max_length=config['inference']['max_length'],
        shuffle=True
    )

    eval_dataloader = create_dataloader(
        data_path=config['data']['eval_file'],
        tokenizer=tokenizer,
        batch_size=config['btm']['batch_size'],
        max_length=config['inference']['max_length'],
        shuffle=False
    )

    logger.info(f"训练数据: {len(train_dataloader.dataset)}条")
    logger.info(f"评估数据: {len(eval_dataloader.dataset)}条")

    # 创建trainer
    trainer = BTMTrainer(
        btm_model=btm_model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        output_dir=output_dir
    )

    # 开始训练
    trainer.train()

    logger.info("全部完成!")


if __name__ == "__main__":
    main()
