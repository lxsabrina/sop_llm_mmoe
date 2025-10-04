import json
from typing import List, Dict, Iterator
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class ConversationDataset(Dataset):
    """对话数据集"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        """
        Args:
            data_path: JSONL文件路径，每行格式: {"input": "...", "output": "..."}
            tokenizer: Tokenizer
            max_length: 最大长度
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data()

    def load_data(self) -> List[Dict]:
        """加载数据"""
        data = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        data.append(item)
            logger.info(f"加载数据: {len(data)}条，来自 {self.data_path}")
        except FileNotFoundError:
            logger.warning(f"数据文件不存在: {self.data_path}")
        except Exception as e:
            logger.error(f"加载数据失败: {e}")

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建prompt (Qwen格式)
        messages = [
            {"role": "user", "content": item["input"]}
        ]

        # 使用apply_chat_template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        input_ids = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )["input_ids"].squeeze(0)

        # Target
        if "output" in item:
            target_text = item["output"]
            target_ids = self.tokenizer(
                target_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )["input_ids"].squeeze(0)
        else:
            target_ids = None

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "input_text": item["input"],
            "target_text": item.get("output", "")
        }


def create_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    shuffle: bool = False
) -> DataLoader:
    """创建DataLoader"""
    dataset = ConversationDataset(data_path, tokenizer, max_length)

    def collate_fn(batch):
        """处理batch，支持动态padding"""
        input_ids = [item["input_ids"] for item in batch]
        target_ids = [item["target_ids"] for item in batch if item["target_ids"] is not None]

        # Padding
        input_ids_padded = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt"
        )

        result = {
            "input_ids": input_ids_padded["input_ids"],
            "attention_mask": input_ids_padded["attention_mask"],
            "input_texts": [item["input_text"] for item in batch],
            "target_texts": [item["target_text"] for item in batch]
        }

        if target_ids:
            target_ids_padded = tokenizer.pad(
                {"input_ids": target_ids},
                padding=True,
                return_tensors="pt"
            )
            result["target_ids"] = target_ids_padded["input_ids"]

        return result

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

    return dataloader


def load_jsonl(file_path: str) -> List[Dict]:
    """简单加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """保存到JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def create_sample_data():
    """创建示例数据文件"""
    train_samples = [
        {
            "input": "请按照SOP流程处理客户退款申请",
            "output": "收到退款申请。根据SOP-REF-001标准流程：\n1. 核实订单信息\n2. 检查退款资格\n3. 审批退款金额\n4. 处理退款操作\n5. 通知客户"
        },
        {
            "input": "如何理解量子纠缠现象？",
            "output": "量子纠缠是量子力学中的重要现象。当两个粒子处于纠缠态时，对其中一个粒子的测量会立即影响另一个粒子的状态，无论它们相距多远。这种关联性超越了经典物理的局域性原理。"
        }
    ]

    eval_samples = [
        {
            "input": "客户要求加急处理订单",
            "output": "遵循SOP-ORD-002加急处理流程：\n1. 评估加急可行性\n2. 计算额外费用\n3. 获取客户确认\n4. 更新订单优先级\n5. 通知仓储部门"
        }
    ]

    import os
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/eval", exist_ok=True)

    save_jsonl(train_samples, "data/train/train.jsonl")
    save_jsonl(eval_samples, "data/eval/eval.jsonl")

    print("示例数据已创建:")
    print("- data/train/train.jsonl")
    print("- data/eval/eval.jsonl")


if __name__ == "__main__":
    # 创建示例数据
    create_sample_data()
