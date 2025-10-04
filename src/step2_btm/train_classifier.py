"""
Query分类器训练脚本
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List
import logging
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import load_config, load_jsonl, save_jsonl
from src.step2_btm.query_classifier import QueryClassifier

logger = logging.getLogger(__name__)


class QueryClassificationDataset(Dataset):
    def __init__(self, queries: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.queries = queries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.queries[idx], max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_training_data(data_file: str):
    data = load_jsonl(data_file)
    queries = []
    labels = []
    for item in data:
        queries.append(item['query'])
        if 'label' in item:
            labels.append(item['label'])
        elif 'type' in item:
            labels.append(0 if item['type'].lower() == 'reasoning' else 1)
        else:
            raise ValueError(f"数据项缺少label或type字段")
    return queries, labels


def create_sample_training_data(output_file: str = "data/classifier/train.jsonl"):
    sample_data = [
        {"query": "解释一下什么是量子纠缠？", "type": "reasoning"},
        {"query": "为什么神经网络能够学习非线性函数？", "type": "reasoning"},
        {"query": "请按照SOP流程处理客户退款申请", "type": "sop"},
        {"query": "客服工单标准处理流程", "type": "sop"},
    ]
    for item in sample_data:
        item['label'] = 0 if item['type'] == 'reasoning' else 1
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_jsonl(sample_data, output_file)
    logger.info(f"示例数据已保存到: {output_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/classifier/train.jsonl")
    parser.add_argument("--create_sample", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.create_sample:
        create_sample_training_data(args.train_file)


if __name__ == "__main__":
    main()
