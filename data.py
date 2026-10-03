"""
data.py
LCQMC 数据加载与预处理。

两种数据集：
    PairDataset:     返回 (句子1, 句子2, 标签)  —— 用于 CrossEncoder
    CorpusDataset:   返回 (单句, 索引)         —— 用于 BiEncoder 检索库

字段名说明（mteb/LCQMC）:
    sentence1: 第一个句子
    sentence2: 第二个句子
    score:     1=相似, 0=不相似
"""

from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from config import CFG


# ==================================================================
# Dataset 1: 句对（用于 CrossEncoder 训练和评估）
# ==================================================================
class PairDataset(Dataset):
    """
    每个样本返回:
        input_ids:      (max_seq_len,)
        attention_mask: (max_seq_len,)
        token_type_ids: (max_seq_len,)  区分两个句子（0=句1, 1=句2）
        label:          int
    """

    def __init__(self, sentences1, sentences2, labels, tokenizer, max_len):
        self.s1 = sentences1
        self.s2 = sentences2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.s1[idx],
            self.s2[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        # token_type_ids 不是所有模型都返回（如 RoBERTa 不返回）
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


# ==================================================================
# Dataset 2: 单句（用于 BiEncoder 训练）
# ==================================================================
class PairForBiEncoder(Dataset):
    """
    BiEncoder 训练需要的是 (anchor, positive) 两个句子。
    LCQMC 的 score=1 就是正例对，score=0 我们暂时丢掉（或作为难负例备用）。

    返回:
        anchor 的 input_ids / attention_mask
        positive 的 input_ids / attention_mask
    """

    def __init__(self, sentences1, sentences2, tokenizer, max_len):
        self.s1 = sentences1
        self.s2 = sentences2
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.s1)

    def _encode(self, text):
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        a_ids, a_mask = self._encode(self.s1[idx])
        p_ids, p_mask = self._encode(self.s2[idx])
        return {
            "anchor_input_ids": a_ids,
            "anchor_attention_mask": a_mask,
            "positive_input_ids": p_ids,
            "positive_attention_mask": p_mask,
        }


# ==================================================================
# Dataset 3: 纯单句（用于构建检索库）
# ==================================================================
class CorpusDataset(Dataset):
    """把一堆句子编码成向量，用于建索引。"""

    def __init__(self, sentences, tokenizer, max_len):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "index": torch.tensor(idx, dtype=torch.long),
        }


# ==================================================================
# 构建函数
# ==================================================================
def get_tokenizer(cfg=CFG):
    """加载 tokenizer。"""
    return AutoTokenizer.from_pretrained(cfg.model_name)


def load_lcqmc_raw(cfg=CFG):
    """加载原始 LCQMC 数据。"""
    print(f"[Data] 加载数据集: {cfg.dataset_name}")
    ds = load_dataset(cfg.dataset_name)
    return ds


def build_crossencoder_loaders(cfg=CFG):
    """
    为 CrossEncoder 构建 train/val/test 的 DataLoader。
    """
    ds = load_lcqmc_raw(cfg)
    tokenizer = get_tokenizer(cfg)

    def make_loader(split, shuffle):
        data = ds[split]
        dset = PairDataset(
            data["sentence1"], data["sentence2"], data["score"],
            tokenizer, cfg.max_seq_len,
        )
        return DataLoader(
            dset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
        )

    train_loader = make_loader("train", True)
    val_loader = make_loader("validation", False)
    test_loader = make_loader("test", False)

    print(f"[Data] train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    return train_loader, val_loader, test_loader, tokenizer


def build_biencoder_loaders(cfg=CFG):
    """
    为 BiEncoder 对比学习构建 DataLoader。

    只使用 score=1 的正例对。
    """
    ds = load_lcqmc_raw(cfg)
    tokenizer = get_tokenizer(cfg)

    # 过滤出正例
    train_data = ds["train"].filter(lambda x: x["score"] == 1)
    print(f"[Data] 训练集正例对: {len(train_data)} 条")

    train_dset = PairForBiEncoder(
        train_data["sentence1"], train_data["sentence2"],
        tokenizer, cfg.max_seq_len,
    )

    train_loader = DataLoader(
        train_dset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,   # 对比学习需要固定 batch 大小（in-batch negatives）
    )

    # 验证集仍用完整句对（包括负例），用于评估
    val_data = ds["validation"]
    val_dset = PairDataset(
        val_data["sentence1"], val_data["sentence2"], val_data["score"],
        tokenizer, cfg.max_seq_len,
    )
    val_loader = DataLoader(
        val_dset, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    test_data = ds["test"]
    test_dset = PairDataset(
        test_data["sentence1"], test_data["sentence2"], test_data["score"],
        tokenizer, cfg.max_seq_len,
    )
    test_loader = DataLoader(
        test_dset, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"[Data] train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    return train_loader, val_loader, test_loader, tokenizer
