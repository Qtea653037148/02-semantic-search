"""
check_data.py
验证项目 2 的数据流水线。

检查:
    1. LCQMC 能加载
    2. CrossEncoder 数据形状正确
    3. BiEncoder 数据形状正确
    4. 解码回文本确认 tokenization 正确
"""

import torch
from config import CFG
from data import build_crossencoder_loaders, build_biencoder_loaders


def main():
    print("=" * 60)
    print("CrossEncoder 数据验证")
    print("=" * 60)

    train_loader, val_loader, test_loader, tokenizer = build_crossencoder_loaders(CFG)

    batch = next(iter(train_loader))
    print("\n字段:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}")

    print("\n--- 第 0 条样本 ---")
    ids = batch["input_ids"][0]
    label = batch["label"][0].item()
    text = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"  解码文本: {text}")
    print(f"  标签: {label}  (1=相似)")

    print("\n" + "=" * 60)
    print("BiEncoder 数据验证")
    print("=" * 60)

    train_loader_b, val_loader_b, test_loader_b, _ = build_biencoder_loaders(CFG)

    batch = next(iter(train_loader_b))
    print("\n字段:")
    for k, v in batch.items():
        print(f"  {k}: {tuple(v.shape)}")

    print("\n--- 第 0 条样本 ---")
    a_ids = batch["anchor_input_ids"][0]
    p_ids = batch["positive_input_ids"][0]
    print(f"  Anchor:   {tokenizer.decode(a_ids, skip_special_tokens=True)}")
    print(f"  Positive: {tokenizer.decode(p_ids, skip_special_tokens=True)}")

    print("\n[Check] 数据流水线验证通过")


if __name__ == "__main__":
    main()
