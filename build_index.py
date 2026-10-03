"""
build_index.py
用 BiEncoder 编码句子库，构建 FAISS 索引。

索引类型:
    IndexFlatIP  精确内积检索（暴力搜索），数据量小（<100万）时够用
    索引保存为 .faiss 文件，对应的句子保存为 .txt
"""

import os
import time
import pickle
import numpy as np
import torch
import faiss

from config import CFG
from data import load_lcqmc_raw, get_tokenizer
from model import BiEncoderModel
from train_biencoder import encode_corpus


def build_index_from_model(model, sentences, tokenizer, cfg, save_dir):
    """
    把 sentences 编码为向量，建 FAISS 索引。
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---------- 1. 编码 ----------
    print(f"[Index] 编码 {len(sentences)} 条句子...")
    t0 = time.time()
    embeddings = encode_corpus(model, sentences, tokenizer, cfg, batch_size=256)
    encode_time = time.time() - t0
    print(f"[Index] 编码完成，耗时 {encode_time:.1f}s")

    emb_np = embeddings.numpy().astype("float32")   # (N, D)
    N, D = emb_np.shape
    print(f"[Index] 向量维度: {D}, 样本数: {N}")

    # ---------- 2. 建索引 ----------
    # IndexFlatIP: 内积索引。因为向量已 L2 归一化，内积 = 余弦相似度
    index = faiss.IndexFlatIP(D)
    index.add(emb_np)
    print(f"[Index] FAISS 索引已建立，共 {index.ntotal} 个向量")

    # ---------- 3. 保存 ----------
    index_path = os.path.join(save_dir, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"[Index] 索引保存到 {index_path}")

    # 句子列表和元数据也用 pickle 保存（比 txt 便于扩展）
    meta_path = os.path.join(save_dir, "metadata.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump({
            "sentences": sentences,
            "model_name": CFG.model_name,
            "pooling": CFG.pooling,
            "max_seq_len": CFG.max_seq_len,
            "embedding_dim": D,
        }, f)
    print(f"[Index] 元数据保存到 {meta_path}")

    # 文件大小
    index_size = os.path.getsize(index_path) / 1e6
    print(f"[Index] 索引文件大小: {index_size:.1f} MB")

    return index_path, meta_path


def main():
    device = CFG.device
    print(f"[Index] 设备: {device}")

    # ---------- 加载模型 ----------
    ckpt_path = os.path.join(CFG.save_dir, "biencoder_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[Index] 加载模型 epoch={ckpt['epoch']}, val R@1={ckpt['recall@1']:.4f}")

    model = BiEncoderModel(CFG.model_name, pooling=CFG.pooling).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = get_tokenizer(CFG)

    # ---------- 准备句子库 ----------
    # 用测试集的 sentence2 作为被检索的库（12500 条）
    # 查询用测试集的 sentence1
    ds = load_lcqmc_raw(CFG)
    corpus = ds["test"]["sentence2"]
    print(f"[Index] 句子库: {len(corpus)} 条（来自测试集 sentence2）")

    # ---------- 建索引 ----------
    save_dir = os.path.join(CFG.save_dir, "faiss_index")
    build_index_from_model(model, list(corpus), tokenizer, CFG, save_dir)

    print("\n[Index] 完成")
    print(f"    索引目录: {save_dir}")
    print(f"    下一步: python search.py")


if __name__ == "__main__":
    main()
