"""
search.py
语义搜索：输入一句话，从索引库检索 Top-K 相似句。

用法:
    python search.py                        # 交互式
    python search.py "怎么办理信用卡"       # 单次查询
"""

import os
import sys
import time
import pickle
import numpy as np
import torch
import faiss

from config import CFG
from data import get_tokenizer
from model import BiEncoderModel


class SemanticSearcher:
    """语义搜索器，封装加载索引、编码查询、检索的逻辑。"""

    def __init__(self, index_dir=None, device=None):
        self.device = device or CFG.device
        self.index_dir = index_dir or os.path.join(CFG.save_dir, "faiss_index")

        # ---------- 加载元数据 ----------
        meta_path = os.path.join(self.index_dir, "metadata.pkl")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.sentences = meta["sentences"]
        self.max_seq_len = meta["max_seq_len"]
        print(f"[Search] 索引库: {len(self.sentences)} 条句子")

        # ---------- 加载 FAISS 索引 ----------
        index_path = os.path.join(self.index_dir, "index.faiss")
        self.index = faiss.read_index(index_path)
        print(f"[Search] FAISS 索引: {self.index.ntotal} 个向量，维度 {self.index.d}")

        # ---------- 加载模型 ----------
        ckpt_path = os.path.join(CFG.save_dir, "biencoder_best.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = BiEncoderModel(CFG.model_name, pooling=CFG.pooling).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.tokenizer = get_tokenizer(CFG)

    @torch.no_grad()
    def encode_query(self, text):
        """把查询文本编码成向量。"""
        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_seq_len,
            padding="max_length", return_tensors="pt",
        )
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        tti = enc.get("token_type_ids")
        if tti is not None:
            tti = tti.to(self.device)
        emb = self.model(ids, mask, tti)
        return emb.cpu().numpy().astype("float32")

    def search(self, query, top_k=5):
        """
        检索 Top-K 相似句。

        返回: [(句子, 相似度), ...]
        """
        # 1. 编码查询
        t0 = time.time()
        query_emb = self.encode_query(query)
        encode_time = (time.time() - t0) * 1000

        # 2. FAISS 检索
        t0 = time.time()
        scores, indices = self.index.search(query_emb, top_k)
        search_time = (time.time() - t0) * 1000

        # 3. 组装结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "sentence": self.sentences[idx],
                "score": float(score),
                "index": int(idx),
            })

        return results, encode_time, search_time


def print_results(query, results, encode_time, search_time):
    print(f"\n查询: {query}")
    print(f"编码耗时: {encode_time:.1f}ms  检索耗时: {search_time:.2f}ms")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] 相似度 {r['score']:.4f}  {r['sentence']}")
    print()


def main():
    searcher = SemanticSearcher()

    # 命令行参数模式
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results, et, st = searcher.search(query, top_k=5)
        print_results(query, results, et, st)
        return

    # 交互模式
    print("\n" + "=" * 70)
    print("语义搜索 (输入 q 退出)")
    print("=" * 70)

    # 预置几个示例
    examples = [
        "怎么办理信用卡",
        "推荐一部科幻电影",
        "感冒了吃什么药",
    ]
    print("\n示例查询:")
    for ex in examples:
        results, et, st = searcher.search(ex, top_k=3)
        print_results(ex, results, et, st)

    # 交互
    while True:
        try:
            query = input("\n输入查询 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() == "q":
            break
        results, et, st = searcher.search(query, top_k=5)
        print_results(query, results, et, st)


if __name__ == "__main__":
    main()
