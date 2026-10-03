"""
compare_with_bge.py
对比自己微调的 BiEncoder 和 BAAI/bge-small-zh-v1.5 的检索效果。
"""

import os
import time
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

from config import CFG
from data import load_lcqmc_raw
from train_biencoder import encode_corpus
from model import BiEncoderModel
from data import get_tokenizer


def evaluate_retrieval_recall(index, query_embs, target_embs, recall_k=[1, 5, 10]):
    """给定索引和查询向量，算 Recall@K。"""
    scores, indices = index.search(query_embs, max(recall_k))
    N = len(query_embs)
    diag = np.arange(N).reshape(-1, 1)
    recalls = {}
    for k in recall_k:
        hit = (indices[:, :k] == diag).any(axis=1).mean()
        recalls[k] = hit
    return recalls


def main():
    device = CFG.device
    ds = load_lcqmc_raw(CFG)
    test_s1 = list(ds["test"]["sentence1"])
    test_s2 = list(ds["test"]["sentence2"])
    print(f"[Compare] 测试集: {len(test_s1)} 对")

    results = {}

    # ============ 模型 1: 自己微调的 rbt3 ============
    print("\n" + "=" * 60)
    print("模型 1: rbt3 fine-tuned BiEncoder")
    print("=" * 60)
    ckpt_path = os.path.join(CFG.save_dir, "biencoder_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tokenizer = get_tokenizer(CFG)
    model = BiEncoderModel(CFG.model_name, pooling=CFG.pooling).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    t0 = time.time()
    q_emb = encode_corpus(model, test_s1, tokenizer, CFG).numpy().astype("float32")
    c_emb = encode_corpus(model, test_s2, tokenizer, CFG).numpy().astype("float32")
    encode_time = time.time() - t0
    print(f"编码耗时: {encode_time:.1f}s")

    index = faiss.IndexFlatIP(q_emb.shape[1])
    index.add(c_emb)
    recalls = evaluate_retrieval_recall(index, q_emb, c_emb)
    print(f"  R@1={recalls[1]:.4f}  R@5={recalls[5]:.4f}  R@10={recalls[10]:.4f}")
    results["rbt3 fine-tuned"] = {"recalls": recalls, "encode_time": encode_time}

    # ============ 模型 2: BGE small ============
    print("\n" + "=" * 60)
    print("模型 2: BAAI/bge-small-zh-v1.5 (24M 参数)")
    print("=" * 60)
    try:
        bge_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)
        t0 = time.time()
        q_emb = bge_model.encode(test_s1, batch_size=128, show_progress_bar=True,
                                  normalize_embeddings=True).astype("float32")
        c_emb = bge_model.encode(test_s2, batch_size=128, show_progress_bar=True,
                                  normalize_embeddings=True).astype("float32")
        encode_time = time.time() - t0

        index = faiss.IndexFlatIP(q_emb.shape[1])
        index.add(c_emb)
        recalls = evaluate_retrieval_recall(index, q_emb, c_emb)
        print(f"  R@1={recalls[1]:.4f}  R@5={recalls[5]:.4f}  R@10={recalls[10]:.4f}")
        results["BGE-small-zh-v1.5"] = {"recalls": recalls, "encode_time": encode_time}
    except Exception as e:
        print(f"  [Skip] BGE 加载失败: {e}")

    # ============ 汇总 ============
    print("\n" + "=" * 60)
    print("对比汇总")
    print("=" * 60)
    print(f"{'模型':<30} {'R@1':>8} {'R@5':>8} {'R@10':>8}")
    print("-" * 60)
    for name, r in results.items():
        rec = r["recalls"]
        print(f"{name:<30} {rec[1]:>8.4f} {rec[5]:>8.4f} {rec[10]:>8.4f}")


if __name__ == "__main__":
    main()
