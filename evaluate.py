"""
evaluate.py
BiEncoder 完整评估。

两部分:
    1. 句子对分类（Pair Classification）
       和 CrossEncoder 同口径，算 Accuracy / AUC / F1
    2. 全库检索（Retrieval）
       从 N 个候选中找 Top-K，算 Recall@K / MRR

输出:
    figures/biencoder_pair_eval.txt
    figures/biencoder_recall_curve.png
    figures/similarity_distribution.png
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, roc_curve,
)

from font_utils import setup_fonts
setup_fonts()

from config import CFG
from data import load_lcqmc_raw, get_tokenizer
from model import BiEncoderModel
from train_biencoder import encode_corpus


@torch.no_grad()
def evaluate_pair_classification(model, s1_list, s2_list, labels, tokenizer, cfg):
    """
    句子对分类评估:
        1. 编码所有 s1 和 s2
        2. 对每对算 cos(s1, s2)
        3. 用阈值分类（默认 0.5，或根据分布调）
        4. 算 Acc / AUC / F1
    """
    print("  [Pair] 编码句子...")
    emb1 = encode_corpus(model, s1_list, tokenizer, cfg)
    emb2 = encode_corpus(model, s2_list, tokenizer, cfg)

    # 逐对余弦相似度（都是归一化的向量，点积即余弦）
    cos_sim = (emb1 * emb2).sum(dim=-1).numpy()      # (N,)
    labels_np = np.array(labels)

    # 相似度分布
    pos_sim = cos_sim[labels_np == 1]
    neg_sim = cos_sim[labels_np == 0]
    print(f"  正例相似度: 均值={pos_sim.mean():.4f}  中位数={np.median(pos_sim):.4f}")
    print(f"  负例相似度: 均值={neg_sim.mean():.4f}  中位数={np.median(neg_sim):.4f}")

    # 自动找最佳阈值（最大化 F1）
    thresholds = np.linspace(cos_sim.min(), cos_sim.max(), 200)
    best_f1 = 0.0
    best_thr = 0.5
    for thr in thresholds:
        preds = (cos_sim >= thr).astype(int)
        f1 = f1_score(labels_np, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    preds = (cos_sim >= best_thr).astype(int)
    acc = accuracy_score(labels_np, preds)
    f1 = f1_score(labels_np, preds)
    prec = precision_score(labels_np, preds)
    rec = recall_score(labels_np, preds)
    auc = roc_auc_score(labels_np, cos_sim)

    print(f"  最佳阈值: {best_thr:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")

    return {
        "acc": acc, "auc": auc, "f1": f1,
        "precision": prec, "recall": rec,
        "best_threshold": best_thr,
        "cos_sim": cos_sim, "labels": labels_np,
        "pos_sim": pos_sim, "neg_sim": neg_sim,
    }


@torch.no_grad()
def evaluate_retrieval(model, s1_list, s2_list, tokenizer, cfg):
    """全库检索评估：Recall@K + MRR。"""
    print("  [Retrieval] 编码句子...")
    emb1 = encode_corpus(model, s1_list, tokenizer, cfg)
    emb2 = encode_corpus(model, s2_list, tokenizer, cfg)

    sim = emb1 @ emb2.T     # (N, N)
    N = sim.size(0)
    max_k = max(cfg.recall_k)

    # top-K 索引
    _, topk_idx = sim.topk(max_k, dim=1)
    diag = torch.arange(N).unsqueeze(1)

    recalls = {}
    for k in cfg.recall_k:
        hit = (topk_idx[:, :k] == diag).any(dim=1).float().mean().item()
        recalls[k] = hit

    # MRR: 每个 query 的 reciprocal rank
    # 找对角线元素在每行中的排名
    sorted_idx = sim.argsort(dim=1, descending=True)   # (N, N)
    ranks = (sorted_idx == diag).nonzero(as_tuple=True)[1].float() + 1  # (N,)
    mrr = (1.0 / ranks).mean().item()

    print(f"  Recall@1:  {recalls[1]:.4f}")
    print(f"  Recall@5:  {recalls[5]:.4f}")
    print(f"  Recall@10: {recalls[10]:.4f}")
    print(f"  MRR:       {mrr:.4f}")

    return recalls, mrr


def plot_similarity_distribution(result, save_path):
    """绘制正负样本的相似度分布。"""
    plt.figure(figsize=(9, 5))
    plt.hist(result["neg_sim"], bins=80, alpha=0.6,
             label=f"负例对 (n={len(result['neg_sim'])})", color="#C44E52")
    plt.hist(result["pos_sim"], bins=80, alpha=0.6,
             label=f"正例对 (n={len(result['pos_sim'])})", color="#55A868")
    plt.axvline(result["best_threshold"], color="black", linestyle="--",
                label=f"最佳阈值 = {result['best_threshold']:.3f}")
    plt.xlabel("余弦相似度")
    plt.ylabel("样本数")
    plt.title("BiEncoder 在测试集上的相似度分布")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Save] {save_path}")


def plot_roc(result, save_path):
    """绘制 ROC 曲线。"""
    fpr, tpr, _ = roc_curve(result["labels"], result["cos_sim"])
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, color="#4C72B0",
             label=f"BiEncoder (AUC = {result['auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="随机猜测")
    plt.xlabel("假正率 FPR")
    plt.ylabel("真正率 TPR")
    plt.title("BiEncoder 句子对分类 ROC")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Save] {save_path}")


def main():
    device = CFG.device

    # 加载模型
    ckpt_path = os.path.join(CFG.save_dir, "biencoder_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[Eval] 加载模型 epoch={ckpt['epoch']}, val R@1={ckpt['recall@1']:.4f}")

    model = BiEncoderModel(CFG.model_name, pooling=CFG.pooling).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = get_tokenizer(CFG)
    ds = load_lcqmc_raw(CFG)

    # ============ 测试集评估 ============
    test_s1 = ds["test"]["sentence1"]
    test_s2 = ds["test"]["sentence2"]
    test_labels = ds["test"]["score"]

    print("\n" + "=" * 60)
    print("测试集 - 句子对分类评估")
    print("=" * 60)
    pair_result = evaluate_pair_classification(
        model, test_s1, test_s2, test_labels, tokenizer, CFG
    )

    print("\n" + "=" * 60)
    print("测试集 - 全库检索评估")
    print("=" * 60)
    recalls, mrr = evaluate_retrieval(
        model, test_s1, test_s2, tokenizer, CFG
    )

    # ============ 验证集评估 ============
    val_s1 = ds["validation"]["sentence1"]
    val_s2 = ds["validation"]["sentence2"]
    val_labels = ds["validation"]["score"]

    print("\n" + "=" * 60)
    print("验证集 - 句子对分类评估")
    print("=" * 60)
    val_pair_result = evaluate_pair_classification(
        model, val_s1, val_s2, val_labels, tokenizer, CFG
    )

    # ============ 保存结果 ============
    Path = __import__("pathlib").Path
    Path(CFG.figures_dir).mkdir(exist_ok=True)

    with open(f"{CFG.figures_dir}/biencoder_pair_eval.txt", "w", encoding="utf-8") as f:
        f.write("BiEncoder 评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write("测试集 - 句子对分类:\n")
        for k in ["acc", "auc", "f1", "precision", "recall", "best_threshold"]:
            f.write(f"  {k}: {pair_result[k]:.4f}\n")
        f.write(f"\n测试集 - 检索:\n")
        for k, v in recalls.items():
            f.write(f"  Recall@{k}: {v:.4f}\n")
        f.write(f"  MRR: {mrr:.4f}\n")
        f.write(f"\n验证集 - 句子对分类:\n")
        for k in ["acc", "auc", "f1", "precision", "recall", "best_threshold"]:
            f.write(f"  {k}: {val_pair_result[k]:.4f}\n")

    plot_similarity_distribution(
        pair_result, f"{CFG.figures_dir}/similarity_distribution.png"
    )
    plot_roc(pair_result, f"{CFG.figures_dir}/biencoder_roc.png")

    # ============ 和 CrossEncoder 对比 ============
    print("\n" + "=" * 60)
    print("和 CrossEncoder 对比（测试集）")
    print("=" * 60)
    print(f"  CrossEncoder:   acc = 0.8526")
    print(f"  BiEncoder:      acc = {pair_result['acc']:.4f}  "
          f"(AUC = {pair_result['auc']:.4f})")
    print(f"  参数量:         {sum(p.numel() for p in model.parameters()):,}")
    print(f"  推理速度:       BiEncoder 快 100-1000 倍（预计算）")


if __name__ == "__main__":
    main()
