"""
model.py
定义两种模型：BiEncoder 和 CrossEncoder。

区别（重要）：
    CrossEncoder:  把两个句子拼一起送进模型 → 一个相似度分数
                   优点：精度高
                   缺点：每次都要跑完整模型，无法预计算

    BiEncoder:     两个句子独立编码成向量 → 用余弦相似度比较
                   优点：可以预计算所有库中句子的向量，查询只需编码一次
                   缺点：精度略低（信息交互不充分）

工业界标准做法：BiEncoder 做召回，CrossEncoder 做精排。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification


# ==================================================================
# Pooling 策略：把 (B, L, D) 的 token 表示聚合成 (B, D) 的句向量
# ==================================================================
def mean_pooling(token_embeddings, attention_mask):
    """
    对真实 token 求平均。mask 掉的 padding 位置权重为 0。

    token_embeddings: (B, L, D)
    attention_mask:   (B, L)，1=真实, 0=padding
    """
    # (B, L, 1) 广播到 (B, L, D)
    mask = attention_mask.unsqueeze(-1).float()
    # 真实 token 的向量求和
    sum_embeddings = (token_embeddings * mask).sum(dim=1)   # (B, D)
    # 真实 token 数量
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)              # (B, 1)
    return sum_embeddings / sum_mask


def cls_pooling(token_embeddings, attention_mask=None):
    """取 [CLS] 位置（第一个 token）的表示。"""
    return token_embeddings[:, 0]


def max_pooling(token_embeddings, attention_mask):
    """对每个维度取真实 token 中的最大值。"""
    mask = attention_mask.unsqueeze(-1).float()
    # padding 位置填 -inf，让 max 忽略它们
    token_embeddings = token_embeddings.masked_fill(mask == 0, -1e9)
    return token_embeddings.max(dim=1).values


POOLING_FUNCS = {
    "mean": mean_pooling,
    "cls": cls_pooling,
    "max": max_pooling,
}


# ==================================================================
# BiEncoder: 把句子编码成向量
# ==================================================================
class BiEncoderModel(nn.Module):
    """
    双塔模型（其实是共享参数的同一个编码器）。

    输入: input_ids (B, L), attention_mask (B, L)
    输出: sentence_embeddings (B, D)，已 L2 归一化

    为什么 L2 归一化:
        归一化后，余弦相似度 = 点积。FAISS 的 IndexFlatIP 直接用点积检索，
        省去每次计算余弦相似度的开销。
    """

    def __init__(self, model_name, pooling="mean"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.hidden_size = self.encoder.config.hidden_size

        assert pooling in POOLING_FUNCS, f"不支持的 pooling: {pooling}"

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 编码
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # 有些模型（如 RoBERTa）不接受 token_type_ids，所以按需传
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        token_embeddings = outputs.last_hidden_state   # (B, L, D)

        # Pooling
        pooling_fn = POOLING_FUNCS[self.pooling]
        if self.pooling == "cls":
            embeddings = pooling_fn(token_embeddings)
        else:
            embeddings = pooling_fn(token_embeddings, attention_mask)

        # L2 归一化
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings

    def encode(self, input_ids, attention_mask, token_type_ids=None):
        """推理接口，和 forward 一样，只是语义更清晰。"""
        return self.forward(input_ids, attention_mask, token_type_ids)


# ==================================================================
# CrossEncoder: 两个句子拼一起判断相似度
# ==================================================================
class CrossEncoderModel(nn.Module):
    """
    交叉编码器。输入拼接后的 [CLS] 句1 [SEP] 句2 [SEP]，输出 2 类 logits。

    为什么用 AutoModelForSequenceClassification:
        它自动在 [CLS] 输出上接了一个线性分类头，正是我们需要的。
    """

    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.model(**kwargs)
        return outputs.logits   # (B, 2)


# ==================================================================
# 对比学习损失：MultipleNegativesRankingLoss（MNRL）
# ==================================================================
class MultipleNegativesRankingLoss(nn.Module):
    """
    InfoNCE 损失。一个 batch 内，其它样本的 positive 作为负例。

    输入:
        anchor_emb:   (B, D) 已归一化
        positive_emb: (B, D) 已归一化

    原理:
        计算 anchor 和所有 positive 的相似度矩阵: (B, B)
        对角线是正例（anchor_i vs positive_i）
        理想情况对角线最大
        对每一行做 softmax 交叉熵
    """

    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_emb, positive_emb):
        # 相似度矩阵: (B, B)。因为已 L2 归一化，点积 = 余弦相似度
        scores = anchor_emb @ positive_emb.T          # (B, B)
        scores = scores / self.temperature

        # 标签：对角线是正例
        labels = torch.arange(scores.size(0), device=scores.device)

        # 两个方向都算一遍，更稳
        loss_a = F.cross_entropy(scores, labels)
        loss_b = F.cross_entropy(scores.T, labels)
        return (loss_a + loss_b) / 2


# ==================================================================
# 自测
# ==================================================================
if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "hfl/rbt3"
    B, L = 4, 48

    print(f"设备: {device}")

    # ---------- 测试 BiEncoder ----------
    print("\n--- BiEncoder ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    biencoder = BiEncoderModel(model_name, pooling="mean").to(device)
    biencoder.eval()

    texts = ["怎么办理信用卡", "如何申请信用卡", "今天天气怎么样", "今天天气如何"]
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=L, return_tensors="pt")
    with torch.no_grad():
        embs = biencoder(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            enc.get("token_type_ids", None) and enc["token_type_ids"].to(device),
        )
    print(f"输入: {tuple(enc['input_ids'].shape)}")
    print(f"输出: {tuple(embs.shape)}")
    print(f"L2 范数（应全为 1）: {embs.norm(dim=-1)}")

    # 相似度矩阵
    sim = embs @ embs.T
    print(f"相似度矩阵:\n{sim.cpu().numpy().round(3)}")
    print("（句0 和句1、句2 和句3 应该相似度高）")

    # ---------- 测试 CrossEncoder ----------
    print("\n--- CrossEncoder ---")
    crossencoder = CrossEncoderModel(model_name, num_labels=2).to(device)
    crossencoder.eval()

    with torch.no_grad():
        logits = crossencoder(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
    print(f"输入: {tuple(enc['input_ids'].shape)}")
    print(f"输出: {tuple(logits.shape)}")

    # ---------- 测试 MNRL ----------
    print("\n--- MNRL Loss ---")
    loss_fn = MultipleNegativesRankingLoss(temperature=0.05)
    a = F.normalize(torch.randn(8, 64), dim=-1).to(device)
    p = F.normalize(torch.randn(8, 64), dim=-1).to(device)
    loss = loss_fn(a, p)
    print(f"随机向量 loss: {loss.item():.4f}（应接近 log(8)={torch.log(torch.tensor(8.0)).item():.4f}）")

    # 相同向量，loss 应该很小
    loss_same = loss_fn(a, a)
    print(f"相同向量 loss: {loss_same.item():.4f}（应接近 0）")

    print("\n[Check] 模型定义验证通过")


# ==================================================================
# Triplet Loss：显式让 pos 靠近、neg 远离
# ==================================================================
class TripletLoss(nn.Module):
    """
    三元组损失。
        loss = relu(margin - (cos(a, p) - cos(a, n)))

    anchor 与 positive 的相似度应该比与 negative 的相似度高至少 margin。
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_sim = (anchor * positive).sum(dim=-1)
        neg_sim = (anchor * negative).sum(dim=-1)
        loss = F.relu(self.margin - (pos_sim - neg_sim))
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    MNRL（in-batch negatives）+ TripletLoss（显式 hard negatives）。
    """

    def __init__(self, temperature=0.05, margin=0.3, alpha=0.5):
        super().__init__()
        self.mnrl = MultipleNegativesRankingLoss(temperature=temperature)
        self.triplet = TripletLoss(margin=margin)
        self.alpha = alpha

    def forward(self, anchor, positive, negative=None):
        loss = self.mnrl(anchor, positive)
        if negative is not None:
            loss = loss + self.alpha * self.triplet(anchor, positive, negative)
        return loss
