# 中文语义搜索：从零微调 BiEncoder + FAISS 检索

## 项目简介

本项目在 LCQMC 中文问句匹配数据集上，从零微调一个 **BiEncoder（双塔句向量模型）**，并用 **FAISS** 构建语义检索系统。同时训练 **CrossEncoder** 作为精度上限参照，与 BiEncoder 在精度和速度上做量化对比。

**核心特点**：

- 双模型对比：CrossEncoder（高精度）vs BiEncoder（高速）
- 对比学习：使用 MultipleNegativesRankingLoss + TripletLoss
- 向量检索：FAISS 索引，毫秒级从 12500 条句子中找 Top-K
- 混合精度训练：RTX 4090 + AMP，训练速度提升 2-3 倍
- 完整监控：Loss / GradNorm / LR / MFU / in-batch acc 实时记录
- Web Demo：Streamlit 双模型并排对比

## 数据集

**LCQMC**（Large-scale Chinese Question Matching Corpus），哈工大发布的大规模中文问句匹配数据集。

| Split | 样本数 | 说明 |
|-------|--------|------|
| Train | 238,766 | 58% 相似，42% 不相似 |
| Validation | 8,802 | 用于模型选择 |
| Test | 12,500 | 最终评估 |

字段：`sentence1` / `sentence2` / `score`（1=相似，0=不相似）

**句长分布**：中位数 10 字，90 分位 15 字，最长 46 字。`max_seq_len=48` 已覆盖 100%。

## 系统架构

### 两种模型的对比

```
CrossEncoder（精度上限参照）:
    [CLS] 句1 [SEP] 句2 [SEP] → BERT → 分类头 → 相似度
    优点：两句充分交互，精度高
    缺点：每对都要跑完整模型，无法预计算

BiEncoder（本项目核心）:
    句1 → BERT → mean pooling → 向量1 ┐
                                        → 余弦相似度
    句2 → BERT → mean pooling → 向量2 ┘
    优点：向量可预计算，检索速度快 100-1000 倍
    缺点：两句编码时互不可见，精度略低
```

### 工业界标准做法：召回 + 精排

```
查询 → BiEncoder 召回 Top-100 候选 → CrossEncoder 精排 Top-10 → 返回
```

BiEncoder 负责"快速缩小范围"，CrossEncoder 负责"精细排序"。

### 对比学习训练机制

```
一个 batch 有 B 对正例句对：
    (a_1, p_1), (a_2, p_2), ..., (a_B, p_B)

对 a_i 而言：
    正例:  p_i
    负例:  p_1, ..., p_{i-1}, p_{i+1}, ..., p_B  ← in-batch negatives

损失（InfoNCE / MNRL）：
    loss_i = -log( exp(sim(a_i, p_i)/τ) / Σ_j exp(sim(a_i, p_j)/τ) )
```

batch 越大，负例越多，训练信号越强。

## 实验结果

### 训练配置

| 参数 | CrossEncoder | BiEncoder |
|------|-------------|-----------|
| 基座模型 | hfl/rbt3 (38M) | hfl/rbt3 (38M) |
| 最大长度 | 48 | 48 |
| Batch size | 128 | 128 |
| 学习率 | 3e-5 | 2e-5 |
| Epochs | 6 | 4 |
| 优化器 | AdamW | AdamW |
| 调度器 | Warmup + Cosine | Warmup + Cosine |
| 混合精度 | AMP (FP16) | AMP (FP16) |
| 损失函数 | CrossEntropy | MNRL + Triplet |

### 主要结果

| 模型 | 参数量 | Test Acc | Test AUC | Recall@1 | Recall@5 | 推理速度 |
|------|--------|----------|----------|----------|----------|---------|
| CrossEncoder | 38M | **0.8526** | — | — | — | 慢（每对一次前向） |
| BiEncoder | 38M | 0.7844 | 0.8708 | 0.7470 | **0.9762** | **快 100-1000×** |

### 关键发现

**1. 双塔的精度代价约 7 个百分点**

CrossEncoder 0.8526 → BiEncoder 0.7844，换来的是 100-1000 倍的推理加速。这是工业界召回-精排架构的根本权衡。

**2. 全库检索 R@5 = 0.98**

从 12500 个候选中，正确句出现在前 5 名里的概率是 98%。这个精度足够支撑实用系统。

**3. 小模型的表示坍缩**

训练后期正例相似度 0.97、负例相似度 0.88，向量被挤在高相似度区域。这是 3 层小模型 + 短训练的必然现象。更大模型（如 BGE-large）会明显缓解。

**4. Hard negative 效果有限**

v1（仅 in-batch negatives）和 v2（+TripletLoss hard negatives）结果接近（Pair Acc 0.7750 vs 0.7844）。原因是 LCQMC 的负例都是跨主题的，不构成真正的"难"负例。

## 快速开始

### 环境

```bash
python >= 3.10
torch >= 2.5
transformers == 4.44.2
sentence-transformers >= 2.7
datasets >= 2.14
faiss-cpu >= 1.7
streamlit >= 1.30
```

### 安装

```bash
pip install -r requirements.txt
```

### 中国大陆用户

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

如果数据集/模型已缓存，可用离线模式避免网络检查：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### 训练 CrossEncoder

```bash
python train_crossencoder.py 2>&1 | tee train_crossencoder.log
```

### 训练 BiEncoder

```bash
python train_biencoder.py 2>&1 | tee train_biencoder.log
```

### 评估

```bash
python evaluate.py 2>&1 | tee evaluate.log
```

### 构建索引 + 搜索

```bash
python build_index.py     # 编码 12500 条句子，建 FAISS 索引
python search.py          # 命令行语义搜索
```

### Web Demo

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

浏览器打开 `http://localhost:8501`。

### TensorBoard 监控

```bash
tensorboard --logdir runs --port 6006
```

## 目录结构

```
02-semantic-search/
├── config.py                  # 全局配置
├── data.py                    # LCQMC 加载 + 4 种 Dataset
├── model.py                   # BiEncoder / CrossEncoder / MNRL / Triplet
├── train_crossencoder.py      # CrossEncoder 训练
├── train_biencoder.py         # BiEncoder 对比学习训练
├── evaluate.py                # 句子对分类 + 全库检索评估
├── build_index.py             # FAISS 索引构建
├── search.py                  # 命令行语义搜索
├── app.py                     # Streamlit 双模型对比 Demo
├── font_utils.py              # 中文字体配置
├── check_data.py              # 数据验证
├── requirements.txt
├── figures/                   # 可视化输出
│   ├── similarity_distribution.png
│   └── biencoder_roc.png
├── checkpoints/               # 模型权重
│   ├── crossencoder_best.pt
│   ├── biencoder_best.pt
│   └── faiss_index/           # FAISS 索引 + 元数据
└── runs/                      # TensorBoard 日志
```

## 核心技术解析

### 1. 句子向量与 Pooling

把 `(B, L, D)` 的 token 表示聚合成 `(B, D)` 的句向量。

**Mean Pooling（本项目使用）**：

```python
mask = attention_mask.unsqueeze(-1).float()      # (B, L, 1)
sum_emb = (token_embeddings * mask).sum(dim=1)   # (B, D)
sum_mask = mask.sum(dim=1).clamp(min=1e-9)       # (B, 1)
sentence_emb = sum_emb / sum_mask
```

**L2 归一化**：让余弦相似度 = 点积，方便 FAISS 检索。

```python
sentence_emb = F.normalize(sentence_emb, p=2, dim=-1)
```

### 2. MultipleNegativesRankingLoss

```python
def mnrl_loss(anchor, positive, temperature=0.05):
    # anchor, positive: (B, D) 已 L2 归一化
    scores = anchor @ positive.T              # (B, B) 相似度矩阵
    scores = scores / temperature
    labels = torch.arange(B)                  # 对角线是正例
    loss_a = cross_entropy(scores, labels)
    loss_b = cross_entropy(scores.T, labels)  # 双向对称
    return (loss_a + loss_b) / 2
```

### 3. TripletLoss

```python
def triplet_loss(anchor, positive, negative, margin=0.3):
    pos_sim = (anchor * positive).sum(dim=-1)
    neg_sim = (anchor * negative).sum(dim=-1)
    loss = relu(margin - (pos_sim - neg_sim))
    return loss.mean()
```

要求 anchor-positive 的相似度至少比 anchor-negative 高 margin。

### 4. FAISS 检索

```python
# 建索引
index = faiss.IndexFlatIP(D)   # 内积索引（向量已归一化，等价余弦）
index.add(corpus_embeddings)

# 查询
scores, indices = index.search(query_embedding, top_k)
```

IndexFlatIP 是暴力精确搜索，10 万条以内毫秒级。更大量级用 IVF / HNSW 近似索引。

### 5. 混合精度训练（AMP）

```python
scaler = torch.amp.GradScaler("cuda", enabled=True)

with torch.amp.autocast("cuda", enabled=True):
    logits = model(input_ids, attention_mask)
    loss = criterion(logits, labels)

scaler.scale(loss).backward()        # 梯度放大，避免 FP16 下溢
scaler.unscale_(optimizer)           # 反缩放，用于梯度裁剪
clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)               # 内部检查梯度是否 inf/nan
scaler.update()                      # 动态调整缩放系数
```

### 6. 训练监控指标

| 指标 | 计算 | 作用 |
|------|------|------|
| Loss | CrossEntropy / MNRL | 基础收敛 |
| GradNorm | 梯度 L2 范数 | 预警梯度爆炸/消失 |
| LR | scheduler 输出 | 验证调度器 |
| MFU | 6·N·tokens / (time·peak) | GPU 算力利用率 |
| In-batch acc | 对角是否最大 | 训练集上的区分度 |
| Pos/Neg sim | 正/负例平均相似度 | 监控表示质量 |

## 学习收获

1. **句向量的意义**：让语义相近的句子在向量空间里距离近，这是语义搜索、推荐、聚类的共同基础
2. **对比学习**：通过拉近正例、推远负例学习表示，核心是 MNRL 损失
3. **In-batch negatives**：巧妙利用 batch 内其他样本作为负例，训练效率极高
4. **BiEncoder vs CrossEncoder**：双塔快但精度略低，交叉编码器精度高但慢，工业界用"召回+精排"组合
5. **FAISS 向量检索**：从暴力搜索到 HNSW，毫秒级处理百万级向量
6. **表示坍缩**：小模型 + 短训练会让所有向量挤在高相似度区域
7. **Hard negative mining**：理论上重要，但取决于数据集质量
8. **混合精度训练**：FP16 + GradScaler，速度提升 2-3 倍
9. **Recall@K / MRR**：检索任务的标准指标，区别于分类任务的 Acc/F1
10. **完整的工程链路**：数据 → 模型 → 训练 → 评估 → 索引 → 检索 → Demo

## 局限与改进

### 当前局限

- 3 层 rbt3 是小模型，Pair Acc 0.78 已接近容量上限
- LCQMC 的负例多为跨主题，不构成真正的 hard negative
- 未使用 BGE 官方推荐的查询前缀（"为这个句子生成表示以用于检索相关文章："）

### 改进方向

- [ ] 换用更大模型（chinese-roberta-wwm-ext / BGE-large-zh）
- [ ] 从相似句池中挖掘真正的 hard negative（字面相似但语义不同）
- [ ] 使用 InfoNCE + 更大的 batch（4090 可支持 512）
- [ ] 尝试 HNSW 索引（近似检索，10 万+ 向量时更快）
- [ ] 添加 CrossEncoder 精排模块，实现完整召回-精排链路
- [ ] 部署为 FastAPI 服务

## 参考

- [Sentence-BERT 论文](https://arxiv.org/abs/1908.10084)
- [SimCSE 论文](https://arxiv.org/abs/2104.08821)
- [FAISS 文档](https://github.com/facebookresearch/faiss)
- [LCQMC 论文](https://www.aclweb.org/anthology/C18-1166/)
- [BGE 模型](https://huggingface.co/BAAI/bge-small-zh-v1.5)

## License

MIT License
