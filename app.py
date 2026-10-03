"""
app.py
双模型语义搜索对比 Demo。

对比:
    - rbt3 fine-tuned:  从零微调，38M 参数
    - BGE-small-zh-v1.5: 工业级预训练，24M 参数

输入一次查询，两个模型并排检索，直接看到差距。
"""

import os
import time
import pickle
import numpy as np
import torch
import faiss
import streamlit as st

from config import CFG
from data import get_tokenizer
from model import BiEncoderModel


st.set_page_config(
    page_title="语义搜索双模型对比",
    page_icon="🔍",
    layout="wide",
)


# ==================================================================
# 加载资源（缓存）
# ==================================================================
@st.cache_resource
def load_rbt3_searcher():
    """加载自己微调的 rbt3 模型 + FAISS 索引。"""
    device = CFG.device

    index_dir = os.path.join(CFG.save_dir, "faiss_index")
    with open(os.path.join(index_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    sentences = meta["sentences"]
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))

    ckpt = torch.load(
        os.path.join(CFG.save_dir, "biencoder_best.pt"),
        map_location=device, weights_only=False,
    )
    model = BiEncoderModel(CFG.model_name, pooling=CFG.pooling).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "name": "rbt3 fine-tuned",
        "sentences": sentences,
        "index": index,
        "model": model,
        "tokenizer": get_tokenizer(CFG),
        "device": device,
        "max_seq_len": meta["max_seq_len"],
        "dim": meta["embedding_dim"],
    }


@st.cache_resource
def load_bge_searcher():
    """
    加载 BGE 模型，动态编码句子库并建 FAISS 索引。
    首次运行需要 10-20 秒编码 12500 条句子，之后缓存。
    """
    from sentence_transformers import SentenceTransformer

    device = CFG.device

    # 复用 rbt3 的句子库（保证两个模型对比的是同一个库）
    index_dir = os.path.join(CFG.save_dir, "faiss_index")
    with open(os.path.join(index_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    sentences = meta["sentences"]

    model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)
    # BGE 官方推荐：查询句加前缀"为这个句子生成表示以用于检索相关文章："
    # 这里为了和 rbt3 公平对比，不加前缀

    embeddings = model.encode(
        sentences, batch_size=256, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {
        "name": "BGE-small-zh-v1.5",
        "sentences": sentences,
        "index": index,
        "model": model,
        "dim": embeddings.shape[1],
    }


# ==================================================================
# 编码与检索
# ==================================================================
@torch.no_grad()
def encode_query_rbt3(searcher, text):
    enc = searcher["tokenizer"](
        text, truncation=True,
        max_length=searcher["max_seq_len"],
        padding="max_length",
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(searcher["device"])
    mask = enc["attention_mask"].to(searcher["device"])
    tti = enc.get("token_type_ids")
    if tti is not None:
        tti = tti.to(searcher["device"])
    emb = searcher["model"](ids, mask, tti)
    return emb.cpu().numpy().astype("float32")


def encode_query_bge(searcher, text):
    emb = searcher["model"].encode(
        [text], normalize_embeddings=True, convert_to_numpy=True,
    )
    return emb.astype("float32")


def search(searcher, query, top_k, model_type):
    """返回 (结果, 编码耗时 ms, 检索耗时 ms)。"""
    t0 = time.time()
    if model_type == "rbt3":
        q_emb = encode_query_rbt3(searcher, query)
    else:
        q_emb = encode_query_bge(searcher, query)
    encode_time = (time.time() - t0) * 1000

    t0 = time.time()
    scores, indices = searcher["index"].search(q_emb, top_k)
    search_time = (time.time() - t0) * 1000

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "sentence": searcher["sentences"][idx],
                "score": float(score),
            })
    return results, encode_time, search_time


# ==================================================================
# 页面
# ==================================================================
st.title("🔍 中文语义搜索：双模型对比")
st.markdown(
    "输入一句话，**两个模型同时检索**，并排看到 Top-K 结果和耗时。"
    "这直接反映「从零微调」和「工业级预训练」的差距。"
)

# 加载模型
with st.spinner("加载 rbt3 模型和索引..."):
    searcher_rbt3 = load_rbt3_searcher()

bge_available = True
try:
    with st.spinner("加载 BGE 模型并编码句子库（首次约 15 秒）..."):
        searcher_bge = load_bge_searcher()
except Exception as e:
    bge_available = False
    st.warning(f"BGE 模型加载失败，将只显示 rbt3 结果: {e}")

# 侧边栏
with st.sidebar:
    st.header("ℹ️ 关于")
    st.markdown(f"""
    **索引库**: {len(searcher_rbt3['sentences']):,} 条句子
    **索引类型**: FAISS IndexFlatIP

    **模型信息**:
    - rbt3: {searcher_rbt3['dim']} 维, 38M 参数
    - BGE: {searcher_bge['dim'] if bge_available else 'N/A'} 维, 24M 参数
    """)

    st.header("⚙️ 参数")
    top_k = st.slider("返回 Top-K", 1, 20, 5)

    st.header("💡 示例查询")
    examples = [
        "怎么办理信用卡",
        "感冒了吃什么药",
        "推荐一部科幻电影",
        "北京哪有好玩的景点",
        "如何学习编程",
    ]
    for ex in examples:
        st.markdown(f"- {ex}")

# 输入区
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "输入查询",
        value="怎么办理信用卡",
        label_visibility="collapsed",
        placeholder="输入一句话...",
    )
with col2:
    search_btn = st.button("🔍 对比搜索", use_container_width=True, type="primary")

# 结果
if search_btn and query.strip():
    # ---------- rbt3 检索 ----------
    with st.spinner("rbt3 检索中..."):
        results_rbt3, enc_t_rbt3, srch_t_rbt3 = search(
            searcher_rbt3, query, top_k, "rbt3"
        )

    # ---------- BGE 检索 ----------
    if bge_available:
        with st.spinner("BGE 检索中..."):
            results_bge, enc_t_bge, srch_t_bge = search(
                searcher_bge, query, top_k, "bge"
            )
    else:
        results_bge = None

    st.markdown("---")
    st.markdown(f"### 查询: `{query}`")

    # ---------- 并排展示 ----------
    if bge_available:
        col_left, col_right = st.columns(2)
    else:
        col_left = st.container()
        col_right = None

    def render_results(col, title, subtitle, results, enc_t, srch_t, color):
        with col:
            st.markdown(f"#### {title}")
            st.caption(f"{subtitle}")
            st.caption(f"⚡ 编码 {enc_t:.1f}ms | 检索 {srch_t:.2f}ms")
            for i, r in enumerate(results, 1):
                score = r["score"]
                # 分数徽章
                st.markdown(
                    f"**{i}.** `{score:.4f}`  {r['sentence']}",
                    help=f"相似度 {score:.4f}",
                )
                st.progress(min(max(score, 0.0), 1.0))

    render_results(
        col_left,
        "🥉 rbt3 fine-tuned",
        "从零微调 · 38M 参数 · 768 维",
        results_rbt3, enc_t_rbt3, srch_t_rbt3, "#4C72B0",
    )

    if bge_available:
        render_results(
            col_right,
            "🏆 BGE-small-zh-v1.5",
            "工业预训练 · 24M 参数 · 512 维",
            results_bge, enc_t_bge, srch_t_bge, "#DD8452",
        )

        # ---------- 差异提示 ----------
        st.markdown("---")
        st.markdown("#### 📊 快速洞察")

        # Top-1 对比
        if results_rbt3 and results_bge:
            top1_rbt3 = results_rbt3[0]["sentence"]
            top1_bge = results_bge[0]["sentence"]
            if top1_rbt3 == top1_bge:
                st.success(f"✅ 两个模型的 Top-1 结果一致：**{top1_rbt3}**")
            else:
                st.info(
                    f"🔀 两模型 Top-1 不同：\n\n"
                    f"- rbt3: **{top1_rbt3}**  (相似度 {results_rbt3[0]['score']:.4f})\n"
                    f"- BGE:  **{top1_bge}**  (相似度 {results_bge[0]['score']:.4f})"
                )

elif search_btn:
    st.warning("请输入查询内容")

# 底部
st.markdown("---")
st.caption(
    "**测试集指标（LCQMC）**:  \n"
    "rbt3 fine-tuned: Pair Acc 0.784, R@1 0.747, R@5 0.976  \n"
    "BGE-small-zh-v1.5: 见 compare_bge.log 对比表"
)
