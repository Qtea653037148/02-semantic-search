"""
train_biencoder.py
用对比学习训练 BiEncoder。

核心改进（v2）:
    - 引入 hard negatives：从 LCQMC 的 score=0 对中采样
    - 损失函数: MNRL (in-batch negatives) + TripletLoss (显式 hard negatives)
    - 评估口径: 句子对分类 (Acc/AUC) + 全库检索 (Recall@K)

监控指标:
    - Train: loss / in_batch_acc / grad_norm / lr / mfu / step_time
    - Val:   Recall@1/5/10
"""

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import CFG
from data import build_biencoder_triplet_loaders, load_lcqmc_raw, get_tokenizer
from model import BiEncoderModel, CombinedLoss


# ==================================================================
# 通用工具
# ==================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_gpu_peak_flops(use_amp):
    """返回 GPU 峰值算力（FLOPS）。"""
    if not torch.cuda.is_available():
        return 0.0
    name = torch.cuda.get_device_name(0)
    if "4090" in name:
        return 165e12 if use_amp else 82.6e12
    if "3090" in name:
        return 71e12 if use_amp else 35.6e12
    if "A100" in name:
        return 312e12 if use_amp else 156e12
    return 50e12


def build_optimizer_and_scheduler(model, cfg, total_steps):
    """AdamW + warmup + cosine 衰减。"""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr)
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler, warmup_steps


def compute_mfu(num_params, batch_size, seq_len, step_time, peak_flops):
    """
    MFU = 实际 FLOPs / 峰值 FLOPs。
    三元组训练每次要编码 3 个句子（anchor, positive, negative），所以 tokens 乘 3。
    """
    if peak_flops <= 0 or step_time <= 0:
        return 0.0
    tokens = 3 * batch_size * seq_len
    flops = 6 * num_params * tokens
    return flops / (step_time * peak_flops)


# ==================================================================
# 训练一个 epoch（三元组）
# ==================================================================
def train_one_epoch(
    model, loader, optimizer, scheduler, criterion, scaler,
    cfg, writer, global_step, epoch, peak_flops, num_params,
):
    model.train()
    total_loss = 0.0
    total_steps = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")

    for batch in pbar:
        a_ids = batch["anchor_input_ids"].to(cfg.device, non_blocking=True)
        a_mask = batch["anchor_attention_mask"].to(cfg.device, non_blocking=True)
        p_ids = batch["positive_input_ids"].to(cfg.device, non_blocking=True)
        p_mask = batch["positive_attention_mask"].to(cfg.device, non_blocking=True)
        n_ids = batch["negative_input_ids"].to(cfg.device, non_blocking=True)
        n_mask = batch["negative_attention_mask"].to(cfg.device, non_blocking=True)

        t0 = time.time()

        # ---------- 前向 ----------
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            a_emb = model(a_ids, a_mask)   # (B, D) 已 L2 归一化
            p_emb = model(p_ids, p_mask)   # (B, D)
            n_emb = model(n_ids, n_mask)   # (B, D)
            loss = criterion(a_emb, p_emb, n_emb)

        # ---------- 反向 ----------
        if cfg.use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm
        )

        if cfg.use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step()

        step_time = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        # ---------- 指标 ----------
        loss_val = loss.item()
        mfu = compute_mfu(
            num_params, a_ids.size(0), a_ids.size(1),
            step_time, peak_flops,
        )

        # in-batch 准确率（对角是否最大）
        with torch.no_grad():
            sim = a_emb @ p_emb.T
            diag = sim.diag().unsqueeze(1)
            ranks = (sim >= diag).sum(dim=1).float()
            in_batch_acc = (ranks == 1).float().mean().item()

            # hard negative 的分离度（辅助观察）
            pos_sim = (a_emb * p_emb).sum(dim=-1).mean().item()
            neg_sim = (a_emb * n_emb).sum(dim=-1).mean().item()

        total_loss += loss_val
        total_steps += 1

        pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "ib": f"{in_batch_acc:.3f}",
            "pos": f"{pos_sim:.3f}",
            "neg": f"{neg_sim:.3f}",
            "gn": f"{grad_norm:.2f}",
            "mfu": f"{mfu*100:.1f}%",
        })

        if global_step % 20 == 0:
            writer.add_scalar("Train/loss", loss_val, global_step)
            writer.add_scalar("Train/in_batch_acc", in_batch_acc, global_step)
            writer.add_scalar("Train/pos_sim", pos_sim, global_step)
            writer.add_scalar("Train/neg_sim", neg_sim, global_step)
            writer.add_scalar("Train/grad_norm", grad_norm, global_step)
            writer.add_scalar("Train/lr", lr, global_step)
            writer.add_scalar("Train/mfu", mfu, global_step)
            writer.add_scalar("Train/step_time_ms", step_time * 1000, global_step)

        global_step += 1

    avg_loss = total_loss / max(1, total_steps)
    return avg_loss, global_step


# ==================================================================
# 编码
# ==================================================================
@torch.no_grad()
def encode_corpus(model, sentences, tokenizer, cfg, batch_size=256):
    """把一批句子编码成 L2 归一化的向量，返回 (N, D)。"""
    model.eval()
    all_emb = []
    for i in range(0, len(sentences), batch_size):
        batch_texts = sentences[i:i + batch_size]
        enc = tokenizer(
            batch_texts, truncation=True, max_length=cfg.max_seq_len,
            padding="max_length", return_tensors="pt",
        )
        ids = enc["input_ids"].to(cfg.device)
        mask = enc["attention_mask"].to(cfg.device)
        tti = enc.get("token_type_ids")
        if tti is not None:
            tti = tti.to(cfg.device)
        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            emb = model(ids, mask, tti)
        all_emb.append(emb.float().cpu())
    return torch.cat(all_emb, dim=0)


# ==================================================================
# 评估：全库检索 Recall@K
# ==================================================================
@torch.no_grad()
def evaluate_recall(model, val_sentences1, val_sentences2, tokenizer, cfg):
    """
    全库检索评估:
        1. 所有 s1 编码成 anchor_embs
        2. 所有 s2 编码成 positive_embs
        3. 算 (N, N) 相似度矩阵
        4. 每行找 top-K，看对角线是否命中
    """
    print("  [Eval] 编码验证集...")
    a_emb = encode_corpus(model, val_sentences1, tokenizer, cfg)
    p_emb = encode_corpus(model, val_sentences2, tokenizer, cfg)

    sim = a_emb @ p_emb.T   # (N, N)
    N = sim.size(0)

    _, topk_idx = sim.topk(max(cfg.recall_k), dim=1)
    diag = torch.arange(N).unsqueeze(1)

    recalls = {}
    for k in cfg.recall_k:
        hit = (topk_idx[:, :k] == diag).any(dim=1).float().mean().item()
        recalls[k] = hit

    pos_sim = sim.diag().mean().item()
    avg_sim = sim.mean().item()

    return recalls, pos_sim, avg_sim


# ==================================================================
# 主流程
# ==================================================================
def main():
    set_seed(CFG.seed)
    device = CFG.device
    print(f"[Train] 设备: {device}")
    if device == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    # ---------- 数据 ----------
    train_loader, val_loader, test_loader, tokenizer = build_biencoder_triplet_loaders(CFG)

    # 验证集原始文本
    print(f"[Train] 加载验证集原始文本用于 Recall@K 评估")
    ds = load_lcqmc_raw(CFG)
    val_s1 = ds["validation"]["sentence1"]
    val_s2 = ds["validation"]["sentence2"]
    print(f"[Train] 验证集: {len(val_s1)} 对")

    # ---------- 模型 ----------
    model = BiEncoderModel(CFG.model_name, pooling=CFG.pooling).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] 模型参数量: {num_params:,}")

    # ---------- 优化器 ----------
    total_steps = len(train_loader) * CFG.epochs
    optimizer, scheduler, warmup_steps = build_optimizer_and_scheduler(
        model, CFG, total_steps
    )
    print(f"[Train] 总步数: {total_steps}, warmup: {warmup_steps}")

    # ---------- 损失函数 ----------
    # CombinedLoss = MNRL + alpha * TripletLoss
    criterion = CombinedLoss(
        temperature=CFG.temperature,
        margin=0.3,
        alpha=0.5,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.use_amp)

    peak_flops = get_gpu_peak_flops(CFG.use_amp)
    print(f"[Train] 峰值算力: {peak_flops / 1e12:.1f} TFLOPS")

    # ---------- 日志 ----------
    Path(CFG.log_dir).mkdir(exist_ok=True, parents=True)
    Path(CFG.save_dir).mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(f"{CFG.log_dir}/biencoder")

    # ---------- 训练 ----------
    global_step = 0
    best_recall1 = 0.0

    for epoch in range(1, CFG.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler,
            CFG, writer, global_step, epoch, peak_flops, num_params,
        )

        recalls, pos_sim, avg_sim = evaluate_recall(
            model, val_s1, val_s2, tokenizer, CFG
        )

        writer.add_scalar("Epoch/train_loss", train_loss, epoch)
        for k, v in recalls.items():
            writer.add_scalar(f"Epoch/recall@{k}", v, epoch)
        writer.add_scalar("Epoch/pos_sim", pos_sim, epoch)
        writer.add_scalar("Epoch/avg_sim", avg_sim, epoch)

        recall_str = "  ".join([f"R@{k}={v:.4f}" for k, v in recalls.items()])
        print(f"\n[Epoch {epoch}] train_loss={train_loss:.4f}")
        print(f"  {recall_str}")
        print(f"  正例平均相似度={pos_sim:.4f}  全体平均相似度={avg_sim:.4f}")

        r1 = recalls[1]
        if r1 > best_recall1:
            best_recall1 = r1
            save_path = os.path.join(CFG.save_dir, "biencoder_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "recall@1": r1,
                "recalls": recalls,
            }, save_path)
            print(f"  [Save] 最佳模型 R@1={r1:.4f} → {save_path}")

    # ---------- 测试集 ----------
    print("\n[Test] 加载最佳模型评估测试集")
    ckpt = torch.load(
        os.path.join(CFG.save_dir, "biencoder_best.pt"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    test_s1 = ds["test"]["sentence1"]
    test_s2 = ds["test"]["sentence2"]
    test_recalls, test_pos_sim, test_avg_sim = evaluate_recall(
        model, test_s1, test_s2, tokenizer, CFG
    )
    print(f"[Test] " + "  ".join([f"R@{k}={v:.4f}" for k, v in test_recalls.items()]))
    print(f"[Test] 正例平均相似度={test_pos_sim:.4f}  全体平均相似度={test_avg_sim:.4f}")

    for k, v in test_recalls.items():
        writer.add_scalar(f"Test/recall@{k}", v, 0)
    writer.close()

    print(f"\n[Train] 完成")
    print(f"    最佳 val R@1: {best_recall1:.4f}")
    print(f"    test R@1:     {test_recalls[1]:.4f}")


if __name__ == "__main__":
    main()
