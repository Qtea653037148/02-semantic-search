"""
train_crossencoder.py
训练 CrossEncoder baseline。

监控指标（和项目 1 一致）:
    - Loss        交叉熵损失
    - PPL         exp(loss)
    - GradNorm    梯度 L2 范数
    - MFU         模型算力利用率
    - LR          实际学习率
    - Val Acc     验证集准确率

新增:
    - AMP 混合精度训练（4090 加速）
    - 更精确的 MFU 计算（含 AMP 的 FP16 峰值算力）
"""

import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import CFG
from data import build_crossencoder_loaders
from model import CrossEncoderModel


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
    """
    返回 GPU 峰值算力（FLOPS）。
    RTX 4090:
        FP32:       82.6 TFLOPS
        FP16 TC:    165 TFLOPS（密集）
    TFLOPS 是 10^12，这里换算成 FLOPS。
    """
    if not torch.cuda.is_available():
        return 0.0
    name = torch.cuda.get_device_name(0)
    if "4090" in name:
        return 165e12 if use_amp else 82.6e12
    if "3090" in name:
        return 71e12 if use_amp else 35.6e12
    if "A100" in name:
        return 312e12 if use_amp else 156e12
    # 未知型号，给一个保守估计
    return 50e12


def build_optimizer_and_scheduler(model, cfg, total_steps):
    """
    AdamW + warmup + cosine 衰减。
    注意：CrossEncoder 通常用较小的学习率（1e-5 ~ 5e-5），因为是在预训练模型上微调。
    """
    # 权重衰减不作用于 LayerNorm 和 bias（标准做法）
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.lr)

    warmup_steps = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler, warmup_steps


def compute_mfu(num_params, batch_size, seq_len, step_time, peak_flops, use_amp):
    """
    MFU = 实际 FLOPs / 峰值 FLOPs。
    前向 + 反向约 6 * N_params * tokens 次浮点运算。
    AMP 下算力峰值更高，但计算量本身不变。
    """
    if peak_flops <= 0 or step_time <= 0:
        return 0.0
    tokens = batch_size * seq_len
    flops = 6 * num_params * tokens
    return flops / (step_time * peak_flops)


# ==================================================================
# 训练一个 epoch
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
        input_ids = batch["input_ids"].to(cfg.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(cfg.device, non_blocking=True)
        labels = batch["label"].to(cfg.device, non_blocking=True)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(cfg.device, non_blocking=True)

        t0 = time.time()

        # ---------- 前向（AMP） ----------
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)

        # ---------- 反向（AMP 用 scaler 缩放 loss） ----------
        if cfg.use_amp:
            scaler.scale(loss).backward()
            # 反缩放梯度，才能拿到真实的 grad_norm
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # ---------- 梯度裁剪（此时梯度已是真实值） ----------
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.max_grad_norm
        )

        # ---------- 更新（AMP 用 scaler 处理跳步） ----------
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
        ppl = math.exp(min(loss_val, 20))   # 防止 exp 溢出
        mfu = compute_mfu(
            num_params, input_ids.size(0), input_ids.size(1),
            step_time, peak_flops, cfg.use_amp,
        )

        total_loss += loss_val
        total_steps += 1

        pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "gn": f"{grad_norm:.3f}",
            "lr": f"{lr:.2e}",
            "mfu": f"{mfu*100:.1f}%",
        })

        if global_step % 20 == 0:
            writer.add_scalar("Train/loss", loss_val, global_step)
            writer.add_scalar("Train/ppl", ppl, global_step)
            writer.add_scalar("Train/grad_norm", grad_norm, global_step)
            writer.add_scalar("Train/mfu", mfu, global_step)
            writer.add_scalar("Train/lr", lr, global_step)
            writer.add_scalar("Train/step_time_ms", step_time * 1000, global_step)

        global_step += 1

    avg_loss = total_loss / max(1, total_steps)
    return avg_loss, global_step


# ==================================================================
# 评估
# ==================================================================
@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    correct = 0
    total = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(cfg.device)
        attention_mask = batch["attention_mask"].to(cfg.device)
        labels = batch["label"].to(cfg.device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(cfg.device)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        total_steps += 1

    return total_loss / max(1, total_steps), correct / max(1, total)


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
    train_loader, val_loader, test_loader, tokenizer = build_crossencoder_loaders(CFG)

    # ---------- 模型 ----------
    model = CrossEncoderModel(CFG.model_name, num_labels=2).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] 模型参数量: {num_params:,}")

    # ---------- 优化器 / 调度器 / 损失 ----------
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * CFG.epochs
    optimizer, scheduler, warmup_steps = build_optimizer_and_scheduler(
        model, CFG, total_steps
    )
    print(f"[Train] 总步数: {total_steps}, warmup: {warmup_steps}")

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.use_amp)

    # ---------- MFU 用的峰值算力 ----------
    peak_flops = get_gpu_peak_flops(CFG.use_amp)
    print(f"[Train] 峰值算力: {peak_flops / 1e12:.1f} TFLOPS "
          f"({'FP16' if CFG.use_amp else 'FP32'})")

    # ---------- 日志 ----------
    Path(CFG.log_dir).mkdir(exist_ok=True, parents=True)
    Path(CFG.save_dir).mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(f"{CFG.log_dir}/crossencoder")

    # ---------- 训练 ----------
    global_step = 0
    best_val_acc = 0.0

    for epoch in range(1, CFG.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler,
            CFG, writer, global_step, epoch, peak_flops, num_params,
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, CFG)

        writer.add_scalar("Epoch/train_loss", train_loss, epoch)
        writer.add_scalar("Epoch/val_loss", val_loss, epoch)
        writer.add_scalar("Epoch/val_acc", val_acc, epoch)

        print(f"\n[Epoch {epoch}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(CFG.save_dir, "crossencoder_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
            }, save_path)
            print(f"[Save] 最佳模型 val_acc={val_acc:.4f} → {save_path}")

    # ---------- 测试集 ----------
    print("\n[Test] 加载最佳模型评估测试集")
    ckpt = torch.load(
        os.path.join(CFG.save_dir, "crossencoder_best.pt"),
        map_location=device, weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, CFG)
    print(f"[Test] loss={test_loss:.4f}  acc={test_acc:.4f}")

    writer.add_scalar("Test/loss", test_loss, 0)
    writer.add_scalar("Test/acc", test_acc, 0)
    writer.close()

    print(f"\n[Train] 完成")
    print(f"    最佳 val_acc: {best_val_acc:.4f}")
    print(f"    test_acc:     {test_acc:.4f}")


if __name__ == "__main__":
    main()
