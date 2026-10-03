"""
config.py
项目 2 全局配置。
"""

from dataclasses import dataclass


@dataclass
class Config:
    # ==================== 数据 ====================
    dataset_name: str = "mteb/LCQMC"
    max_seq_len: int = 48            # 中文问句一般 10-30 字，64 足够
    # 注：如果 64 不够，可以改成 96，但对比学习对长序列不敏感

    # ==================== 模型 ====================
    model_name: str = "hfl/rbt3"     # 3 层 RoBERTa，约 38M 参数
    # 备选：
    #   "hfl/chinese-roberta-wwm-ext"  更大更强，12 层，102M
    #   "BAAI/bge-small-zh-v1.5"       专门为句向量设计，24M

    embedding_dim: int = 768         # 从预训练模型拿的 hidden size
    # rbt3: 768
    # bge-small-zh: 512

    pooling: str = "mean"            # 句向量池化方式: mean / cls / max

    # ==================== 训练 ====================
    batch_size: int = 128             # 对比学习喜欢大 batch
    # 4090 25GB 显存，rbt3 训练 batch=128 也够
    # 但先跑 64 稳妥

    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06        # 前 10% 步数做 warmup
    epochs: int = 4

    # 对比学习专用
    temperature: float = 0.1        # 温度系数 τ，越小分布越尖锐
    use_amp: bool = True             # 混合精度训练（4090 加速）
    grad_accum_steps: int = 1        # 梯度累积
    max_grad_norm: float = 1.0

    # ==================== 评估 ====================
    eval_batch_size: int = 128
    recall_k: list = None            # 初始化时改

    # ==================== 路径 ====================
    save_dir: str = "checkpoints"
    log_dir: str = "runs"
    figures_dir: str = "figures"

    # ==================== 设备 ====================
    device: str = "cuda"
    seed: int = 42

    def __post_init__(self):
        if self.recall_k is None:
            self.recall_k = [1, 5, 10]

        import torch
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("[Config] CUDA 不可用，切换到 CPU")

        # 根据模型调整 embedding_dim
        if "rbt3" in self.model_name:
            self.embedding_dim = 768
        elif "bge-small" in self.model_name:
            self.embedding_dim = 512
        elif "roberta" in self.model_name:
            self.embedding_dim = 768


CFG = Config()


if __name__ == "__main__":
    import torch
    print(CFG)
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
