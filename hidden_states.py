import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.TransformerNER import TransformerNER
from dataset.dataset import NERDataset, collate_fn


# =========================
# 1. 路径配置
# =========================
train_path = "data/train.txt"
train_tag_path = "data/train_TAG.txt"
dev_path = "data/dev.txt"
dev_tag_path = "data/dev_TAG.txt"
model_path = "checkpoints/best_model.pt"


# =========================
# 2. 工具函数
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_hidden_and_delta_stats(model, dataloader, device, max_batches=None, desc="Collecting"):
    """
    收集：
    1. 每层 hidden state 的平均幅值
    2. 每层增量 delta = h_l - h_{l-1} 的平均幅值

    返回：
    {
        "hidden_l2_mean": [...],
        "hidden_rms_mean": [...],
        "delta_l2_mean": [...],
        "delta_rms_mean": [...],
    }
    """
    model.eval()

    hidden_l2_sum = None
    hidden_rms_sum = None
    hidden_count = None

    delta_l2_sum = None
    delta_rms_sum = None
    delta_count = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc)):
            if max_batches is not None and batch_idx >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)

            # 要求你的模型支持 return_hidden_states=True
            logits, hidden_states = model(input_ids, return_hidden_states=True)

            # mask: [B, L]，只统计有效 token
            mask = (label_ids != -100)

            num_hidden = len(hidden_states)   # = num_layers + 1
            num_delta = num_hidden - 1

            if hidden_l2_sum is None:
                hidden_l2_sum = [0.0] * num_hidden
                hidden_rms_sum = [0.0] * num_hidden
                hidden_count = [0.0] * num_hidden

                delta_l2_sum = [0.0] * num_delta
                delta_rms_sum = [0.0] * num_delta
                delta_count = [0.0] * num_delta

            valid_count = mask.sum().item()

            # 统计 hidden state 幅值
            for i, h in enumerate(hidden_states):
                # h: [B, L, D]
                l2 = torch.norm(h, dim=-1)  # [B, L]
                rms = torch.sqrt(torch.mean(h ** 2, dim=-1) + 1e-12)  # [B, L]

                hidden_l2_sum[i] += l2[mask].sum().item()
                hidden_rms_sum[i] += rms[mask].sum().item()
                hidden_count[i] += valid_count

            # 统计 delta 幅值
            for i in range(num_delta):
                delta = hidden_states[i + 1] - hidden_states[i]  # [B, L, D]
                l2 = torch.norm(delta, dim=-1)  # [B, L]
                rms = torch.sqrt(torch.mean(delta ** 2, dim=-1) + 1e-12)  # [B, L]

                delta_l2_sum[i] += l2[mask].sum().item()
                delta_rms_sum[i] += rms[mask].sum().item()
                delta_count[i] += valid_count

    hidden_l2_mean = [s / c for s, c in zip(hidden_l2_sum, hidden_count)]
    hidden_rms_mean = [s / c for s, c in zip(hidden_rms_sum, hidden_count)]

    delta_l2_mean = [s / c for s, c in zip(delta_l2_sum, delta_count)]
    delta_rms_mean = [s / c for s, c in zip(delta_rms_sum, delta_count)]

    return {
        "hidden_l2_mean": hidden_l2_mean,
        "hidden_rms_mean": hidden_rms_mean,
        "delta_l2_mean": delta_l2_mean,
        "delta_rms_mean": delta_rms_mean,
    }


def plot_hidden_and_delta_stats(stats, save_prefix="dev_hidden_delta"):
    hidden_x = list(range(len(stats["hidden_l2_mean"])))
    delta_x = list(range(1, len(stats["delta_l2_mean"]) + 1))

    # 图1：hidden state 幅值
    plt.figure(figsize=(8, 5))
    plt.plot(hidden_x, stats["hidden_l2_mean"], marker="o", label="Hidden Mean L2")
    plt.plot(hidden_x, stats["hidden_rms_mean"], marker="s", label="Hidden Mean RMS")
    plt.xlabel("State Index")
    plt.ylabel("Magnitude")
    plt.title("Hidden State Magnitude by Depth")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_hidden.png", dpi=200)
    plt.show()

    # 图2：delta 幅值
    plt.figure(figsize=(8, 5))
    plt.plot(delta_x, stats["delta_l2_mean"], marker="o", label="Delta Mean L2")
    plt.plot(delta_x, stats["delta_rms_mean"], marker="s", label="Delta Mean RMS")
    plt.xlabel("Layer Index")
    plt.ylabel("Magnitude")
    plt.title("Residual Increment Magnitude by Layer")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_delta.png", dpi=200)
    plt.show()


# =========================
# 3. 读取映射表
# =========================
char2id = load_json("meta/char2id.json")
id2char = load_json("meta/id2char.json")
tag2id = load_json("meta/tag2id.json")
id2tag = load_json("meta/id2tag.json")

id2char = {int(k): v for k, v in id2char.items()}
id2tag = {int(k): v for k, v in id2tag.items()}


# =========================
# 4. 数据集
# =========================
dev_dataset = NERDataset(dev_path, dev_tag_path, char2id, tag2id)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)


# =========================
# 5. 加载模型
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device)

checkpoint = torch.load(model_path, map_location=device)
config = checkpoint["config"]

model = TransformerNER(
    vocab_size=config["vocab_size"],
    num_tags=config["num_tags"],
    d_model=config["d_model"],
    n_heads=config["n_heads"],
    d_ff=config["d_ff"],
    num_layers=config["num_layers"],
    max_len=config["max_len"],
    dropout=config["dropout"],
    pad_token_id=config["pad_token_id"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# =========================
# 6. 收集统计并可视化
# =========================
stats = collect_hidden_and_delta_stats(
    model=model,
    dataloader=dev_loader,
    device=device,
    max_batches=None,   # 想先快速试验可以改成 50
    desc="Collect Hidden/Delta Stats on Dev"
)

print("\n===== Hidden State Magnitude (Dev) =====")
for i, (l2v, rmsv) in enumerate(zip(stats["hidden_l2_mean"], stats["hidden_rms_mean"])):
    print(f"state_{i:02d}  hidden_l2={l2v:.6f}  hidden_rms={rmsv:.6f}")

print("\n===== Residual Increment Magnitude (Dev) =====")
for i, (l2v, rmsv) in enumerate(zip(stats["delta_l2_mean"], stats["delta_rms_mean"]), start=1):
    print(f"layer_{i:02d}  delta_l2={l2v:.6f}  delta_rms={rmsv:.6f}")

plot_hidden_and_delta_stats(stats, save_prefix="dev_hidden_delta")