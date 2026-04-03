import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.TransformerNER import TransformerNER
from dataset.dataset import NERDataset, collate_fn


# =========================
# 1. 路径与超参数
# =========================
train_path = "data/train.txt"
train_tag_path = "data/train_TAG.txt"
dev_path = "data/dev.txt"
dev_tag_path = "data/dev_TAG.txt"

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

batch_size = 32
lr = 1e-3
num_epochs = 50

d_model = 256
n_heads = 4
d_ff = 512
num_layers = 8
max_len = 5000
dropout = 0.1


# =========================
# 2. 工具函数
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(model, dataloader, criterion, device):
    """
    在 dev 集上评估：
    1) 平均 loss
    2) token-level accuracy（忽略 -100）
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)

            logits = model(input_ids)   # [B, L, num_tags]

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                label_ids.reshape(-1)
            )
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)  # [B, L]

            mask = (label_ids != -100)
            correct_tokens += ((preds == label_ids) & mask).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, acc


# =========================
# 3. 读取映射表
# =========================
char2id = load_json("meta/char2id.json")
id2char = load_json("meta/id2char.json")
tag2id = load_json("meta/tag2id.json")
id2tag = load_json("meta/id2tag.json")

id2char = {int(k): v for k, v in id2char.items()}
id2tag = {int(k): v for k, v in id2tag.items()}

vocab_size = max(char2id.values()) + 1
num_tags = max(tag2id.values()) + 1
pad_token_id = char2id["<PAD>"]


# =========================
# 4. 数据集与 DataLoader
# =========================
train_dataset = NERDataset(train_path, train_tag_path, char2id, tag2id)
dev_dataset = NERDataset(dev_path, dev_tag_path, char2id, tag2id)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_fn
)


# =========================
# 5. 模型、损失、优化器
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerNER(
    vocab_size=vocab_size,
    num_tags=num_tags,
    d_model=d_model,
    n_heads=n_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    max_len=max_len,
    dropout=dropout,
    pad_token_id=pad_token_id,
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# =========================
# 6. 训练主循环
# =========================
best_dev_acc = 0.0
best_epoch = 0

history = {
    "train_loss": [],
    "dev_loss": [],
    "dev_acc": []
}

for epoch in range(1, num_epochs + 1):
    model.train()
    total_train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        label_ids = batch["label_ids"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            label_ids.reshape(-1)
        )

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device)

    history["train_loss"].append(avg_train_loss)
    history["dev_loss"].append(dev_loss)
    history["dev_acc"].append(dev_acc)

    print(
        f"[Epoch {epoch}] "
        f"train_loss={avg_train_loss:.4f} "
        f"dev_loss={dev_loss:.4f} "
        f"dev_acc={dev_acc:.4f}"
    )

    # 只保存最优模型
    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        best_epoch = epoch

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dev_acc": dev_acc,
                "config": {
                    "vocab_size": vocab_size,
                    "num_tags": num_tags,
                    "d_model": d_model,
                    "n_heads": n_heads,
                    "d_ff": d_ff,
                    "num_layers": num_layers,
                    "max_len": max_len,
                    "dropout": dropout,
                    "pad_token_id": pad_token_id,
                }
            },
            os.path.join(save_dir, "best_model.pt")
        )
        print(
            f"New best model saved! "
            f"best_epoch = {best_epoch}, best_dev_acc = {best_dev_acc:.4f}"
        )


# =========================
# 7. 保存训练历史
# =========================
with open(os.path.join(save_dir, "history.json"), "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print("Training finished.")
print(f"Best epoch = {best_epoch}")
print(f"Best dev acc = {best_dev_acc:.4f}")