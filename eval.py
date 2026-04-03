import json
import torch
import torch.nn as nn
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


def extract_entities(tag_ids, id2tag):
    """
    将 BIO 标签序列转换为实体集合
    返回:
        set((start, end, ent_type))
    其中 start/end 都是闭区间
    """
    entities = set()
    start = None
    ent_type = None

    for i, tag_id in enumerate(tag_ids):
        tag = id2tag[int(tag_id)]

        if tag == "O":
            if start is not None:
                entities.add((start, i - 1, ent_type))
                start = None
                ent_type = None
            continue

        if "_" not in tag:
            if start is not None:
                entities.add((start, i - 1, ent_type))
                start = None
                ent_type = None
            continue

        prefix, cur_type = tag.split("_", 1)

        if prefix == "B":
            if start is not None:
                entities.add((start, i - 1, ent_type))
            start = i
            ent_type = cur_type

        elif prefix == "I":
            if start is not None and ent_type == cur_type:
                pass
            else:
                if start is not None:
                    entities.add((start, i - 1, ent_type))
                start = i
                ent_type = cur_type

        else:
            if start is not None:
                entities.add((start, i - 1, ent_type))
                start = None
                ent_type = None

    if start is not None:
        entities.add((start, len(tag_ids) - 1, ent_type))

    return entities


def compute_entity_metrics(pred_batch, gold_batch, id2tag):
    tp = 0
    fp = 0
    fn = 0

    for pred_ids, gold_ids in zip(pred_batch, gold_batch):
        pred_entities = extract_entities(pred_ids, id2tag)
        gold_entities = extract_entities(gold_ids, id2tag)

        tp += len(pred_entities & gold_entities)
        fp += len(pred_entities - gold_entities)
        fn += len(gold_entities - pred_entities)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate(model, dataloader, criterion, device, id2tag, desc="Evaluating"):
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    all_pred_tags = []
    all_gold_tags = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            label_ids = batch["label_ids"].to(device)

            logits = model(input_ids)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                label_ids.reshape(-1)
            )
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)

            mask = (label_ids != -100)
            correct_tokens += ((preds == label_ids) & mask).sum().item()
            total_tokens += mask.sum().item()

            preds_cpu = preds.cpu().tolist()
            labels_cpu = label_ids.cpu().tolist()

            for pred_seq, gold_seq in zip(preds_cpu, labels_cpu):
                valid_pred = []
                valid_gold = []
                for p, g in zip(pred_seq, gold_seq):
                    if g != -100:
                        valid_pred.append(p)
                        valid_gold.append(g)

                all_pred_tags.append(valid_pred)
                all_gold_tags.append(valid_gold)

    avg_loss = total_loss / len(dataloader)
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    entity_metrics = compute_entity_metrics(all_pred_tags, all_gold_tags, id2tag)

    return {
        "loss": avg_loss,
        "token_acc": token_acc,
        "entity_precision": entity_metrics["precision"],
        "entity_recall": entity_metrics["recall"],
        "entity_f1": entity_metrics["f1"],
        "tp": entity_metrics["tp"],
        "fp": entity_metrics["fp"],
        "fn": entity_metrics["fn"],
    }


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
train_dataset = NERDataset(train_path, train_tag_path, char2id, tag2id)
dev_dataset = NERDataset(dev_path, dev_tag_path, char2id, tag2id)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

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

criterion = nn.CrossEntropyLoss(ignore_index=-100)


# =========================
# 6. 开始评估
# =========================
train_metrics = evaluate(model, train_loader, criterion, device, id2tag, desc="Evaluating Train")
dev_metrics = evaluate(model, dev_loader, criterion, device, id2tag, desc="Evaluating Dev")

print("\n===== Train Result =====")
print(f"train_loss         = {train_metrics['loss']:.6f}")
print(f"train_token_acc    = {train_metrics['token_acc']:.6f}")
print(f"train_entity_p     = {train_metrics['entity_precision']:.6f}")
print(f"train_entity_r     = {train_metrics['entity_recall']:.6f}")
print(f"train_entity_f1    = {train_metrics['entity_f1']:.6f}")
print(f"TP = {train_metrics['tp']}, FP = {train_metrics['fp']}, FN = {train_metrics['fn']}")

print("\n===== Dev Result =====")
print(f"dev_loss           = {dev_metrics['loss']:.6f}")
print(f"dev_token_acc      = {dev_metrics['token_acc']:.6f}")
print(f"dev_entity_p       = {dev_metrics['entity_precision']:.6f}")
print(f"dev_entity_r       = {dev_metrics['entity_recall']:.6f}")
print(f"dev_entity_f1      = {dev_metrics['entity_f1']:.6f}")
print(f"TP = {dev_metrics['tp']}, FP = {dev_metrics['fp']}, FN = {dev_metrics['fn']}")