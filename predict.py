"""
对 test.txt 进行 NER 推理，输出预测标签到 txt 文件。
用法：
    python predict.py
    python predict.py --model_name transformer --output_path results/test_pred_TAG.txt
"""
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tools.loader import load_json
from tools.builder import build_model


# ─────────────────── 配置 ───────────────────
CKPT_PATH = "checkpoints/transformer/best_model.pt"
TEST_PATH = "data/test.txt"
OUTPUT_PATH = "results/test_pred_TAG.txt"

META_DIR = "meta"
CHAR2ID_PATH = os.path.join(META_DIR, "char2id.json")
ID2CHAR_PATH = os.path.join(META_DIR, "id2char.json")
TAG2ID_PATH  = os.path.join(META_DIR, "tag2id.json")
ID2TAG_PATH  = os.path.join(META_DIR, "id2tag.json")


# ─────────────── 仅文本的 Dataset ───────────────
class TestNERDataset(Dataset):
    """只读取文本（无标签），用于推理。"""
    def __init__(self, data_path, char2id):
        self.samples = []
        self.raw_chars = []          # 保留原始字符，方便后续输出
        self.char2id = char2id

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                chars = line.strip().split()
                if not chars:
                    continue
                input_ids = [char2id.get(ch, char2id["<UNK>"]) for ch in chars]
                self.samples.append({
                    "input_ids": input_ids,
                    "length": len(input_ids),
                })
                self.raw_chars.append(chars)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "length": sample["length"],
        }


def test_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    lengths   = [item["length"]    for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (input_ids != 0).long()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


# ─────────────── 加载模型 ───────────────
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt["model_config"]

    # 构造一个简易 args 对象，传给 build_model
    class Args:
        pass
    args = Args()
    for k, v in cfg.items():
        setattr(args, k, v)
    args.model_name = ckpt.get("train_config", {}).get("model_name", "transformer")

    model = build_model(
        args,
        vocab_size=cfg["vocab_size"],
        num_tags=cfg["num_tags"],
        pad_token_id=cfg["pad_token_id"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ─────────────── 推理 ───────────────
@torch.no_grad()
def predict(model, dataloader, id2tag, device):
    """返回每条样本的预测标签列表 (list[list[str]])。"""
    all_pred_tags = []
    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths        = batch["lengths"].tolist()

        logits = model(input_ids, attention_mask=attention_mask)   # [B, L, num_tags]
        preds  = torch.argmax(logits, dim=-1).cpu().tolist()       # [B, L]

        for pred_seq, length in zip(preds, lengths):
            tags = [id2tag[p] for p in pred_seq[:length]]
            all_pred_tags.append(tags)

    return all_pred_tags


# ─────────────── 写出结果 ───────────────
def save_predictions(output_path, all_pred_tags):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for tags in all_pred_tags:
            f.write(" ".join(tags) + "\n")
    print(f"✅ 预测标签已保存到: {output_path}")
    print(f"   共 {len(all_pred_tags)} 条样本")


# ─────────────── main ───────────────
def main():
    parser = argparse.ArgumentParser(description="NER Predict on test.txt")
    parser.add_argument("--ckpt_path",   type=str, default=CKPT_PATH)
    parser.add_argument("--test_path",   type=str, default=TEST_PATH)
    parser.add_argument("--output_path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--batch_size",  type=int, default=128)
    parser.add_argument("--device",      type=str, default="cuda")
    cli_args = parser.parse_args()

    device = torch.device(
        "cuda" if cli_args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    print(f"🔧 Device: {device}")

    # 1. 加载 meta
    char2id = load_json(CHAR2ID_PATH)
    id2tag  = load_json(ID2TAG_PATH)
    id2tag  = {int(k): v for k, v in id2tag.items()}

    # 2. 构建测试集
    test_dataset = TestNERDataset(cli_args.test_path, char2id)
    test_loader  = DataLoader(
        test_dataset,
        batch_size=cli_args.batch_size,
        shuffle=False,
        collate_fn=test_collate_fn,
    )
    print(f"📄 测试集: {cli_args.test_path}  ({len(test_dataset)} 条)")

    # 3. 加载模型
    model = load_model(cli_args.ckpt_path, device)
    print(f"📦 Checkpoint: {cli_args.ckpt_path}")

    # 4. 推理
    all_pred_tags = predict(model, test_loader, id2tag, device)

    # 5. 保存
    save_predictions(cli_args.output_path, all_pred_tags)


if __name__ == "__main__":
    main()