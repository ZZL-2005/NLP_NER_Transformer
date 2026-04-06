import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Train NER model")

    # ---------- 路径参数 ----------
    parser.add_argument("--train_path", type=str, default="data/train.txt")
    parser.add_argument("--train_tag_path", type=str, default="data/train_TAG.txt")
    parser.add_argument("--dev_path", type=str, default="data/dev.txt")
    parser.add_argument("--dev_tag_path", type=str, default="data/dev_TAG.txt")

    parser.add_argument("--char2id_path", type=str, default="meta/char2id.json")
    parser.add_argument("--id2char_path", type=str, default="meta/id2char.json")
    parser.add_argument("--tag2id_path", type=str, default="meta/tag2id.json")
    parser.add_argument("--id2tag_path", type=str, default="meta/id2tag.json")

    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--exp_name", type=str, default="baseline")

    # ---------- 训练参数 ----------
    parser.add_argument("--model_name", type=str, default="transformer")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # ---------- 模型参数 ----------
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=5000)
    parser.add_argument("--dropout", type=float, default=0.1)

    return parser.parse_args()

def print_config(args, meta, device):
    print("=" * 60)
    print("Train Config")
    print("-" * 60)
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("-" * 60)
    print(f"device: {device}")
    print(f"vocab_size: {meta['vocab_size']}")
    print(f"num_tags: {meta['num_tags']}")
    print(f"pad_token_id: {meta['pad_token_id']}")
    print("=" * 60)

