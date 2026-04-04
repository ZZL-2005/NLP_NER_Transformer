import os
import json
import torch

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_meta(args):
    char2id = load_json(args.char2id_path)
    id2char = load_json(args.id2char_path)
    tag2id = load_json(args.tag2id_path)
    id2tag = load_json(args.id2tag_path)

    # json 里的 key 可能是字符串，需要转回 int
    id2char = {int(k): v for k, v in id2char.items()}
    id2tag = {int(k): v for k, v in id2tag.items()}

    vocab_size = max(char2id.values()) + 1
    num_tags = max(tag2id.values()) + 1
    pad_token_id = char2id["<PAD>"]

    meta = {
        "char2id": char2id,
        "id2char": id2char,
        "tag2id": tag2id,
        "id2tag": id2tag,
        "vocab_size": vocab_size,
        "num_tags": num_tags,
        "pad_token_id": pad_token_id,
    }
    return meta
