import json
import torch
from torch.utils.data import DataLoader

from models.TransformerNER import TransformerNER
from dataset.dataset import NERDataset, collate_fn


train_path = "data/train.txt"
train_tag_path = "data/train_TAG.txt"
dev_path = "data/dev.txt"
dev_tag_path = "data/dev_TAG.txt"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


char2id = load_json("meta/char2id.json")
id2char = load_json("meta/id2char.json")
tag2id = load_json("meta/tag2id.json")
id2tag = load_json("meta/id2tag.json")

id2char = {int(k): v for k, v in id2char.items()}
id2tag = {int(k): v for k, v in id2tag.items()}

train_dataset = NERDataset(train_path, train_tag_path, char2id, tag2id)
dev_dataset = NERDataset(dev_path, dev_tag_path, char2id, tag2id)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

vocab_size = max(char2id.values()) + 1
num_tags = max(tag2id.values()) + 1
pad_token_id = char2id["<PAD>"]

model = TransformerNER(
    vocab_size=vocab_size,
    num_tags=num_tags,
    d_model=256,
    n_heads=4,
    d_ff=512,
    num_layers=4,
    max_len=512,
    dropout=0.1,
    pad_token_id=pad_token_id,
)

batch = next(iter(train_loader))
input_ids = batch["input_ids"]
label_ids = batch["label_ids"]
logits = model(input_ids)
import torch.nn as nn

criterion = nn.CrossEntropyLoss(ignore_index=-100)

loss = criterion(
    logits.view(-1, logits.size(-1)),   # [B*L, num_tags]
    label_ids.view(-1)                  # [B*L]
)

print("loss =", loss.item())

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
batch = next(iter(train_loader))

input_ids = batch["input_ids"].to(device)
label_ids = batch["label_ids"].to(device)

optimizer.zero_grad()

logits = model(input_ids)
loss = criterion(
    logits.view(-1, logits.size(-1)),
    label_ids.view(-1)
)

loss.backward()
optimizer.step()

print("train loss =", loss.item())