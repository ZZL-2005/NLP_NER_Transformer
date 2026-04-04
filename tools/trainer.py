import torch
from tqdm import tqdm


def ids_to_tags(id_seq, id2tag):
    return [id2tag[idx] for idx in id_seq]


def extract_entities(tag_seq):
    """
    BIO 实体抽取
    输入:
        ["O", "B_LOC", "I_LOC", "O", "B_PER", "I_PER"]
    输出:
        {("LOC", 1, 2), ("PER", 4, 5)}
    """
    entities = set()
    start = None
    ent_type = None

    for i, tag in enumerate(tag_seq):
        if tag == "O":
            if start is not None:
                entities.add((ent_type, start, i - 1))
                start = None
                ent_type = None
            continue

        if "_" not in tag:
            if start is not None:
                entities.add((ent_type, start, i - 1))
                start = None
                ent_type = None
            continue

        prefix, cur_type = tag.split("_", 1)

        if prefix == "B":
            if start is not None:
                entities.add((ent_type, start, i - 1))
            start = i
            ent_type = cur_type

        elif prefix == "I":
            if start is not None and ent_type == cur_type:
                continue
            else:
                if start is not None:
                    entities.add((ent_type, start, i - 1))
                start = i
                ent_type = cur_type

        else:
            if start is not None:
                entities.add((ent_type, start, i - 1))
                start = None
                ent_type = None

    if start is not None:
        entities.add((ent_type, start, len(tag_seq) - 1))

    return entities


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}/{num_epochs}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        label_ids = batch["label_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask=attention_mask)  # [B, L, num_tags]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            label_ids.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, id2tag):
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    tp = 0
    fp = 0
    fn = 0

    pbar = tqdm(dataloader, desc="Evaluate", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        label_ids = batch["label_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            label_ids.reshape(-1),
        )
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)  # [B, L]

        # token-level accuracy
        mask = (label_ids != -100)
        correct_tokens += ((preds == label_ids) & mask).sum().item()
        total_tokens += mask.sum().item()

        # entity-level metrics
        preds = preds.cpu().tolist()
        label_ids = label_ids.cpu().tolist()

        for pred_seq, gold_seq in zip(preds, label_ids):
            valid_pred_ids = []
            valid_gold_ids = []

            for p, g in zip(pred_seq, gold_seq):
                if g == -100:
                    continue
                valid_pred_ids.append(p)
                valid_gold_ids.append(g)

            pred_tags = ids_to_tags(valid_pred_ids, id2tag)
            gold_tags = ids_to_tags(valid_gold_ids, id2tag)

            pred_entities = extract_entities(pred_tags)
            gold_entities = extract_entities(gold_tags)

            tp += len(pred_entities & gold_entities)
            fp += len(pred_entities - gold_entities)
            fn += len(gold_entities - pred_entities)

    avg_loss = total_loss / len(dataloader)
    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    entity_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    entity_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    entity_f1 = (
        2 * entity_precision * entity_recall / (entity_precision + entity_recall)
        if (entity_precision + entity_recall) > 0
        else 0.0
    )

    return {
        "loss": avg_loss,
        "token_acc": token_acc,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def save_checkpoint(path, model, optimizer, epoch, best_dev_metric, args, meta):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_dev_metric": best_dev_metric,
        "model_config": {
            "vocab_size": meta["vocab_size"],
            "num_tags": meta["num_tags"],
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "num_layers": args.num_layers,
            "max_len": args.max_len,
            "dropout": args.dropout,
            "pad_token_id": meta["pad_token_id"],
        },
        "train_config": vars(args),
    }
    torch.save(ckpt, path)
