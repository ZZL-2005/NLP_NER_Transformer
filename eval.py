"""
在 dev 集上评估模型，输出各实体类型的细粒度 P/R/F1。
用法：
    python eval.py --model_name transformer
    python eval.py --model_name attnres_transformer
    python eval.py --model_name attnres_transformer_my
"""
import torch
from collections import defaultdict

from tools.configer import parse_args
from tools.loader import load_meta
from tools.builder import build_dataloaders, build_model
from tools.trainer import extract_entities


CKPT_PATHS = {
    "transformer":         "checkpoints/transformer/best_model.pt",
    "attnres_transformer": "checkpoints/attnres_transformer/best_model.pt",
    "attnres_transformer_my": "checkpoints/attnres_transformer_my/best_model.pt",
}


def load_model(args, meta):
    ckpt = torch.load(CKPT_PATHS[args.model_name], map_location="cpu")
    cfg  = ckpt["model_config"]
    # 用 checkpoint 里保存的模型配置，而不是 args（避免层数不一致）
    for k, v in cfg.items():
        setattr(args, k, v)
    model = build_model(
        args,
        vocab_size=cfg["vocab_size"],
        num_tags=cfg["num_tags"],
        pad_token_id=cfg["pad_token_id"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model


@torch.no_grad()
def evaluate_detail(model, dataloader, id2tag, device):
    model.eval()

    # per-type: tp / fp / fn
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for batch in dataloader:
        input_ids      = batch["input_ids"].to(device)
        label_ids      = batch["label_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)
        preds  = torch.argmax(logits, dim=-1).cpu().tolist()
        labels = label_ids.cpu().tolist()

        for pred_seq, gold_seq in zip(preds, labels):
            valid_pred, valid_gold = [], []
            for p, g in zip(pred_seq, gold_seq):
                if g == -100:
                    continue
                valid_pred.append(p)
                valid_gold.append(g)

            pred_tags = [id2tag[i] for i in valid_pred]
            gold_tags = [id2tag[i] for i in valid_gold]

            pred_ents = extract_entities(pred_tags)
            gold_ents = extract_entities(gold_tags)

            for ent in pred_ents & gold_ents:
                tp[ent[0]] += 1
            for ent in pred_ents - gold_ents:
                fp[ent[0]] += 1
            for ent in gold_ents - pred_ents:
                fn[ent[0]] += 1

    # 汇总
    all_types = sorted(set(list(tp) + list(fp) + list(fn)))
    rows = []
    total_tp = total_fp = total_fn = 0

    for t in all_types:
        p  = tp[t] / (tp[t] + fp[t]) if (tp[t] + fp[t]) > 0 else 0.0
        r  = tp[t] / (tp[t] + fn[t]) if (tp[t] + fn[t]) > 0 else 0.0
        f1 = 2 * p * r / (p + r)     if (p + r) > 0          else 0.0
        rows.append((t, tp[t], fp[t], fn[t], p, r, f1))
        total_tp += tp[t]
        total_fp += fp[t]
        total_fn += fn[t]

    # micro overall
    p_all  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r_all  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_all = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0.0
    rows.append(("ALL (micro)", total_tp, total_fp, total_fn, p_all, r_all, f1_all))

    return rows


def print_table(rows, model_name):
    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")
    print(f"{'Type':<14} {'TP':>6} {'FP':>6} {'FN':>6}  {'P':>7} {'R':>7} {'F1':>7}")
    print(f"{'-'*60}")
    for t, tp, fp, fn, p, r, f1 in rows:
        marker = "  ←" if t == "ALL (micro)" else ""
        print(f"{t:<14} {tp:>6} {fp:>6} {fn:>6}  {p:>7.4f} {r:>7.4f} {f1:>7.4f}{marker}")
    print(f"{'='*60}")


def main():
    args   = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    meta   = load_meta(args)

    _, dev_loader = build_dataloaders(args, meta["char2id"], meta["tag2id"])

    model = load_model(args, meta).to(device)
    rows  = evaluate_detail(model, dev_loader, meta["id2tag"], device)
    print_table(rows, args.model_name)


if __name__ == "__main__":
    main()
