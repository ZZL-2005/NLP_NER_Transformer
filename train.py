from pathlib import Path

import torch
import torch.nn as nn

from tools.builder import build_dataloaders, build_model
from tools.configer import parse_args, print_config
from tools.trainer import train_one_epoch, evaluate, save_checkpoint
from tools.loader import save_json, set_seed, load_meta


def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir) / args.exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 读取 meta
    meta = load_meta(args)

    # 打印配置
    print_config(args, meta, device)

    # 数据
    train_loader, dev_loader = build_dataloaders(
        args=args,
        char2id=meta["char2id"],
        tag2id=meta["tag2id"],
    )

    # 模型
    model = build_model(
        args=args,
        vocab_size=meta["vocab_size"],
        num_tags=meta["num_tags"],
        pad_token_id=meta["pad_token_id"],
    ).to(device)

    print(model)

    # 损失 & 优化器
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练记录
    history = {
        "train_loss_epoch": [],
        "train_eval_loss": [],
        "train_token_acc": [],
        "train_entity_precision": [],
        "train_entity_recall": [],
        "train_entity_f1": [],
        "dev_loss": [],
        "dev_token_acc": [],
        "dev_entity_precision": [],
        "dev_entity_recall": [],
        "dev_entity_f1": [],
    }

    best_dev_f1 = -1.0
    best_epoch = -1

    # 训练主循环
    for epoch in range(1, args.num_epochs + 1):
        # 1. 训练一轮，得到“训练过程中的平均loss”
        train_loss_epoch = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=args.num_epochs,
        )

        # 2. epoch结束后，重新静态评估 train
        train_metrics = evaluate(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            id2tag=meta["id2tag"],
        )

        # 3. 静态评估 dev
        dev_metrics = evaluate(
            model=model,
            dataloader=dev_loader,
            criterion=criterion,
            device=device,
            id2tag=meta["id2tag"],
        )

        # 记录 train
        history["train_loss_epoch"].append(train_loss_epoch)
        history["train_eval_loss"].append(train_metrics["loss"])
        history["train_token_acc"].append(train_metrics["token_acc"])
        history["train_entity_precision"].append(train_metrics["entity_precision"])
        history["train_entity_recall"].append(train_metrics["entity_recall"])
        history["train_entity_f1"].append(train_metrics["entity_f1"])

        # 记录 dev
        history["dev_loss"].append(dev_metrics["loss"])
        history["dev_token_acc"].append(dev_metrics["token_acc"])
        history["dev_entity_precision"].append(dev_metrics["entity_precision"])
        history["dev_entity_recall"].append(dev_metrics["entity_recall"])
        history["dev_entity_f1"].append(dev_metrics["entity_f1"])

        # 打印日志
        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss_epoch={train_loss_epoch:.4f} "
            f"train_eval_loss={train_metrics['loss']:.4f} "
            f"train_token_acc={train_metrics['token_acc']:.4f} "
            f"train_entity_p={train_metrics['entity_precision']:.4f} "
            f"train_entity_r={train_metrics['entity_recall']:.4f} "
            f"train_entity_f1={train_metrics['entity_f1']:.4f} "
            f"dev_loss={dev_metrics['loss']:.4f} "
            f"dev_token_acc={dev_metrics['token_acc']:.4f} "
            f"dev_entity_p={dev_metrics['entity_precision']:.4f} "
            f"dev_entity_r={dev_metrics['entity_recall']:.4f} "
            f"dev_entity_f1={dev_metrics['entity_f1']:.4f}"
        )

        # 保存最优模型：按 dev entity_f1
        if dev_metrics["entity_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["entity_f1"]
            best_epoch = epoch

            save_checkpoint(
                path=save_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_dev_metric=best_dev_f1,
                args=args,
                meta=meta,
            )

            print(
                f"New best model saved! "
                f"best_epoch={best_epoch}, best_dev_f1={best_dev_f1:.4f}"
            )

    # 保存训练历史
    save_json(history, save_dir / "history.json")

    # 保存训练摘要
    summary = {
        "best_epoch": best_epoch,
        "best_dev_f1": best_dev_f1,
        "num_epochs": args.num_epochs,
        "exp_name": args.exp_name,
        "model_name": args.model_name,
    }
    save_json(summary, save_dir / "train_summary.json")

    print("Training finished.")
    print(f"Best epoch = {best_epoch}")
    print(f"Best dev f1 = {best_dev_f1:.4f}")


if __name__ == "__main__":
    main()