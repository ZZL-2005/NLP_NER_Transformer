"""
可视化各模型训练历史曲线。

用法：
    python plot_history.py                     # 对比所有已有 history
    python plot_history.py --models transformer attnres_transformer
    python plot_history.py --save_path my_plot.png
"""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
DEFAULT_MODELS = [
    "transformer",
    "attnres_transformer",
    "attnres_transformer_my",
    "transformer_crf",
    "attnres_transformer_crf",
]

COLORS = [
    "steelblue",
    "tomato",
    "seagreen",
    "darkorchid",
    "darkorange",
]

LINESTYLES = ["-", "-", "-", "--", "--"]

DISPLAY_NAMES = {
    "transformer":            "Transformer",
    "attnres_transformer":    "AttnRes",
    "attnres_transformer_my": "AttnRes-MY",
    "transformer_crf":        "Transformer+CRF",
    "attnres_transformer_crf":"AttnRes+CRF",
}

# history.json 里各字段 → (子图标题, y轴标签)
METRICS = {
    # 损失
    "train_loss_epoch": ("Train Loss (per epoch)",   "Loss"),
    "train_eval_loss":  ("Train Loss (on train set eval)", "Loss"),
    "dev_loss":         ("Dev Loss",                 "Loss"),
    # Token 准确率
    "train_token_acc":  ("Train Token Accuracy",     "Accuracy"),
    "dev_token_acc":    ("Dev Token Accuracy",        "Accuracy"),
    # 实体级指标
    "train_entity_f1":        ("Train Entity F1",        "F1"),
    "dev_entity_f1":          ("Dev Entity F1",           "F1"),
    "train_entity_precision": ("Train Entity Precision",  "Precision"),
    "dev_entity_precision":   ("Dev Entity Precision",    "Precision"),
    "train_entity_recall":    ("Train Entity Recall",     "Recall"),
    "dev_entity_recall":      ("Dev Entity Recall",       "Recall"),
}

# 图布局：[(key, key, ...), ...]，每行一组
LAYOUT = [
    ["train_loss_epoch",      "dev_loss"],
    ["train_entity_f1",       "dev_entity_f1"],
    ["train_entity_precision","dev_entity_precision"],
    ["train_entity_recall",   "dev_entity_recall"],
    ["train_token_acc",       "dev_token_acc"],
]


# ─────────────────────────────────────────────
# 读取
# ─────────────────────────────────────────────
def load_history(model_name, save_dir="checkpoints"):
    path = Path(save_dir) / model_name / "history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# 单个子图绘制
# ─────────────────────────────────────────────
def _draw_ax(ax, key, histories):
    title, ylabel = METRICS[key]
    for (name, hist), color, ls in zip(histories.items(), COLORS, LINESTYLES):
        if key not in hist or not hist[key]:
            continue
        vals  = hist[key]
        xs    = list(range(1, len(vals) + 1))
        label = DISPLAY_NAMES.get(name, name)
        ax.plot(xs, vals, label=label,
                color=color, linestyle=ls, linewidth=1.8,
                marker="o", markersize=2.5)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if "loss" in key.lower():
        ax.set_ylim(bottom=0)
    else:
        all_vals = [v for h in histories.values() for v in h.get(key, [])]
        if all_vals:
            lo = max(0.5, min(all_vals) - 0.02)
            ax.set_ylim(bottom=lo, top=1.01)


# ─────────────────────────────────────────────
# 绘图：每行单独保存一张图
# ─────────────────────────────────────────────
ROW_NAMES = ["loss", "entity_f1", "entity_precision", "entity_recall", "token_acc"]

def plot(histories, save_path="training_curves.png"):
    """
    每行 LAYOUT 单独保存为一张图。
    save_path 作为前缀，实际文件名形如：
        training_curves_loss.png
        training_curves_entity_f1.png
        ...
    """
    save_path = Path(save_path)
    stem   = save_path.stem
    suffix = save_path.suffix
    parent = save_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    for row_keys, row_name in zip(LAYOUT, ROW_NAMES):
        n_cols = len(row_keys)
        fig, axes = plt.subplots(1, n_cols,
                                 figsize=(7 * n_cols, 4),
                                 squeeze=False)

        for col_idx, key in enumerate(row_keys):
            _draw_ax(axes[0][col_idx], key, histories)

        plt.tight_layout()
        out = parent / f"{stem}_{row_name}{suffix}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存 {out}")


# ─────────────────────────────────────────────
# 打印最终最优值表格
# ─────────────────────────────────────────────
def print_summary(histories):
    KEY_METRICS = [
        ("dev_entity_f1",        "Dev F1 (best)",   max),
        ("dev_entity_precision", "Dev P (@ best F1)", None),
        ("dev_entity_recall",    "Dev R (@ best F1)", None),
        ("dev_loss",             "Dev Loss (min)",  min),
        ("train_loss_epoch",     "Train Loss (final)", None),
    ]

    print(f"\n{'─'*75}")
    print(f"{'Model':<28}", end="")
    print(f"{'Dev F1↑':>10} {'Dev P':>8} {'Dev R':>8} "
          f"{'DevLoss↓':>10} {'TrLoss':>8}")
    print(f"{'─'*75}")

    for name, hist in histories.items():
        f1s = hist.get("dev_entity_f1", [])
        ps  = hist.get("dev_entity_precision", [])
        rs  = hist.get("dev_entity_recall", [])
        dl  = hist.get("dev_loss", [])
        tl  = hist.get("train_loss_epoch", [])

        if not f1s:
            continue
        best_ep  = int(max(range(len(f1s)), key=lambda i: f1s[i]))
        best_f1  = f1s[best_ep]
        best_p   = ps[best_ep]  if ps  else float("nan")
        best_r   = rs[best_ep]  if rs  else float("nan")
        min_dl   = min(dl)      if dl  else float("nan")
        last_tl  = tl[-1]       if tl  else float("nan")

        label = DISPLAY_NAMES.get(name, name)
        print(f"{label:<28}"
              f"{best_f1:>10.4f} {best_p:>8.4f} {best_r:>8.4f} "
              f"{min_dl:>10.4f} {last_tl:>8.4f}")

    print(f"{'─'*75}\n")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="指定模型名列表，默认自动搜索所有已有 history")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_path", type=str, default="training_curves.png")
    args = parser.parse_args()

    # 确定要画哪些模型
    if args.models:
        candidates = args.models
    else:
        candidates = DEFAULT_MODELS

    histories = {}
    for name in candidates:
        hist = load_history(name, args.save_dir)
        if hist is not None:
            histories[name] = hist
            print(f"✓ 加载 {name}  ({len(hist.get('dev_entity_f1', []))} epochs)")
        else:
            print(f"  跳过 {name}（未找到 history.json）")

    if not histories:
        print("没有找到任何 history.json，退出。")
        return

    print_summary(histories)
    plot(histories, args.save_path)


if __name__ == "__main__":
    main()
