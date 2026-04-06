"""
可视化 AttnRes 每一轮聚合的注意力权重分布（alpha 热力图）。

包含各 encoder layer 内的 attn_res / ffn_res，以及最终的 final_res。

用法：
    python analyze_peak.py                    # 默认分析 attnres
    python analyze_peak.py --model attnres_my
    python analyze_peak.py --model both
"""
import json
import argparse
import sys
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.loader import load_meta
from tools.builder import build_dataloaders
from tools.configer import parse_args as _parse_args
from models.Attention_Residual_Kimi import AttnResTransformerNER, AttnResTransformerNER_MY
from models.FullAttentionResidual import FullAttentionResidual, FullAttentionResidual_MY

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_BATCHES = 50

CKPT_PATHS = {
    "attnres":    "checkpoints/attnres_transformer/best_model.pt",
    "attnres_my": "checkpoints/attnres_transformer_my/best_model.pt",
}
MODEL_CLASSES = {
    "attnres":    AttnResTransformerNER,
    "attnres_my": AttnResTransformerNER_MY,
}


# ─────────────────────────────────────────────
# 加载模型
# ─────────────────────────────────────────────
def load_model(name):
    ckpt  = torch.load(CKPT_PATHS[name], map_location="cpu")
    model = MODEL_CLASSES[name](**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(DEVICE)


# ─────────────────────────────────────────────
# 收集 alpha
# ─────────────────────────────────────────────
@torch.no_grad()
def collect_alpha(model, dataloader, num_batches):
    """
    收集每一轮聚合的 alpha 权重。
    返回 alpha_per_source: list[dict{source_idx -> list[float]}]
    长度 = 2*num_layers + 1（含 final_res）
    """
    captured_alpha  = []
    captured_sources = []

    def make_alpha_wrapper(module):
        orig = module.forward

        def patched_forward(query, sources, attention_mask=None):
            V = torch.stack(sources, dim=0)
            if isinstance(module, FullAttentionResidual):
                K      = module.norm(V)
                logits = torch.einsum("d,sbtd->sbt", query, K)
                alpha  = torch.softmax(logits, dim=0)
            else:
                mask_f = attention_mask
                if mask_f is not None:
                    denom  = mask_f.to(V.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
                    pooled = (V * mask_f.to(V.dtype).unsqueeze(0).unsqueeze(-1)).sum(dim=2) / denom.unsqueeze(0)
                else:
                    pooled = V.mean(dim=2)
                K      = module.k_proj(pooled)
                logits = torch.einsum("d,sbd->sb", query, K) / math.sqrt(module.d_model)
                alpha  = torch.softmax(logits, dim=0)
            captured_alpha.append(alpha.detach())
            captured_sources.append(len(sources))
            return orig(query, sources, attention_mask)

        module.forward = patched_forward

    # 挂 monkey-patch
    for layer in model.encoder.layers:
        make_alpha_wrapper(layer.attn_res)
        make_alpha_wrapper(layer.ffn_res)
    if hasattr(model.encoder, 'final_res'):
        make_alpha_wrapper(model.encoder.final_res)

    # 收集
    n_sublayers = len(model.encoder.layers) * 2
    if hasattr(model.encoder, 'final_res'):
        n_sublayers += 1

    alpha_per_source = [None for _ in range(n_sublayers)]

    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE).bool()

        captured_alpha.clear()
        captured_sources.clear()
        model(ids, attention_mask=mask)

        for l, alpha in enumerate(captured_alpha):
            n_src = captured_sources[l]

            if alpha_per_source[l] is None:
                alpha_per_source[l] = {s: [] for s in range(n_src)}
            for s in range(n_src):
                if s not in alpha_per_source[l]:
                    alpha_per_source[l][s] = []

            for s in range(n_src):
                if alpha.dim() == 3:
                    # [S, B, T] → per token
                    a_s = alpha[s][mask]
                    alpha_per_source[l][s].extend(a_s.cpu().tolist())
                else:
                    # [S, B] → per batch, expand to tokens
                    a_s = alpha[s]
                    for b in range(a_s.shape[0]):
                        n_tok = mask[b].sum().item()
                        alpha_per_source[l][s].extend([a_s[b].item()] * int(n_tok))

    return alpha_per_source


# ─────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────
def summarize_alpha(alpha_per_source):
    result = []
    for l_dict in alpha_per_source:
        layer_summary = {}
        for s, vals in l_dict.items():
            t = torch.tensor(vals, dtype=torch.float32)
            layer_summary[s] = {"mean": t.mean().item(), "std": t.std().item()}
        result.append(layer_summary)
    return result


# ─────────────────────────────────────────────
# 打印
# ─────────────────────────────────────────────
def get_sublayer_names(n):
    names = []
    n_layers = (n - 1) // 2 if n % 2 == 1 else n // 2
    for i in range(n_layers):
        names.append(f"L{i}_attn")
        names.append(f"L{i}_ffn")
    if n % 2 == 1:
        names.append("final_res")
    return names


def print_alpha(summary, model_name):
    n = len(summary)
    names = get_sublayer_names(n)

    print(f"\n{'='*80}")
    print(f"  {model_name}  —  Alpha 权重分配")
    print(f"{'='*80}")
    for l in range(n):
        n_src = len(summary[l])
        alpha_strs = []
        for s in range(n_src):
            a = summary[l][s]
            src_name = "emb" if s == 0 else f"v{s}"
            alpha_strs.append(f"{src_name}:{a['mean']:.3f}")
        print(f"  {names[l]:<12}  α = [{', '.join(alpha_strs)}]")
    print(f"{'='*80}\n")


# ─────────────────────────────────────────────
# 绘图：alpha 热力图
# ─────────────────────────────────────────────
def plot(all_summaries, save_path="analyze_peak.png"):
    n_models = len(all_summaries)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6), squeeze=False)

    for idx, (name, summary) in enumerate(all_summaries.items()):
        ax = axes[0][idx]
        n = len(summary)
        max_src = max(len(d) for d in summary)
        names = get_sublayer_names(n)

        mat = np.zeros((n, max_src))
        for l in range(n):
            for s, val in summary[l].items():
                mat[l, s] = val["mean"]

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(f"Alpha Weights ({name})", fontsize=12)
        ax.set_xlabel("Source index (0=emb)", fontsize=10)
        ax.set_ylabel("Aggregation step", fontsize=10)
        ax.set_yticks(range(n))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xticks(range(max_src))
        src_labels = ["emb"] + [f"v{i}" for i in range(1, max_src)]
        ax.set_xticklabels(src_labels, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)

        # 标注数值
        for l in range(n):
            for s in range(max_src):
                if mat[l, s] > 0.005:
                    ax.text(s, l, f"{mat[l,s]:.2f}", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if mat[l, s] > 0.5 else "black")

    plt.suptitle("Attention Residual Alpha Weights per Aggregation Step", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"图已保存到 {save_path}")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="attnres",
                        choices=["attnres", "attnres_my", "both"])
    parser.add_argument("--save_json", type=str, default="analyze_peak.json")
    parser.add_argument("--save_png",  type=str, default="analyze_peak.png")
    my_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    base_args = _parse_args()
    meta = load_meta(base_args)
    train_loader, _ = build_dataloaders(base_args, meta["char2id"], meta["tag2id"])

    names = ["attnres", "attnres_my"] if my_args.model == "both" else [my_args.model]

    all_summaries = {}
    for name in names:
        print(f"\n收集 {name} ...")
        model = load_model(name)
        raw   = collect_alpha(model, train_loader, NUM_BATCHES)
        summary = summarize_alpha(raw)
        all_summaries[name] = summary
        print_alpha(summary, name)
        del model

    # 保存 json
    save_data = {}
    for name, summary in all_summaries.items():
        save_data[name] = []
        for l_dict in summary:
            save_data[name].append({str(s): v for s, v in l_dict.items()})
    with open(my_args.save_json, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"数据已保存到 {my_args.save_json}")

    plot(all_summaries, my_args.save_png)


if __name__ == "__main__":
    main()
