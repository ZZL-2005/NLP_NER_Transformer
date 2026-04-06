"""
分析不同模型各层 hidden state L2 范数分布

图1：子层输入范数（h_l）—— 语义对齐的比较
  - Transformer : history[l]（累积残差和）
  - AttnRes     : FullAttentionResidual 的输出（子层真正的输入）

图2：AttnRes 子层输出范数（delta v_l）
  - 对比 attnres vs attnres_my 的 delta 分布
"""
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from tools.loader import load_meta
from tools.builder import build_dataloaders
from tools.configer import parse_args
from models.TransformerNER import TransformerNER
from models.Attention_Residual_Kimi import AttnResTransformerNER, AttnResTransformerNER_MY


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_BATCHES = 50  # batch 数量

CKPT_PATHS = {
    "transformer": "checkpoints/transformer/best_model.pt",
    "attnres":     "checkpoints/attnres_transformer/best_model.pt",
    "attnres_my":  "checkpoints/attnres_transformer_my/best_model.pt",
}
MODEL_CLASSES = {
    "transformer": TransformerNER,
    "attnres":     AttnResTransformerNER,
    "attnres_my":  AttnResTransformerNER_MY,
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
# 工具：把有效 token 的 L2 范数追加到列表
# ─────────────────────────────────────────────
def append_norms(store, idx, h, mask):
    """store: list of list; idx: 层索引; h: [B,T,D]; mask: [B,T] bool"""
    valid = h[mask]                  # [N, D]
    store[idx].extend(valid.norm(dim=-1).cpu().tolist())


# ─────────────────────────────────────────────
# 图1 数据收集
# ─────────────────────────────────────────────
@torch.no_grad()
def collect_input_norms_transformer(model, dataloader, num_batches):
    """
    用 hook 捕获每个 sublayer 前的累积残差状态。
    hook 挂在 norm1（attn 前）和 norm2（FFN 前）的输入上。
    captured[0] = norm1 input of layer 0 = embedding。
    最后额外捕获最后一层 encoder layer 的输出 h8。
    返回 2*num_layers + 1 个列表。
    """
    captured = []

    def make_hook():
        def hook(_module, args, _output):
            # args[0] 是 norm 的输入，即 sublayer 前的累积残差
            captured.append(args[0].detach())
        return hook

    # 捕获最后一层 layer 的输出（= h8）
    def make_output_hook():
        def hook(_module, _input, output):
            captured.append(output.detach())
        return hook

    handles = []
    for layer in model.encoder.layers:
        handles.append(layer.norm1.register_forward_hook(make_hook()))
        handles.append(layer.norm2.register_forward_hook(make_hook()))
    # 最后一层的输出
    handles.append(model.encoder.layers[-1].register_forward_hook(make_output_hook()))

    store = None
    for i, batch in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE).bool()

        captured.clear()
        model(ids, attention_mask=mask)
        all_inputs = list(captured)  # 2*num_layers + 1 个点

        if store is None:
            store = [[] for _ in all_inputs]
        for l, h in enumerate(all_inputs):
            append_norms(store, l, h, mask)

    for h in handles:
        h.remove()
    return store


@torch.no_grad()
def collect_input_norms_attnres(model, dataloader, num_batches):
    """
    用 forward hook 捕获每个 FullAttentionResidual 的输出（= 子层输入 h_l）。
    captured[0] = attn_res output of layer 0 = embedding（1个source时权重为1）
    共 2*num_layers + 1 个点（最后一个是 final_res 的输出 h_final）。
    """
    captured = []

    def make_hook():
        def hook(_module, _input, output):
            captured.append(output.detach())
        return hook

    handles = []
    for layer in model.encoder.layers:
        handles.append(layer.attn_res.register_forward_hook(make_hook()))
        handles.append(layer.ffn_res.register_forward_hook(make_hook()))

    # final_res hook（如果存在）
    if hasattr(model.encoder, 'final_res'):
        handles.append(model.encoder.final_res.register_forward_hook(make_hook()))

    store = None
    for i, batch in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE).bool()

        captured.clear()
        model(ids, attention_mask=mask)
        all_inputs = list(captured)

        if store is None:
            store = [[] for _ in all_inputs]
        for l, h in enumerate(all_inputs):
            append_norms(store, l, h, mask)

    for h in handles:
        h.remove()
    return store


# ─────────────────────────────────────────────
# 图2 数据收集：AttnRes delta（子层输出 v_l）
# ─────────────────────────────────────────────
@torch.no_grad()
def collect_delta_norms_attnres(model, dataloader, num_batches):
    """返回 history 中各 delta 的范数（不含 embedding）。"""
    store = None
    for i, batch in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE).bool()

        x = model.embedding(ids)
        x = model.pos_encoding(x)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        _, history = model.encoder(x, attention_mask=mask)

        deltas = history[1:]   # 去掉 embedding，只看子层输出
        if store is None:
            store = [[] for _ in deltas]
        for l, h in enumerate(deltas):
            append_norms(store, l, h, mask)
    return store


# ─────────────────────────────────────────────
# 图2 数据收集：Transformer delta（各子层残差分支输出）
# ─────────────────────────────────────────────
@torch.no_grad()
def collect_delta_norms_transformer(model, dataloader, num_batches):
    """
    用 hook 捕获每个子层（attn、ffn）的输出，即被加到残差上的增量。
    hook 挂在 self_attn 和 linear2（FFN 第二层）的输出上。
    共 2*num_layers 个点，与 AttnRes delta 对齐。
    """
    captured = []

    def make_hook():
        def hook(_module, _input, output):
            # self_attn 输出是 (attn_output, attn_weights)，取第一个
            if isinstance(output, tuple):
                output = output[0]
            captured.append(output.detach())
        return hook

    handles = []
    for layer in model.encoder.layers:
        handles.append(layer.attn.register_forward_hook(make_hook()))
        handles.append(layer.ffn.register_forward_hook(make_hook()))

    store = None
    for i, batch in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE).bool()

        captured.clear()
        model(ids, attention_mask=mask)

        if store is None:
            store = [[] for _ in captured]
        for l, h in enumerate(captured):
            append_norms(store, l, h, mask)

    for h in handles:
        h.remove()
    return store


# ─────────────────────────────────────────────
# 统计
# ─────────────────────────────────────────────
def summarize(store):
    means, stds = [], []
    for norms in store:
        t = torch.tensor(norms)
        means.append(t.mean().item())
        stds.append(t.std().item())
    return means, stds


# ─────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────
COLORS  = {"transformer": "steelblue", "attnres": "tomato", "attnres_my": "seagreen"}
MARKERS = {"transformer": "o",         "attnres": "s",       "attnres_my": "^"}

def _plot_one(ax, name, means, stds, xs=None):
    if xs is None:
        xs = list(range(len(means)))
    c, m = COLORS[name], MARKERS[name]
    ax.plot(xs, means, label=name, color=c, marker=m, markersize=4)
    ax.fill_between(
        xs,
        [u - s for u, s in zip(means, stds)],
        [u + s for u, s in zip(means, stds)],
        alpha=0.15, color=c,
    )


def plot_all(input_results, delta_results, save_path="dev_hidden_delta_hidden.png"):
    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 图1：子层输入范数 ──
    ax = axes[0]
    for name, (means, stds) in input_results.items():
        _plot_one(ax, name, means, stds)
    ax.set_title("Sublayer Input Norm (h_l)  ±1 std\n"
                 "Transformer: accumulated residual | AttnRes: AttnRes aggregation output")
    ax.set_xlabel("Sublayer index  (0 = embedding)")
    ax.set_ylabel("Mean L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 图2：AttnRes delta 范数 ──
    ax = axes[1]
    for name, (means, stds) in delta_results.items():
        xs = list(range(1, len(means) + 1))   # 从 1 开始，对应子层编号
        _plot_one(ax, name, means, stds, xs)
    ax.set_title("Sublayer Output Norm (delta v_l)  ±1 std\n"
                 "Transformer: sublayer output | AttnRes: v_l from history")
    ax.set_xlabel("Sublayer index  (1 = first attn output)")
    ax.set_ylabel("Mean L2 Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"图已保存到 {save_path}")


# ─────────────────────────────────────────────
# main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    meta = load_meta(args)
    train_loader, _ = build_dataloaders(args, meta["char2id"], meta["tag2id"])

    input_results = {}
    delta_results = {}

    # Transformer
    print("收集 transformer ...")
    model_t = load_model("transformer")
    input_results["transformer"] = summarize(
        collect_input_norms_transformer(model_t, train_loader, NUM_BATCHES)
    )
    delta_results["transformer"] = summarize(
        collect_delta_norms_transformer(model_t, train_loader, NUM_BATCHES)
    )
    del model_t

    # AttnRes
    print("收集 attnres ...")
    model_a = load_model("attnres")
    input_results["attnres"] = summarize(
        collect_input_norms_attnres(model_a, train_loader, NUM_BATCHES)
    )
    delta_results["attnres"] = summarize(
        collect_delta_norms_attnres(model_a, train_loader, NUM_BATCHES)
    )
    del model_a

    # AttnRes MY
    print("收集 attnres_my ...")
    model_m = load_model("attnres_my")
    input_results["attnres_my"] = summarize(
        collect_input_norms_attnres(model_m, train_loader, NUM_BATCHES)
    )
    delta_results["attnres_my"] = summarize(
        collect_delta_norms_attnres(model_m, train_loader, NUM_BATCHES)
    )
    del model_m

    # 打印
    for label, results in [("=== 子层输入范数 ===", input_results),
                            ("=== AttnRes delta 范数 ===", delta_results)]:
        print(f"\n{label}")
        for name, (means, stds) in results.items():
            print(f"  [{name}]")
            for l, (m, s) in enumerate(zip(means, stds)):
                print(f"    layer {l:2d}: mean={m:.4f}  std={s:.4f}")

    plot_all(input_results, delta_results)


if __name__ == "__main__":
    main()
