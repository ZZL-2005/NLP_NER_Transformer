# NLP_NER_Transformer

## 环境配置

conda create -n nlp-ner-transformer python=3.10 -y
conda activate nlp-ner-transformer
pip install jupyter ipykernel matplotlib
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124

## 项目架构说明

```
NLP_NER_Transformer/
├── train.py                    # 主训练脚本（Transformer / AttnRes / AttnRes-MY）
├── eval.py                     # 评估脚本（dev 集上各实体类型的 P/R/F1）
├── case_study.py               # 案例分析（FP/FN 展示、双模型对比）
├── hidden_states.py            # 各层 hidden state L2 范数分析与可视化
├── analyze_peak.py             # AttnRes peak 层深度分析（alpha/范数/放大率/参数）
├── plot_history.py             # 训练曲线可视化（Loss/F1/P/R/Acc 对比）
├── data_proc.ipynb             # 数据预处理 notebook
│
├── models/                     # 模型定义
│   ├── TransformerNER.py                   # 标准 Transformer NER 模型
│   ├── TransformerEncoder.py               # Transformer Encoder（层堆叠）
│   ├── TransformerEncoderLayer.py          # 单层 Encoder（PreNorm: norm→attn→norm→ffn）
│   ├── Attention_Residual_Kimi.py          # AttnRes NER 模型（论文复现 + MY 变体）
│   ├── AttnResTransformerEncoder.py        # AttnRes Encoder（层堆叠，维护 history）
│   ├── AttnResTransformerEncoderLayer.py   # AttnRes 单层（attn_res→attn→ffn_res→ffn）
│   ├── FullAttentionResidual.py            # 注意力残差聚合模块（论文原版 + MY 变体）
│   ├── MultiHeadSelfAttention.py           # 多头自注意力（手写实现）
│   ├── FFN.py                              # 前馈网络（Linear→ReLU→Linear）
│   ├── positional_encoding.py              # 正弦位置编码
│   └── Normalization.py                    # RMSNorm
│
├── tools/                      # 工具函数
│   ├── configer.py             # 命令行参数解析
│   ├── loader.py               # 数据加载、meta 信息读取、工具函数
│   ├── builder.py              # DataLoader 和模型的构建工厂
│   └── trainer.py              # 训练/评估循环、实体抽取
│
├── dataset/                    # 数据集定义
│   └── dataset.py              # NERDataset（字符→id、标签→id、padding）
│
├── data/                       # 原始数据
│   ├── train.txt / train_TAG.txt    # 训练集（文本 + BIO 标签）
│   └── dev.txt / dev_TAG.txt        # 验证集
│
├── meta/                       # 预处理后的映射表
│   ├── char2id.json / id2char.json  # 字符↔id
│   └── tag2id.json / id2tag.json    # 标签↔id（B_LOC, I_LOC, B_ORG, ...）
│
├── checkpoints/                # 模型存档
│   ├── transformer/            # best_model.pt + history.json
│   ├── attnres_transformer/
│   └── attnres_transformer_my/
│
└── docs/                       # 文档
    └── report.md               # 实验报告
```

### 模型架构

三个模型共享 embedding + 位置编码 + 分类头，区别在 encoder 内部的残差机制：

| 模型 | 残差方式 | 说明 |
|------|---------|------|
| `transformer` | 标准残差 `h_l = h_{l-1} + sublayer(norm(h_{l-1}))` | 基线模型 |
| `attnres_transformer` | 注意力残差 `h_l = Σ α_i · v_i`，α 通过 RMSNorm + 可学习 query 计算 | 论文复现 |
| `attnres_transformer_my` | 注意力残差，α 通过 masked mean pooling + 线性投影计算 | 自定义变体 |

### 分析脚本

| 脚本 | 功能 |
|------|------|
| `eval.py` | 各实体类型（LOC/ORG/PER/T）细粒度 P/R/F1 |
| `case_study.py` | FP/FN 典型案例展示，双模型差异对比 |
| `hidden_states.py` | 各层 hidden state 范数分布，验证 PreNorm dilution |
| `analyze_peak.py` | AttnRes peak 层成因分析（alpha 分配、范数传导链、参数权重） |
| `plot_history.py` | 训练曲线对比（每个指标单独保存一张图） |

## 训练指令

python train.py \
  --exp_name transformer \
  --model_name transformer \
  --batch_size 32 \
  --lr 1e-3 \
  --num_epochs 50 \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 512 \
  --num_layers 4 \
  --max_len 5000 \
  --dropout 0.1 \
  --device cuda


python train.py \
  --exp_name attnres_transformer \
  --model_name attnres_transformer \
  --batch_size 32 \
  --lr 1e-3 \
  --num_epochs 50 \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 512 \
  --num_layers 4 \
  --max_len 5000 \
  --dropout 0.1 \
  --device cuda


python train.py \
  --exp_name attnres_transformer_my \
  --model_name attnres_transformer_my \
  --batch_size 32 \
  --lr 1e-3 \
  --num_epochs 50 \
  --d_model 256 \
  --n_heads 4 \
  --d_ff 512 \
  --num_layers 4 \
  --max_len 5000 \
  --dropout 0.1 \
  --device cuda
