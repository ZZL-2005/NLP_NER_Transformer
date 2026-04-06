# NLP_NER_Transformer/models/FullAttentionResidual.py
import torch
import torch.nn as nn
from models.Normalization import RMSNorm
import math

class FullAttentionResidual(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.d_model = d_model
        self.norm = RMSNorm(d_model, eps=eps)

    def forward(
        self,
        query: torch.Tensor,
        sources: list[torch.Tensor],
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        query:   [D]
        sources: 长度为S的list，每个元素形状 [B, T, D]

        return:
            h: [B, T, D]
        """
        assert len(sources) > 0, "sources不能为空"

        # [S, B, T, D]
        V = torch.stack(sources, dim=0)

        # K = RMSNorm(V)
        K = self.norm(V)

        # logits: [S, B, T]
        # 每个source、每个batch、每个token位置一个分数
        logits = torch.einsum("d,sbtd->sbt", query, K)

        # 在source维做softmax
        alpha = torch.softmax(logits, dim=0)

        # 加权求和 -> [B, T, D]
        h = torch.einsum("sbt,sbtd->btd", alpha, V)

        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1).to(h.dtype)

        return h
    

class FullAttentionResidual_MY(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # 把每个history压缩后的向量映射成key
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        sources: list[torch.Tensor],
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        query: [D]
            可学习query向量，由外部作为nn.Parameter维护

        sources: 长度为S的list，每个元素形状 [B, T, D]
            所有历史hidden states

        attention_mask: [B, T] or None
            1表示有效token，0表示padding

        return:
            h: [B, T, D]
        """
        assert len(sources) > 0, "sources不能为空"
        assert query.dim() == 1 and query.shape[0] == self.d_model, \
            f"query形状应为[{self.d_model}]，实际得到{tuple(query.shape)}"

        # [S, B, T, D]
        V = torch.stack(sources, dim=0)

        # -----------------------------------
        # 1. 先把每个source在token维上做平均 -> [S, B, D]
        # -----------------------------------
        if attention_mask is not None:
            # [B, T] -> float
            mask = attention_mask.to(V.dtype)

            # 防止全padding时分母为0
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)   # [B, 1]

            # 广播后:
            # V:    [S, B, T, D]
            # mask: [1, B, T, 1]
            pooled = (V * mask.unsqueeze(0).unsqueeze(-1)).sum(dim=2) / denom.unsqueeze(0)
            # pooled: [S, B, D]
        else:
            pooled = V.mean(dim=2)  # [S, B, D]

        # -----------------------------------
        # 2. 用可学习投影得到K -> [S, B, D]
        # -----------------------------------
        K = self.k_proj(pooled)

        # -----------------------------------
        # 3. qk / sqrt(d) -> logits: [S, B]
        # -----------------------------------
        logits = torch.einsum("d,sbd->sb", query, K) / math.sqrt(self.d_model)

        # 在source维归一化
        alpha = torch.softmax(logits, dim=0)   # [S, B]

        # -----------------------------------
        # 4. 用alpha对原始V加权求和 -> [B, T, D]
        # -----------------------------------
        h = torch.einsum("sb,sbtd->btd", alpha, V)

        # padding位置清零
        if attention_mask is not None:
            h = h * attention_mask.unsqueeze(-1).to(h.dtype)

        return h