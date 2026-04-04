import torch
import torch.nn as nn
from models.Normalization import RMSNorm


class FullAttentionResidual(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.d_model = d_model
        self.norm = RMSNorm(d_model, eps=eps)

    def forward(
        self,
        query: torch.Tensor,
        sources: list[torch.Tensor],
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

        return h