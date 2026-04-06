# NLP_NER_Transformer/models/AttnResTransformerEncoder.py
from typing import List, Tuple
import torch
import torch.nn as nn
from models.AttnResTransformerEncoderLayer import AttnResTransformerEncoderLayer,AttnResTransformerEncoderLayer_MY
from models.FullAttentionResidual import FullAttentionResidual, FullAttentionResidual_MY

class AttnResTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            AttnResTransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 最终聚合：对所有 history 做一次 attention residual，得到最终输出
        self.final_res = FullAttentionResidual(d_model)
        self.final_res_query = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: [B, T, D]

        return:
            last_hidden: [B, T, D]
            history: List[[B, T, D]]
        """
        history = [x]

        for layer in self.layers:
            history = layer(history, attention_mask=attention_mask)

        # 最终聚合：h_L = Σ α_i · v_i
        last_hidden = self.final_res(
            self.final_res_query,
            history,
            attention_mask=attention_mask,
        )
        return last_hidden, history



class AttnResTransformerEncoder_MY(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            AttnResTransformerEncoderLayer_MY(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 最终聚合
        self.final_res = FullAttentionResidual_MY(d_model)
        self.final_res_query = nn.Parameter(torch.zeros(d_model))

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: [B, T, D]

        return:
            last_hidden: [B, T, D]
            history: List[[B, T, D]]
        """
        history = [x]

        for layer in self.layers:
            history = layer(history, attention_mask=attention_mask)

        # 最终聚合
        last_hidden = self.final_res(
            self.final_res_query,
            history,
            attention_mask=attention_mask,
        )
        return last_hidden, history