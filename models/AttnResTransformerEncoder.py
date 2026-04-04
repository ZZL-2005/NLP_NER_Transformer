from typing import List, Tuple
import torch
import torch.nn as nn

from models.AttnResTransformerEncoderLayer import AttnResTransformerEncoderLayer


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

        last_hidden = history[-1]
        return last_hidden, history