import torch
import torch.nn as nn

from models.TransformerEncoderLayer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
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
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, return_hidden_states=False,attention_mask=None):
        hidden_states = []

        if return_hidden_states:
            hidden_states.append(x)   # 输入到第1层前的表示

        for layer in self.layers:
            x = layer(x,attention_mask=attention_mask)
            if return_hidden_states:
                hidden_states.append(x)   # 每层输出都记下来

        if return_hidden_states:
            return x, hidden_states
        return x