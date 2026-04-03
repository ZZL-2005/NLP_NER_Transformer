import torch
import torch.nn as nn

from models.MultiHeadSelfAttention import MultiHeadSelfAttention
from models.FFN import FeedForward

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x,attention_mask=None):
        x = x + self.dropout1(self.attn(self.norm1(x),attention_mask=attention_mask))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x