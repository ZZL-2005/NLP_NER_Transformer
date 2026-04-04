from models.FullAttentionResidual import FullAttentionResidual,FullAttentionResidual_MY
from models.FFN import FeedForward
from models.MultiHeadSelfAttention import MultiHeadSelfAttention
import torch
import torch.nn as nn

class AttnResTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attn_res_query = nn.Parameter(torch.zeros(d_model))
        self.ffn_res_query = nn.Parameter(torch.zeros(d_model))

        self.attn_res = FullAttentionResidual(d_model)
        self.ffn_res = FullAttentionResidual(d_model)

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, history: list[torch.Tensor], attention_mask=None) -> list[torch.Tensor]:
        # 1) attention前的depth聚合
        h_attn_in = self.attn_res(
            self.attn_res_query,
            history,
            attention_mask=attention_mask,
        )   # [B, T, D]

        # 2) self-attention
        attn_out = self.attn(self.attn_norm(h_attn_in), attention_mask=attention_mask)
        attn_out = self.attn_dropout(attn_out)

        # 3) 把attn输出加入history
        history = history + [attn_out]

        # 4) FFN前的depth聚合
        h_ffn_in = self.ffn_res(
            self.ffn_res_query,
            history,
            attention_mask=attention_mask,
        )      # [B, T, D]

        # 5) FFN
        ffn_out = self.ffn(self.ffn_norm(h_ffn_in))
        ffn_out = self.ffn_dropout(ffn_out)

        # 6) 把ffn输出加入history
        history = history + [ffn_out]

        return history



class AttnResTransformerEncoderLayer_MY(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attn_res_query = nn.Parameter(torch.zeros(d_model))
        self.ffn_res_query = nn.Parameter(torch.zeros(d_model))

        self.attn_res = FullAttentionResidual_MY(d_model)
        self.ffn_res = FullAttentionResidual_MY(d_model)

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, history: list[torch.Tensor], attention_mask=None) -> list[torch.Tensor]:
        # 1) attention前的depth聚合
        h_attn_in = self.attn_res(
            self.attn_res_query,
            history,
            attention_mask=attention_mask,
        )   # [B, T, D]

        # 2) self-attention
        attn_out = self.attn(self.attn_norm(h_attn_in), attention_mask=attention_mask)
        attn_out = self.attn_dropout(attn_out)

        # 3) 把attn输出加入history
        history = history + [attn_out]

        # 4) FFN前的depth聚合
        h_ffn_in = self.ffn_res(
            self.ffn_res_query,
            history,
            attention_mask=attention_mask,
        )      # [B, T, D]

        # 5) FFN
        ffn_out = self.ffn(self.ffn_norm(h_ffn_in))
        ffn_out = self.ffn_dropout(ffn_out)

        # 6) 把ffn输出加入history
        history = history + [ffn_out]

        return history
