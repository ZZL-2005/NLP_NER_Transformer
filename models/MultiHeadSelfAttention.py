# NLP_NER_Transformer/models/MultiHeadSelfAttention.py
import math
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        # x: [B, L, D],out: [B, L, D]
        B, L, D = x.shape

        Q = self.q_proj(x)  # [B, L, D]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # [B, L, D] -> [B, L, H, Hd] -> [B, H, L, Hd]
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # attention score: [B, H, L, L]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            # [B, L] -> [B, 1, 1, L]
            mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)

            # mask=False 的位置是 PAD，把对应 score 变成一个极小值
            scores = scores.masked_fill(~mask, float("-inf"))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # [B, H, L, L] @ [B, H, L, Hd] -> [B, H, L, Hd]
        context = torch.matmul(attn_weights, V)

        # [B, H, L, Hd] -> [B, L, H, Hd] -> [B, L, D]
        context = context.transpose(1, 2).contiguous().view(B, L, D)

        if attention_mask is not None:
            context = context * attention_mask.unsqueeze(-1).to(context.dtype)

        out = self.out_proj(context)
        return out