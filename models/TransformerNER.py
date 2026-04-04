import torch
import torch.nn as nn

from models.positional_encoding import SinusoidalPositionalEncoding
from models.TransformerEncoder import TransformerEncoder


class TransformerNER(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        num_layers: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.pad_token_id=pad_token_id
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id,
        )

        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.classifier = nn.Linear(d_model, num_tags)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
        attention_mask: torch.Tensor = None,
    ):
        # input_ids: [B, L]
        x = self.embedding(input_ids)   # [B, L, D]
        x = self.pos_encoding(x)        # [B, L, D]
        x = self.dropout(x)
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id)   # [B, L]
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        if return_hidden_states:
            x, hidden_states = self.encoder(x, return_hidden_states=True,attention_mask=attention_mask)
        else:
            x = self.encoder(x,attention_mask=attention_mask)

        logits = self.classifier(x)     # [B, L, num_tags]

        if return_hidden_states:
            return logits, hidden_states
        return logits
