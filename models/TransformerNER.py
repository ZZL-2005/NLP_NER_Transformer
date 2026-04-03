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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, L]
        x = self.embedding(input_ids)   # [B, L, D]
        x = self.pos_encoding(x)        # [B, L, D]
        x = self.dropout(x)

        x = self.encoder(x)             # [B, L, D]
        logits = self.classifier(x)     # [B, L, num_tags]
        return logits