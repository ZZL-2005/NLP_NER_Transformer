## 按照Kimi的论文中的描述复现https://arxiv.org/pdf/2603.15031
import torch
import torch.nn as nn

from models.positional_encoding import SinusoidalPositionalEncoding
from models.AttnResTransformerEncoder import AttnResTransformerEncoder


class AttnResTransformerNER(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_token_id,
        )
        self.pad_token_id = pad_token_id

        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.encoder = AttnResTransformerEncoder(
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
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        input_ids: [B, T]
        attention_mask: [B, T]

        return:
            logits: [B, T, num_tags]
        """
        x = self.embedding(input_ids)      # [B, T, D]
        x = self.pos_encoding(x)           # [B, T, D]
        x = self.dropout(x)

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id)
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        x, history = self.encoder(x, attention_mask=attention_mask)

        logits = self.classifier(x)        # [B, T, num_tags]
        return logits
