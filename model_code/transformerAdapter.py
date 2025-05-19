import torch
import torch.nn as nn
import math


class TransformerAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        # self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, output_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.ln = nn.LayerNorm(output_dim)

        # nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.input_proj(x)
        # x = x + self.pos_embed
        x = self.transformer(x)
        return self.ln(x)
