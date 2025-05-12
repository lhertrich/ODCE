import torch
import torch.nn as nn
import torch.nn.functional as F

class QFormerAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_queries: int = 32,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        """
        A stack of cross-attention + self-attention layers (Q-Former),
        mapping CLIP tokens → a small set of learnable query embeddings.
        """
        super().__init__()
        self.d_model = output_dim
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, output_dim) * 0.02)
        self.clip_proj = nn.Linear(input_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, clip_feats: torch.Tensor) -> torch.Tensor:
        """
        clip_feats: (B, seq_len, clip_dim)
        returns:   (B, num_queries, d_model)
        """
        B, seq_len, _ = clip_feats.size()
        tokens = self.clip_proj(clip_feats)               
        queries = self.query_tokens.expand(B, -1, -1)      
        cat = torch.cat([queries, tokens], dim=1)          
        out = self.transformer(cat)                        
        
        return out[:, :queries.size(1), :]                 
