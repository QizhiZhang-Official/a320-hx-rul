# src/models/encoder.py
import torch
import torch.nn as nn


class PositionalEmbedder(nn.Module):
    def __init__(self, d_model, max_len=7000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class Encoder(nn.Module):
    def __init__(self, feat_dim, embed_dim, n_head, n_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, embed_dim)
        self.pos_encoder = PositionalEmbedder(d_model=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.recon_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, feat_dim),
        )
    
    def forward(self, x, padding_mask):
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        
        return self.recon_head(h)


# class QAREncoder