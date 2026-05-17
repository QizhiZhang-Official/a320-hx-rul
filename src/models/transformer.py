# src/models/transformer.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class QARTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        output_dim: int = 32
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        # src: (batch, seq_len, input_dim)
        # padding_mask: (batch, seq_len), True = pad
        x = self.input_proj(src)
        x = self.pos_encoder(x)
        
        # Transformer 编码
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # 🎯 Masked Mean Pooling: 将变长序列压缩为固定向量
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(-1).float()  # True=有效 -> 1.0
            x = x * mask
            x = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)  # 防除零
        else:
            x = x.mean(dim=1)
            
        x = self.fc_out(x)
        return x