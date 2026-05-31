# src/models/encoder.py
import torch
import torch.nn as nn

from tqdm import tqdm


class PositionalEmbedder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 7000):
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
    def __init__(
        self, feat_dim: int, embed_dim: int, n_head: int, n_layers: int, dropout: float
    ):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, embed_dim)
        self.pos_encoder = PositionalEmbedder(d_model=embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=embed_dim * 4,
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
        self.mode = 'training'

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, device: str):
        checkpoint = torch.load(f=checkpoint_path, map_location=device)
        encoder = cls(
            feat_dim=checkpoint["feat_dim"],
            embed_dim=checkpoint["embed_dim"],
            n_head=checkpoint["n_head"],
            n_layers=checkpoint["n_layers"],
            dropout=checkpoint["dropout"],
        )
        encoder.load_state_dict(state_dict=checkpoint["model_state_dict"])

        encoder.to(device=device)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        encoder.mode = 'inference'

        return encoder

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.encoder(h, src_key_padding_mask=padding_mask)

        if self.mode == 'training':
            y = self.recon_head(h)
        if self.mode == 'inference':
            valid_mask = ~padding_mask
            full2d_mask = valid_mask.unsqueeze(-1).float()
            full3d_mask = full2d_mask.repeat(1, 1, h.shape[-1])
            valid_h = h * full3d_mask
            y = valid_h.sum(dim=1) / full3d_mask.sum(dim=1)

        return y
    
    def encode_one_lifecycle(self, lifecycle_data: list[dict], verbose: bool) -> list[dict]:
        for i in tqdm(range(len(lifecycle_data)), disable=not verbose):
            x = lifecycle_data[i]['x']
            padding_mask = lifecycle_data[i]['padding_mask']
            y = self.forward(x=x, padding_mask=padding_mask)
            lifecycle_data[i]['y'] = y
            lifecycle_data[i].pop('x')
        
        return lifecycle_data
