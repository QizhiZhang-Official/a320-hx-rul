# src/data/dataloader.py
import torch
from torch.utils.data import DataLoader


def collate_fn(batch: list) -> dict:
    features = [item["features"] for item in batch]
    rul = [item["rul"] for item in batch]
    meta = [item["meta"] for item in batch]

    lengths = [f.size(0) for f in features]
    max_len = max(lengths)
    num_feat = features[0].size(1)
    num_batch = len(features)

    padded_features = torch.zeros(num_batch, max_len, num_feat)
    padding_mask = torch.zeros(num_batch, max_len, dtype=torch.bool)

    for index, feature in enumerate(features):
        seq_len = feature.size(0)
        padded_features[index, :seq_len, :] = feature
        padding_mask[index, seq_len:] = True

    return {
        "padded_features": padded_features,
        "padding_mask": padding_mask,
        "rul": torch.stack(rul),
        "meta": meta,
    }
