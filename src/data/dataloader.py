import torch
from torch.utils.data import DataLoader


def collate_fn(batch) -> dict:
    features = [item["features"] for item in batch]
    rul = [item["rul"] for item in batch]
    meta = [item["meta"] for item in batch]

    lengths = [f.size(0) for f in features]
    max_len = max(lengths)
    num_feat = features[0].size(1)
    num_batch = len(features)

    padded_features = torch.zeros(num_batch, max_len, num_feat)
    mask = torch.zeros(num_batch, max_len)

    for index, feature in enumerate(features):
        len = feature.size(0)
        padded_features[index, :len, :] = feature
        mask[index, :len] = 1.0

    return {"feature": padded_features, "mask": mask, "rul": rul, "meta": meta}

