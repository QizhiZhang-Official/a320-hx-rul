import torch
from torch.utils.data import DataLoader

def collate_fn(batch) -> dict:
    features = [item['features'] for item in batch]
    rul = [item['rul'] for item in batch]
    meta = [item['meta'] for item in batch]
    
    lengths = [f.size(0) for f in features]
    max_len = max(lengths)
    