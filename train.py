import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from src.data.dataset import HXRULDataset
from src.data.dataloader import collate_fn
from src.utils.scaler import DataScaler
from src.utils.self_supervise_mask_gen import SelfSuperviseMaskGenerator
from src.models.encoder import Encoder


def train_encoder():
    with open("configs/train_config.yaml", "r") as f:
        train_config = yaml.safe_load(f)
    CONFIG = train_config["encoder_training_config"]

    scaler = DataScaler(
        raw_data_dir=CONFIG["raw_data_dir"],
        use_sample=CONFIG["use_sample_for_scaler_fitting"],
    )

    dataset = HXRULDataset(CONFIG["raw_data_dir"], scaler=scaler)
    n_total = len(dataset)
    n_val = n_total * CONFIG["val_set_ratio"]
    train_set_indices = list(range(0, n_total - n_val))
    val_set_indices = list(range(n_total - n_val, n_total))
    train_set = Subset(dataset=dataset, indices=train_set_indices)
    val_set = Subset(dataset=dataset, indices=val_set_indices)

    train_set_loader = DataLoader(
        dataset=train_set,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    val_set_loader = DataLoader(
        dataset=val_set,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    model = Encoder(
        feat_dim=CONFIG["feat_dim"],
        embed_dim=CONFIG["embed_dim"],
        n_head=CONFIG["n_head"],
        n_layers=CONFIG["n_layers"],
        dropout=CONFIG["dropout"],
    ).to(device=CONFIG["device"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=CONFIG["epochs"]
    )
    criterion = torch.nn.SmoothL1Loss()
    amp_scaler = torch.amp.GradScaler()
    mask_generator = SelfSuperviseMaskGenerator(
        total_mask_ratio=CONFIG["total_mask_ratio"],
        block_len=CONFIG["block_len"],
        channel_mask_ratio=CONFIG["channel_mask_ratio"],
    )
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        
