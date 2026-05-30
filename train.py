import torch
import yaml
import os
import time
import csv
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, Subset

from src.data.dataset import HXRULDataset
from src.data.dataloader import collate_fn
from src.utils.scaler import DataScaler
from src.utils.self_supervise_mask_gen import SelfSuperviseMaskGenerator
from src.models.encoder import Encoder


def format_sec(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def train_encoder():
    with open("configs/train_config.yaml", "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)
    CONFIG = train_config["encoder_training_config"]
    RAW_DATA_DIR = ''
    for dir in CONFIG["raw_data_dir"]:
        if os.path.exists(dir):
            RAW_DATA_DIR = dir
            break

    print("\nInitializing Scaler...")
    scaler = DataScaler(
        raw_data_dir=RAW_DATA_DIR,
        use_sample=CONFIG["use_sample_for_scaler_fitting"],
    )

    print("\nInitializing DataLoader...")
    
    dataset = HXRULDataset(raw_data_dir=RAW_DATA_DIR, scaler=scaler)
    n_total = len(dataset)
    n_val = int(n_total * CONFIG["val_set_ratio"])
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

    print("\nInitializing Model...")
    model = Encoder(
        feat_dim=CONFIG["feat_dim"],
        embed_dim=CONFIG["embed_dim"],
        n_head=CONFIG["n_head"],
        n_layers=CONFIG["n_layers"],
        dropout=CONFIG["dropout"],
    ).to(device=CONFIG["device"])
    print("\nInitializing Optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    print("\nInitializing Scheduler...")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=CONFIG["epochs"]
    )
    print("\nInitializing Criterion...")
    criterion = torch.nn.SmoothL1Loss()
    print("\nInitializing AmpScaler...")
    amp_scaler = torch.amp.GradScaler()
    print("\nInitializing MaskGenerator...")
    mask_generator = SelfSuperviseMaskGenerator(
        total_mask_ratio=CONFIG["total_mask_ratio"],
        block_len=CONFIG["block_len"],
        channel_mask_ratio=CONFIG["channel_mask_ratio"],
    )

    log_file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    os.makedirs("log/", exist_ok=True)
    with open(
        os.path.join("log", log_file_name), "w", newline="", encoding="utf-8"
    ) as f:
        csv.writer(f).writerow(
            ["Record_time", "Epoch", "Train_Loss", "Val_Loss", "LR", "Elapsed", "ETA"]
        )

    overall_start = time.time()
    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}")

        print("Training...")
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_set_loader):
            x = batch["padded_features"].to(CONFIG["device"])
            padding_mask = batch["padding_mask"].to(CONFIG["device"])
            x_masked, mask_bool = mask_generator.exec(x=x, padding_mask=padding_mask)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=CONFIG["device"]):
                recon = model(x_masked, padding_mask)
                loss = criterion(recon[mask_bool], x[mask_bool])

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer=optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer=optimizer)
            amp_scaler.update()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        print("Evaluating...")
        model.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_set_loader):
                x = batch["padded_features"].to(CONFIG["device"])
                padding_mask = batch["padding_mask"].to(CONFIG["device"])
                x_masked, mask_bool = mask_generator.exec(
                    x=x, padding_mask=padding_mask
                )
                with torch.amp.autocast(device_type=CONFIG["device"]):
                    recon = model(x_masked, padding_mask)
                    val_loss += criterion(recon[mask_bool], x[mask_bool]).item()
                n_val_batches += 1

        avg_train = train_loss / n_batches
        avg_val = val_loss / n_val_batches if n_val_batches > 0 else float("inf")
        elapsed = time.time() - overall_start
        avg_epoch = elapsed / (epoch + 1)
        eta = avg_epoch * (CONFIG["epochs"] - epoch - 1)
        print(
            f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {scheduler.get_last_lr()[0]:.4f} | Elapsed: {format_sec(elapsed)} | ETA: {format_sec(eta)}"
        )

        with open(
            os.path.join("log", log_file_name),
            "a",
            newline="",
            encoding="utf-8",
        ) as f:
            csv.writer(f).writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch + 1,
                    f"{avg_train:.4f}",
                    f"{avg_val:.4f}",
                    f"{scheduler.get_last_lr()[0]:.4f}",
                    f"{format_sec(elapsed)}",
                    f"{format_sec(eta)}",
                ]
            )

        save_path = os.path.join(os.getcwd(), "checkpoints", f"encoder{epoch + 1}.pth")
        torch.save(
            {
                "feat_dim": CONFIG["feat_dim"],
                "embed_dim": CONFIG["embed_dim"],
                "n_head": CONFIG["n_head"],
                "n_layers": CONFIG["n_layers"],
                "dropout": CONFIG["dropout"],
                "model_state_dict": model.state_dict(),
            },
            save_path,
        )


if __name__ == "__main__":
    train_encoder()
