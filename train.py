# train.py
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from src.data.dataset import HXRULDataset
from src.data.dataloader import collate_fn
from src.models.transformer import QARTransformerEncoder


def main():
    # 1. 初始化数据
    dataset = HXRULDataset(raw_data_dir="D:/raw_data/")
    # 动态获取特征维度（取第一个样本的 pack 特征数）
    sample = dataset[0]
    input_dim = sample["features"].shape[1]

    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4
    )

    # 2. 初始化模型 & 优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QARTransformerEncoder(
        input_dim=input_dim,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=128,
        dropout=0.1,
        output_dim=1,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = MSELoss()

    # 3. 训练循环
    num_epochs = 50
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            features = batch["padded_features"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            rul = batch["rul"].to(device)

            optimizer.zero_grad()

            # 前向传播
            encoded_vec = model(features, padding_mask)  # (batch, 32)

            # 📌 此处 encoded_vec 就是你想要的“航段特征向量”
            # 后续可接 MLP 预测 RUL，或直接与 rul 计算损失
            pred_rul = encoded_vec.squeeze(-1)  # 假设 output_dim=1 或接一个回归头
            loss = criterion(pred_rul, rul)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}"
        )


if __name__ == "__main__":
    main()
