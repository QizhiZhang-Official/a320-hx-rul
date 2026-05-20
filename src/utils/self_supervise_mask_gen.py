# src/utils/self_supervise_mask_gen.py
import torch
import numpy as np


class SelfSuperviseMaskGenerator:
    def __init__(
        self, total_mask_ratio: float, block_len: int, channel_mask_ratio: float
    ):
        self.total_mask_ratio = total_mask_ratio
        self.block_len = (block_len,)
        self.channel_mask_ratio = channel_mask_ratio

    def exec(
        self, x: torch.Tensor, padding_mask: torch.Tensor
    ) -> torch.Tensor | torch.Tensor:
        batch, seq_len, dim = x.shape
        valid_mask = ~padding_mask
        mask_bool = torch.zeros(batch, seq_len, dim, dtype=torch.bool, device=x.device)

        for batch_idx in range(batch):
            valid_seq_len = valid_mask[batch_idx].count_nonzero().item()

            # block masking
            n_blocks = int(valid_seq_len * self.total_mask_ratio)
            for i in range(n_blocks):
                max_mask_start_idx = valid_seq_len - self.block_len
                mask_start_idx = torch.randint(
                    0, max_mask_start_idx, device=x.device
                ).item()
                mask_bool[
                    batch_idx, mask_start_idx : mask_start_idx + self.block_len, :
                ] = True

            # channel masking
            for seq in range(valid_seq_len):
                for channel in range(dim):
                    random_value = torch.rand(device=x.device).item()
                    if random_value < self.channel_mask_ratio:
                        mask_bool[batch_idx, seq, channel] = True

        x_masked = x.clone()
        x_masked[mask_bool] = 0.0

        return x_masked, mask_bool
