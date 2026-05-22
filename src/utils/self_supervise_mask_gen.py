# src/utils/self_supervise_mask_gen.py
import torch
from tqdm import tqdm


class SelfSuperviseMaskGenerator:
    def __init__(
        self, total_mask_ratio: float, block_len: int, channel_mask_ratio: float
    ):
        self.total_mask_ratio = total_mask_ratio
        self.block_len = block_len
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
            max_mask_start_idx = valid_seq_len - self.block_len
            mask_start_idx = torch.randint(low=0, high=max_mask_start_idx, size=(n_blocks, 1), device=x.device)
            offset = torch.arange(start=0, end=self.block_len, step=1, dtype=torch.int, device=x.device)
            offsets = offset.repeat(n_blocks, 1)
            mask_indices = mask_start_idx.repeat(1, self.block_len)
            mask_indices = mask_indices + offsets
            mask_indices = mask_indices.view(-1)
            mask_bool[batch_idx, mask_indices, :] = True
            
            for i in range(n_blocks):
                max_mask_start_idx = valid_seq_len - self.block_len
                mask_start_idx = torch.randint(
                    low=0, high=max_mask_start_idx, size=(1,), device=x.device
                ).item()
                mask_bool[
                    batch_idx, mask_start_idx : mask_start_idx + self.block_len, :
                ] = True

            # channel masking
            channel_mask = torch.rand((valid_seq_len, dim), device=x.device) < self.channel_mask_ratio
            mask_bool[batch_idx, :valid_seq_len, :] |= channel_mask[:, :]

        x_masked = x.clone()
        x_masked[mask_bool] = 0.0

        return x_masked, mask_bool
