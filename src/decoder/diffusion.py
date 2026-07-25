import torch
import torch.nn.functional as F


def noise_schedule_linear(num_steps: int, eps: float = 1e-3) -> torch.Tensor:
    t = torch.linspace(eps, 1.0, num_steps)
    return 1 - t


class DiscreteDiffusion:
    def __init__(self, mask_token_id: int, num_steps: int = 1000,
                 schedule: str = "linear"):
        self.mask_token_id = mask_token_id
        self.num_steps = num_steps
        if schedule == "linear":
            self.sigma = noise_schedule_linear(num_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    def corrupt(self, token_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        corrupted = token_ids.clone()
        batch_size, seq_len = token_ids.shape
        t_vals = t.view(-1, 1)
        mask_prob = 1.0 - self.sigma[t.long()].view(-1, 1)
        mask = torch.rand(batch_size, seq_len, device=token_ids.device) < mask_prob
        pad_mask = token_ids == (self.mask_token_id + 1)
        mask = mask & ~pad_mask
        corrupted[mask] = self.mask_token_id
        return corrupted

    def compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor,
                     corrupted_ids: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            ignore_index=-100,
        )
