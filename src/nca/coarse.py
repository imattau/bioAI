import torch
import torch.nn as nn


class CoarseConditioner(nn.Module):
    def __init__(self, vsa_dim: int, grid_h: int, grid_w: int):
        super().__init__()
        self.proj = nn.Linear(vsa_dim, grid_h * grid_w)

    def forward(self, vsa_vector: torch.Tensor) -> torch.Tensor:
        return self.proj(vsa_vector).sigmoid().reshape(1, 1, 28, 28)
