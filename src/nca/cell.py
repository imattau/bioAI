import torch
import torch.nn as nn
import torch.nn.functional as F


class NCACell(nn.Module):
    def __init__(self, hidden_dim: int = 128, fire_rate: float = 0.5):
        super().__init__()
        self.fire_rate = fire_rate
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.update = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.target_proj = nn.Conv2d(1, hidden_dim, kernel_size=1) if hidden_dim > 1 else None

    def forward(self, state: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        x = state
        if target is not None and self.target_proj is not None:
            x = x + self.target_proj(target)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        dx = self.update(x)
        mask = (torch.rand_like(x[:, :1, :, :]) < self.fire_rate).float()
        return state + mask * dx


class NCA(nn.Module):
    def __init__(self, cell: NCACell, grid_size: tuple[int, int] = (32, 32),
                 channels: int = 16):
        super().__init__()
        self.cell = cell
        self.grid_size = grid_size
        self.channels = channels
        self.input_proj = nn.Conv2d(1, channels, kernel_size=1)
        self.channels = channels

    def forward(self, seed: torch.Tensor, steps: int = 64,
                target: torch.Tensor | None = None) -> torch.Tensor:
        state = self.input_proj(seed)
        states = [state]
        for _ in range(steps):
            state = self.cell(state, target)
            states.append(state)
        return torch.stack(states)

    def generate(self, seed: torch.Tensor, steps: int = 64,
                 target: torch.Tensor | None = None,
                 consistency_fn=None) -> torch.Tensor:
        state = self.input_proj(seed)
        for _ in range(steps):
            new_state = self.cell(state, target)
            if consistency_fn is not None:
                new_state = consistency_fn(state, new_state)
            state = new_state
        return state
