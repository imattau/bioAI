import torch
import torch.nn as nn


class ClonalModule(nn.Module):
    def __init__(self, receptor: torch.Tensor, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.register_buffer("receptor", receptor.detach().clone())
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.age = 0
        self.last_used = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def affinity(self, x: torch.Tensor) -> torch.Tensor:
        a = x.flatten()
        b = self.receptor.flatten()
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)

    def clone(self, receptor: torch.Tensor, mutation_strength: float = 0.01) -> "ClonalModule":
        child = ClonalModule(receptor, self.net[0].in_features, self.net[0].out_features)
        child.load_state_dict(self.state_dict())
        with torch.no_grad():
            for p in child.net.parameters():
                p += mutation_strength * torch.randn_like(p)
        return child

    def local_update(self, x: torch.Tensor, lr: float = 0.01):
        pred = self.forward(x)
        loss = torch.nn.functional.mse_loss(pred, x)
        loss.backward()
        with torch.no_grad():
            for p in self.net.parameters():
                p -= lr * p.grad
                p.grad = None
