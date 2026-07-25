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
        self.use_count = 1
        self.birth = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def affinity(self, x: torch.Tensor) -> torch.Tensor:
        a = x.flatten()
        b = self.receptor.flatten()
        return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)

    def clone(self, receptor: torch.Tensor, mutation_strength: float = 0.01) -> "ClonalModule":
        child = ClonalModule(receptor, self.net[0].in_features, self.net[0].out_features)
        child.load_state_dict(self.state_dict())
        child.use_count = 1
        child.birth = 0
        with torch.no_grad():
            for p in child.net.parameters():
                p += mutation_strength * torch.randn_like(p)
        return child

    def local_update(self, x: torch.Tensor, lr: float = 0.01,
                     exemplar: torch.Tensor | None = None,
                     replay_weight: float = 0.1):
        pred = self.forward(x)
        loss = torch.nn.functional.mse_loss(pred, x)
        if exemplar is not None:
            pred_ex = self.forward(exemplar)
            loss = loss + replay_weight * torch.nn.functional.mse_loss(pred_ex, exemplar)
        loss.backward()
        with torch.no_grad():
            for p in self.net.parameters():
                p -= lr * p.grad
                p.grad = None
