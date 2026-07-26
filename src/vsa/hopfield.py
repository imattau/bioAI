import torch
import torch.nn as nn


class HopfieldNet(nn.Module):
    def __init__(self, dim: int, max_patterns: int = 100):
        super().__init__()
        self.dim = dim
        self.patterns: list[torch.Tensor] = []
        self.weights: torch.Tensor | None = None

    def store(self, pattern: torch.Tensor):
        p = pattern.detach().clone().flatten()
        self.patterns.append(p)
        if self.weights is None:
            self.weights = torch.zeros(self.dim, self.dim, device=p.device)
        self.weights += torch.outer(p, p)
        self.weights.fill_diagonal_(0)

    def store_batch(self, patterns: list[torch.Tensor]):
        for p in patterns:
            self.store(p)

    def recall(self, cue: torch.Tensor, steps: int = 10,
               beta: float = 1.0) -> torch.Tensor:
        if self.weights is None or len(self.patterns) == 0:
            return cue.detach().clone()
        state = cue.detach().clone().flatten()
        for _ in range(steps):
            logits = self.weights @ state
            state = torch.tanh(beta * logits)
        return state.reshape(cue.shape)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            return torch.tensor(0.0)
        s = state.flatten()
        return -0.5 * (s @ self.weights @ s)

    def get_state(self) -> dict:
        return {"patterns": self.patterns, "weights": self.weights}

    def set_state(self, patterns: list[torch.Tensor], weights: torch.Tensor | None):
        self.patterns = patterns
        self.weights = weights

    def __len__(self) -> int:
        return len(self.patterns)
