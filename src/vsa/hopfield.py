import torch
import torch.nn as nn


class HopfieldNet(nn.Module):
    def __init__(self, dim: int, max_patterns: int = 100):
        super().__init__()
        self.dim = dim
        self.patterns: list[torch.Tensor] = []
        self.weights: torch.Tensor | None = None

    def store(self, pattern: torch.Tensor):
        self.patterns.append(pattern.detach().clone().flatten())
        self._update_weights()

    def store_batch(self, patterns: list[torch.Tensor]):
        for p in patterns:
            self.store(p)

    def _update_weights(self):
        stacked = torch.stack(self.patterns)
        self.weights = stacked.T @ stacked
        self.weights.fill_diagonal_(0)

    def recall(self, cue: torch.Tensor, steps: int = 10,
               beta: float = 1.0) -> torch.Tensor:
        state = cue.detach().clone().flatten()
        for _ in range(steps):
            logits = self.weights @ state
            state = torch.tanh(beta * logits)
        return state.reshape(cue.shape)

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        s = state.flatten()
        return -0.5 * (s @ self.weights @ s) if self.weights is not None else torch.tensor(0.0)

    def __len__(self) -> int:
        return len(self.patterns)
