import torch
import torch.nn as nn


class HopfieldNet(nn.Module):
    def __init__(self, dim: int, max_patterns: int = 100,
                 retrieval_mode: str = "classical",
                 modern_beta: float = 50.0):
        super().__init__()
        if retrieval_mode not in {"classical", "modern"}:
            raise ValueError("retrieval_mode must be 'classical' or 'modern'")
        self.dim = dim
        self.retrieval_mode = retrieval_mode
        self.modern_beta = modern_beta
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
        if self.retrieval_mode == "modern":
            patterns = torch.stack(self.patterns)
            query = cue.detach().clone().flatten()
            normalised_patterns = torch.nn.functional.normalize(patterns, dim=1)
            normalised_query = torch.nn.functional.normalize(
                query.unsqueeze(0), dim=1
            ).squeeze(0)
            scores = self.modern_beta * (normalised_patterns @ normalised_query)
            return (torch.softmax(scores, dim=0) @ patterns).reshape(cue.shape)
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
        return {
            "patterns": self.patterns,
            "retrieval_mode": self.retrieval_mode,
            "modern_beta": self.modern_beta,
        }

    def set_state(self, patterns: list[torch.Tensor]):
        self.patterns = patterns
        self.weights = None
        for p in patterns:
            if self.weights is None:
                self.weights = torch.zeros(self.dim, self.dim, device=p.device)
            self.weights += torch.outer(p, p)
            self.weights.fill_diagonal_(0)

    def __len__(self) -> int:
        return len(self.patterns)
