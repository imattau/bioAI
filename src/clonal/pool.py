import torch
from .module import ClonalModule


class ClonalPool:
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 affinity_threshold: float = 0.6,
                 clone_margin: float = 0.2,
                 max_modules: int = 50, lr: float = 0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.affinity_threshold = affinity_threshold
        self.clone_margin = clone_margin
        self.max_modules = max_modules
        self.default_lr = lr
        self.modules: list[ClonalModule] = []
        self.age = 0

    def process(self, x: torch.Tensor, lr: float | None = None,
                mutation_strength: float = 0.01) -> tuple[torch.Tensor, bool]:
        lr = lr or self.default_lr
        self.age += 1

        if not self.modules:
            module = ClonalModule(x, self.input_dim, self.hidden_dim)
            module.local_update(x, lr * 0.5)
            module.last_used = self.age
            self.modules.append(module)
            return module(x), True

        affs = [m.affinity(x) for m in self.modules]
        best_idx = max(range(len(affs)), key=lambda i: affs[i])
        best_aff = affs[best_idx]

        if best_aff > self.affinity_threshold + self.clone_margin:
            child = self.modules[best_idx].clone(x, mutation_strength)
            child.local_update(x, lr * 0.5)
            child.last_used = self.age
            self.modules.append(child)
            self._prune()
            return child(x), True

        elif best_aff > self.affinity_threshold:
            module = self.modules[best_idx]
            module.local_update(x, lr * 0.1)
            module.last_used = self.age
            return module(x), False
        else:
            module = ClonalModule(x, self.input_dim, self.hidden_dim)
            module.local_update(x, lr * 0.5)
            module.last_used = self.age
            self.modules.append(module)
            self._prune()
            return module(x), True

    def _prune(self):
        if len(self.modules) <= self.max_modules:
            return
        self.modules.sort(key=lambda m: m.last_used)
        self.modules = self.modules[-(self.max_modules * 3 // 4):]

    def __len__(self) -> int:
        return len(self.modules)
