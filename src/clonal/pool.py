import torch
from .module import ClonalModule


class ClonalPool:
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 affinity_threshold: float = 0.6,
                 max_modules: int = 50, prune_after: int = 100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.affinity_threshold = affinity_threshold
        self.max_modules = max_modules
        self.prune_after = prune_after
        self.modules: list[ClonalModule] = []
        self.step = 0

    def process(self, x: torch.Tensor, mutation_strength: float = 0.01,
                lr: float = 0.01) -> tuple[torch.Tensor, bool]:
        if not self.modules:
            module = ClonalModule(x, self.input_dim, self.hidden_dim)
            self.modules.append(module)
            self.step += 1
            return module(x), True

        affs = [m.affinity(x) for m in self.modules]
        best_idx = max(range(len(affs)), key=lambda i: affs[i])
        best_aff = affs[best_idx]

        if best_aff > self.affinity_threshold:
            module = self.modules[best_idx]
            if best_aff > self.affinity_threshold + 0.2:
                child = module.clone(x, mutation_strength)
                child.local_update(x, lr)
                self.modules.append(child)
                self._prune()
                self.step += 1
                return child(x), True
            else:
                module.local_update(x, lr)
                module.last_used = self.step
                self.step += 1
                return module(x), False
        else:
            module = ClonalModule(x, self.input_dim, self.hidden_dim)
            module.local_update(x, lr)
            self.modules.append(module)
            self._prune()
            self.step += 1
            return module(x), True

    def _prune(self):
        if len(self.modules) <= self.max_modules:
            return
        self.modules.sort(key=lambda m: m.last_used)
        self.modules = self.modules[-(self.max_modules * 3 // 4):]

    def __len__(self) -> int:
        return len(self.modules)
