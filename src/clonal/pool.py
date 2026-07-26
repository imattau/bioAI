import torch
from .module import ClonalModule


class ClonalPool:
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 affinity_threshold: float = 0.6,
                 clone_margin: float = 0.2,
                 max_modules: int = 50, lr: float = 0.01,
                 replay_weight: float = 0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.affinity_threshold = affinity_threshold
        self.clone_margin = clone_margin
        self.max_modules = max_modules
        self.default_lr = lr
        self.replay_weight = replay_weight
        self.modules: list[ClonalModule] = []
        self.age = 0

    def process(self, x: torch.Tensor, lr: float | None = None,
                mutation_strength: float = 0.01) -> tuple[torch.Tensor, bool]:
        lr = lr or self.default_lr
        self.age += 1

        if not self.modules:
            module = ClonalModule(x, self.input_dim, self.hidden_dim)
            module.birth = self.age
            module.local_update(x, lr * 0.5)
            self.modules.append(module)
            return module(x), True

        affs = [m.affinity(x) for m in self.modules]
        best_idx = max(range(len(affs)), key=lambda i: affs[i])
        best_aff = affs[best_idx]

        if best_aff > self.affinity_threshold + self.clone_margin:
            parent = self.modules[best_idx]
            child = parent.clone(x, mutation_strength)
            child.birth = self.age
            child.local_update(x, lr * 0.5,
                               exemplar=child.receptor,
                               replay_weight=self.replay_weight)
            self.modules.append(child)
            self._prune()
            return child(x), True

        elif best_aff > self.affinity_threshold:
            module = self.modules[best_idx]
            module.local_update(x, lr * 0.1,
                                exemplar=module.receptor,
                                replay_weight=self.replay_weight)
            module.use_count += 1
            return module(x), False
        else:
            module = ClonalModule(x, self.input_dim, self.hidden_dim)
            module.birth = self.age
            module.local_update(x, lr * 0.5,
                                 exemplar=module.receptor,
                                 replay_weight=self.replay_weight)
            self.modules.append(module)
            self._prune()
            return module(x), True

    def _prune(self):
        if len(self.modules) <= self.max_modules:
            return
        def _score(m):
            return m.use_count * (self.age - m.birth)
        self.modules.sort(key=_score, reverse=True)
        self.modules = self.modules[:self.max_modules]

    def get_state(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "affinity_threshold": self.affinity_threshold,
            "clone_margin": self.clone_margin,
            "max_modules": self.max_modules,
            "default_lr": self.default_lr,
            "replay_weight": self.replay_weight,
            "age": self.age,
            "modules": [
                {"receptor": m.receptor,
                 "net_state": m.net.state_dict(),
                 "use_count": m.use_count,
                 "birth": m.birth}
                for m in self.modules
            ],
        }

    def set_state(self, state: dict):
        self.age = state["age"]
        self.modules.clear()
        for ms in state["modules"]:
            m = ClonalModule(ms["receptor"], state["input_dim"], state["hidden_dim"])
            m.net.load_state_dict(ms["net_state"])
            m.use_count = ms["use_count"]
            m.birth = ms["birth"]
            self.modules.append(m)

    def __len__(self) -> int:
        return len(self.modules)
