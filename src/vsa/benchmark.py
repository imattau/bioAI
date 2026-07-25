import torch
from .primitives import VSA
from .store import AssociativeStore
from .hopfield import HopfieldNet


class RavenLikeTask:
    def __init__(self, vsa: VSA, dim: int = 1000):
        self.vsa = vsa
        self.dim = dim

    def make_analogy(self, a_char: str, b_char: str, c_char: str, d_char: str):
        role_v = self.vsa.encode_roles(["A", "B", "C", "D"])
        filler_v = self.vsa.encode_fillers([a_char, b_char, c_char, d_char])
        ab = self.vsa.bind(role_v["A"], filler_v[a_char]) + self.vsa.bind(role_v["B"], filler_v[b_char])
        cd = self.vsa.bind(role_v["C"], filler_v[c_char]) + self.vsa.bind(role_v["D"], filler_v[d_char])
        return ab, cd

    def make_distractors(self, n: int = 5) -> list[torch.Tensor]:
        return [self.vsa.make_vector() for _ in range(n)]


def accuracy(predictions: list[int], targets: list[int]) -> float:
    correct = sum(1 for p, t in zip(predictions, targets) if p == t)
    return correct / len(targets) if targets else 0.0
