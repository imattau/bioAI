import torch

from .primitives import VSA
from .hopfield import HopfieldNet

BASE_RELATIONS = [
    "father_of", "mother_of", "brother_of", "sister_of",
    "son_of", "daughter_of", "husband_of", "wife_of",
    "parent_of", "child_of",
]

COMPOSITION_FORMULAS = {
    "grandfather_of": ("father_of", "father_of"),
    "grandmother_of": ("mother_of", "mother_of"),
    "uncle_of": ("brother_of", "father_of"),
    "aunt_of": ("sister_of", "father_of"),
    "parent_of_parent": ("parent_of", "parent_of"),
}

TRIPLE_COMPOSITIONS = {
    "great_grandfather_of": ("father_of", "grandfather_of"),
}


class OdysseyReasoning:
    def __init__(self, vsa: VSA):
        self.vsa = vsa
        self.dim = vsa.dim

        self.base = {}
        self.compositions = {}
        self.triple = {}
        self.hop = None

    def build(self):
        self._make_base_relations()
        self._make_compositions()
        self._make_triple_compositions()
        self.hop = HopfieldNet(dim=self.dim)
        for v in list(self.base.values()) + list(self.compositions.values()) + list(self.triple.values()):
            self.hop.store(v)

    def _make_base_relations(self):
        for name in BASE_RELATIONS:
            self.base[name] = self.vsa.make_vector()

    def _compose(self, a_name: str, b_name: str) -> torch.Tensor:
        a = self.base.get(a_name)
        if a is None:
            a = self.compositions.get(a_name)
        if a is None:
            a = self.triple.get(a_name)
        b = self.base.get(b_name)
        if b is None:
            b = self.compositions.get(b_name)
        if b is None:
            b = self.triple.get(b_name)
        assert a is not None, f"Relation '{a_name}' not found"
        assert b is not None, f"Relation '{b_name}' not found"
        return self.vsa.bind(a, self.vsa.permute(b, shifts=1))

    def _make_compositions(self):
        for name, (a, b) in COMPOSITION_FORMULAS.items():
            self.compositions[name] = self._compose(a, b)

    def _make_triple_compositions(self):
        for name, (a, b) in TRIPLE_COMPOSITIONS.items():
            self.triple[name] = self._compose(a, b)

    def all_vectors(self) -> dict[str, torch.Tensor]:
        return {**self.base, **self.compositions, **self.triple}

    def query(self, composition_name: str,
              noise: float = 0.0,
              min_confidence: float = 0.0) -> tuple[str, float]:
        vec = self.compositions.get(composition_name)
        if vec is None:
            vec = self.triple.get(composition_name)
        if vec is None:
            raise ValueError(f"Unknown composition: {composition_name}")

        cue = vec
        if noise > 0:
            cue = vec + noise * torch.randn(self.dim, device=vec.device)

        assert self.hop is not None, "Call build() before query()"
        recalled = self.hop.recall(cue, steps=30, beta=3.0)

        if min_confidence > 0:
            sim_to_cue = self.vsa.similarity(recalled, cue).item()
            if sim_to_cue < min_confidence:
                return ("NOVEL", sim_to_cue)

        best_name = ""
        best_sim = -float("inf")
        for name, stored in self.all_vectors().items():
            sim = self.vsa.similarity(recalled, stored).item()
            if sim > best_sim:
                best_sim = sim
                best_name = name

        return best_name, best_sim

    def accuracy(self, noise: float = 0.0,
                 min_confidence: float = 0.0) -> float:
        correct = 0
        for name in self.compositions:
            predicted, _ = self.query(name, noise=noise,
                                      min_confidence=min_confidence)
            if predicted == name:
                correct += 1
        total = len(self.compositions)
        return correct / total if total > 0 else 0.0
