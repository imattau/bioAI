import torch
from src.vsa import VSA


class TestVSA:
    def setup_method(self):
        self.vsa = VSA(dim=1000)

    def test_make_vector(self):
        v = self.vsa.make_vector()
        assert v.shape == (1000,)
        assert v.unique().numel() <= 2

    def test_bind_unbind_roundtrip(self):
        a = self.vsa.make_vector()
        b = self.vsa.make_vector()
        bound = self.vsa.bind(a, b)
        recovered = self.vsa.unbind(bound, b)
        sim = self.vsa.similarity(a, recovered)
        assert sim.item() > 0.9

    def test_bundle_preserves_similarity(self):
        items = [self.vsa.make_vector() for _ in range(3)]
        bundle = self.vsa.bundle(items)
        sims = [self.vsa.similarity(bundle, item).item() for item in items]
        assert all(s > 0.3 for s in sims)

    def test_permute_changes_vector(self):
        v = self.vsa.make_vector()
        p = self.vsa.permute(v, shifts=3)
        sim = self.vsa.similarity(v, p)
        assert sim.item() < 0.1

    def test_encoding_triple(self):
        roles = self.vsa.encode_roles(["A", "B"])
        fillers = self.vsa.encode_fillers(["cat", "dog"])
        triple = self.vsa.encode_triple(roles, fillers, "A", "cat")
        assert triple.dim() == 1 and triple.shape[0] == 1000
