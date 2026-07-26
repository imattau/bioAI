"""Experiment 1: Encoder Validation — Does Text → VSA Preserve Information?"""

import torch
from src.text import VSAEncoder
from src.vsa import VSA


class TestEncoderDeterminism:
    def setup_method(self):
        self.encoder = VSAEncoder(vsa=VSA(dim=1000, device="cpu"))

    def test_same_text_identical_vector(self):
        v1 = self.encoder.encode("hello world")
        v2 = self.encoder.encode("hello world")
        sim = self.encoder.vsa.similarity(v1, v2).item()
        assert sim > 0.99, f"Same text should give identical vector, sim={sim}"

    def test_different_text_different_vector(self):
        v1 = self.encoder.encode("cat sat on mat")
        v2 = self.encoder.encode("quantum physics")
        sim = self.encoder.vsa.similarity(v1, v2).item()
        assert sim < 0.5, f"Different texts should differ, sim={sim}"

    def test_output_is_bipolar(self):
        hv = self.encoder.encode("a simple test")
        assert hv.dim() == 1
        assert hv.shape[0] == 1000
        assert torch.all((hv == 1) | (hv == -1))


class TestEncoderSimilarity:
    def setup_method(self):
        self.encoder = VSAEncoder(vsa=VSA(dim=1000, device="cpu"))

    def test_word_overlap_preserved(self):
        sim_high = self.encoder.similarity(
            "the cat sat on the mat", "the cat sat on the rug")
        sim_low = self.encoder.similarity(
            "the cat sat on the mat", "quantum physics")
        assert sim_high > sim_low, (
            f"Overlap should be more similar: "
            f"overlap={sim_high:.4f} vs distinct={sim_low:.4f}"
        )

    def test_order_matters(self):
        sim = self.encoder.similarity("dog bites man", "man bites dog")
        assert sim < 0.8, (
            f"Order-reversed should be distinguishable: sim={sim}"
        )

    def test_prefix_more_similar_than_unrelated(self):
        ab = self.encoder.encode("hello world")
        abc = self.encoder.encode("hello world foo")
        xyz = self.encoder.encode("bar baz qux")
        sim_prefix = self.encoder.vsa.similarity(ab, abc).item()
        sim_unrelated = self.encoder.vsa.similarity(ab, xyz).item()
        assert sim_prefix > sim_unrelated, (
            f"Prefix should beat unrelated: "
            f"prefix={sim_prefix:.4f} vs unrelated={sim_unrelated:.4f}"
        )

    def test_subword_similarity_gradient(self):
        a = self.encoder.encode("the cat")
        b = self.encoder.encode("the cat sat")
        c = self.encoder.encode("the cat sat on")
        d = self.encoder.encode("quantum physics")
        sim_ab = self.encoder.vsa.similarity(a, b).item()
        sim_ac = self.encoder.vsa.similarity(a, c).item()
        sim_ad = self.encoder.vsa.similarity(a, d).item()
        assert sim_ab > sim_ad, f"1-word overlap beats none: {sim_ab:.4f} vs {sim_ad:.4f}"
        assert sim_ac > sim_ad, f"2-word overlap beats none: {sim_ac:.4f} vs {sim_ad:.4f}"


class TestEncoderCapacity:
    def setup_method(self):
        self.encoder = VSAEncoder(vsa=VSA(dim=1000, device="cpu"))

    def test_low_collision_rate(self):
        torch.manual_seed(42)
        sentences = []
        for _ in range(1000):
            words = [f"w{torch.randint(0, 500, ()).item()}" for _ in range(3)]
            sentences.append(" ".join(words))
        hvs = self.encoder.encode_batch(sentences)
        dots = hvs @ hvs.T
        diag_mask = torch.eye(1000, dtype=torch.bool)
        off_diag = dots[~diag_mask]
        collisions = int((off_diag > 990).sum().item())
        assert collisions < 10, (
            f"Too many near-identical pairs: {collisions}/500k"
        )

    def test_deterministic_gives_same_batch(self):
        hvs1 = self.encoder.encode_batch(["hello world", "foo bar"])
        hvs2 = self.encoder.encode_batch(["hello world", "foo bar"])
        for i in range(2):
            sim = self.encoder.vsa.similarity(hvs1[i], hvs2[i]).item()
            assert sim > 0.99, f"Batch {i} not deterministic: sim={sim}"
