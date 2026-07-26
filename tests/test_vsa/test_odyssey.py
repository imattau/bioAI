import torch
from src.vsa import VSA
from src.vsa.odyssey import (
    OdysseyReasoning, BASE_RELATIONS, COMPOSITION_FORMULAS,
)


class TestOdysseyReasoning:
    DIM = 1000

    def setup_method(self):
        self.vsa = VSA(dim=self.DIM, device="cpu")
        self.odyssey = OdysseyReasoning(self.vsa)
        self.odyssey.build()

    def test_base_relation_recall(self):
        hop = self.odyssey.hop
        assert hop is not None
        for name, vec in self.odyssey.base.items():
            recalled = hop.recall(vec, steps=30, beta=3.0)
            sim = self.vsa.similarity(recalled, vec).item()
            assert sim > 0.99, f"{name}: recall cos_sim={sim}"

    def test_compositional_accuracy(self):
        acc = self.odyssey.accuracy(noise=0.0)
        print(f"\n  Compositional accuracy (clean): {acc*100:.0f}%")
        assert acc > 0.90, f"Accuracy {acc} < 0.90"

    def test_noise_robustness(self):
        for noise_level in [0.1, 0.3, 0.5, 1.0]:
            acc = self.odyssey.accuracy(noise=noise_level)
            print(f"  Accuracy @ noise={noise_level}: {acc*100:.0f}%")
            assert acc > 0.80, (
                f"Accuracy {acc} < 0.80 at noise={noise_level}"
            )

    def test_non_commutativity(self):
        fb = self.odyssey._compose("father_of", "brother_of")
        bf = self.odyssey._compose("brother_of", "father_of")
        sim = self.vsa.similarity(fb, bf).item()
        assert abs(sim) < 0.15, (
            f"father∘brother vs brother∘father too similar: sim={sim}"
        )

    def test_triple_composition(self):
        predicted, sim = self.odyssey.query("great_grandfather_of")
        assert predicted == "great_grandfather_of", (
            f"Triple composition: got '{predicted}' (sim={sim:.4f}), "
            f"expected 'great_grandfather_of'"
        )
        assert sim > 0.99, f"Triple recall sim={sim}"

    def test_heavy_noise_cleaned_to_stored_pattern(self):
        hop = self.odyssey.hop
        assert hop is not None
        for noise_level in [0.5, 1.0, 2.0, 3.0]:
            for vec in list(self.odyssey.compositions.values())[:3]:
                noisy = vec + noise_level * torch.randn(self.DIM, device=vec.device)
                recalled = hop.recall(noisy, steps=30, beta=3.0)
                recalled_max_sim = max(
                    self.vsa.similarity(recalled, stored).item()
                    for stored in self.odyssey.all_vectors().values()
                )
                assert recalled_max_sim > 0.99, (
                    f"Noise={noise_level}: recalled not close to any pattern "
                    f"(max_sim={recalled_max_sim})"
                )

    def test_orthogonality(self):
        all_vecs = self.odyssey.all_vectors()
        names = list(all_vecs.keys())

        max_sim = 0.0
        worst_pair = ("", "")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = abs(self.vsa.similarity(all_vecs[names[i]], all_vecs[names[j]]).item())
                if sim > max_sim:
                    max_sim = sim
                    worst_pair = (names[i], names[j])

        print(f"  Max cross-similarity: {max_sim:.4f} ({worst_pair[0]} vs {worst_pair[1]})")
        assert max_sim < 0.3, (
            f"Vectors not sufficiently orthogonal: max_sim={max_sim}"
        )

    def test_repeated_queries_deterministic(self):
        results = []
        for _ in range(5):
            name, sim = self.odyssey.query("grandfather_of")
            results.append((name, sim))
        first = results[0]
        for r in results[1:]:
            assert r[0] == first[0], f"Determinism broken: {r[0]} != {first[0]}"
            assert abs(r[1] - first[1]) < 1e-6, f"Sim changed: {r[1]} != {first[1]}"

    def test_all_compositional_queries_return_correct(self):
        for name in COMPOSITION_FORMULAS:
            predicted, sim = self.odyssey.query(name)
            assert predicted == name, (
                f"'{name}' predicted as '{predicted}' (sim={sim:.4f})"
            )
            assert sim > 0.99, f"'{name}': recall sim={sim}"

    def test_composition_distinct_from_base(self):
        for comp_name, comp_vec in self.odyssey.compositions.items():
            for base_name, base_vec in self.odyssey.base.items():
                sim = self.vsa.similarity(comp_vec, base_vec).item()
                assert abs(sim) < 0.3, (
                    f"Composition '{comp_name}' too similar to base "
                    f"'{base_name}': sim={sim}"
                )

    def test_hopfield_capacity_adequate(self):
        hop = self.odyssey.hop
        assert hop is not None
        n_stored = len(hop)
        capacity_estimate = int(0.14 * self.DIM)
        assert n_stored <= capacity_estimate, (
            f"Stored {n_stored} patterns exceeds estimated capacity "
            f"{capacity_estimate} for D={self.DIM}"
        )

    def test_novel_input_hallucinates_without_confidence(self):
        hop = self.odyssey.hop
        assert hop is not None

        novel = self.odyssey._compose("son_of", "daughter_of")
        recalled = hop.recall(novel, steps=30, beta=3.0)

        sim_to_cue = self.vsa.similarity(recalled, novel).item()
        max_stored = max(
            self.vsa.similarity(recalled, v).item()
            for v in self.odyssey.all_vectors().values()
        )

        assert sim_to_cue < 0.5, (
            f"Novel input converged to output similar to cue: "
            f"sim(cue, recalled)={sim_to_cue}"
        )

    def test_known_composition_passes_confidence(self):
        for name in COMPOSITION_FORMULAS:
            predicted, sim = self.odyssey.query(name, min_confidence=0.5)
            assert predicted == name, (
                f"'{name}' rejected at min_confidence=0.5: predicted='{predicted}'"
            )
            assert sim > 0.99, f"'{name}': sim={sim}"

    def test_confidence_rejects_novel_input(self):
        hop = self.odyssey.hop
        assert hop is not None

        novel = self.odyssey._compose("son_of", "daughter_of")
        recalled = hop.recall(novel, steps=30, beta=3.0)
        sim_to_cue = self.vsa.similarity(recalled, novel).item()

        assert sim_to_cue < 0.5, (
            f"Novel input should trigger low confidence: "
            f"sim(cue, recalled)={sim_to_cue}"
        )

    def test_known_composition_noisy_passes_confidence(self):
        for noise_level in [0.1, 0.3, 0.5]:
            acc = self.odyssey.accuracy(noise=noise_level, min_confidence=0.5)
            assert acc > 0.80, (
                f"Accuracy {acc} < 0.80 at noise={noise_level}, "
                f"min_confidence=0.5"
            )

    def test_confidence_threshold_separates_known_from_novel(self):
        hop = self.odyssey.hop
        assert hop is not None

        novel = self.odyssey._compose("son_of", "daughter_of")
        novel_recalled = hop.recall(novel, steps=30, beta=3.0)
        novel_sim = self.vsa.similarity(novel_recalled, novel).item()

        known = list(self.odyssey.compositions.values())[0]
        known_recalled = hop.recall(known, steps=30, beta=3.0)
        known_sim = self.vsa.similarity(known_recalled, known).item()

        print(f"\n  Known composition: sim(cue, recalled)={known_sim:.4f}")
        print(f"  Novel composition: sim(cue, recalled)={novel_sim:.4f}")

        assert known_sim > 0.99, (
            f"Known composition should have sim(cue, recalled) near 1.0: "
            f"{known_sim}"
        )
        assert novel_sim < 0.5, (
            f"Novel composition should have sim(cue, recalled) < 0.5: "
            f"{novel_sim}"
        )
        separation = known_sim - novel_sim
        assert separation > 0.5, (
            f"Insufficient separation between known ({known_sim}) and "
            f"novel ({novel_sim}): gap={separation}"
        )
