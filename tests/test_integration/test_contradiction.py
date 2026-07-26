import torch
from src.vsa import VSA, AssociativeStore


class TestContradiction:
    DIM = 1000

    def setup_method(self):
        self.vsa = VSA(dim=self.DIM, device="cpu")

        self.role_subject = self.vsa.make_vector()
        self.role_property = self.vsa.make_vector()

        self.mammal = self.vsa.make_vector()
        self.anteater = self.vsa.make_vector()
        self.warm = self.vsa.make_vector()
        self.cold = self.vsa.make_vector()

        self.fact1 = self._encode(self.mammal, self.warm)
        self.fact2 = self._encode(self.anteater, self.mammal)
        self.fact3 = self._encode(self.anteater, self.cold)
        self.derived = self._encode(self.anteater, self.warm)

        self.store = AssociativeStore(dim=self.DIM, capacity=10)

    def _encode(self, subj: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        return self.vsa.bundle([
            self.vsa.bind(self.role_subject, subj),
            self.vsa.bind(self.role_property, prop),
        ])

    def _same_subject_count(self, query: torch.Tensor,
                            query_subject: torch.Tensor) -> int:
        subject_ref = self.vsa.bind(self.role_subject, query_subject)
        results = self.store.lookup(query, k=3)
        return sum(
            1 for v, _ in results
            if self.vsa.similarity(v, subject_ref).item() > 0.5
        )

    def _has_contradiction(self, query: torch.Tensor,
                           query_subject: torch.Tensor) -> bool:
        top_results = self.store.lookup(query, k=1)
        if top_results and top_results[0][1] > 0.99:
            return False
        return self._same_subject_count(query, query_subject) >= 2

    def test_contradiction_detected(self):
        self.store.insert(self.fact1, self.fact1)
        self.store.insert(self.fact2, self.fact2)
        self.store.insert(self.fact3, self.fact3)

        contra = self._has_contradiction(self.derived, self.anteater)
        count = self._same_subject_count(self.derived, self.anteater)

        print(f"\n  Facts about anteater in top-3: {count}")
        assert contra, "Contradiction not detected with fact3 present"

    def test_non_contradictory_query_returns_confident(self):
        self.store.insert(self.fact1, self.fact1)
        self.store.insert(self.fact2, self.fact2)
        self.store.insert(self.fact3, self.fact3)

        results = self.store.lookup(self.fact2, k=1)
        assert len(results) > 0
        assert results[0][1] > 0.99, (
            f"Direct fact2 should match exactly: {results[0][1]}"
        )

        contra = self._has_contradiction(self.fact2, self.anteater)
        assert not contra, "Direct query of a stored fact should not be contradiction"

    def test_consistent_only_no_false_contradiction(self):
        self.store.insert(self.fact1, self.fact1)
        self.store.insert(self.fact2, self.fact2)

        contra = self._has_contradiction(self.derived, self.anteater)
        count = self._same_subject_count(self.derived, self.anteater)

        print(f"\n  Facts about anteater in top-3 (no fact3): {count}")
        assert not contra, (
            "False contradiction without contradictory fact"
        )

    def test_novel_query_rejected(self):
        self.store.insert(self.fact1, self.fact1)
        self.store.insert(self.fact2, self.fact2)
        self.store.insert(self.fact3, self.fact3)

        novel_subj = self.vsa.make_vector()
        novel_prop = self.vsa.make_vector()
        novel = self._encode(novel_subj, novel_prop)
        results = self.store.lookup(novel, k=1)
        assert results[0][1] < 0.1, (
            f"Novel query should have low sim: {results[0][1]}"
        )

    def test_direct_fact3_query_confident(self):
        self.store.insert(self.fact1, self.fact1)
        self.store.insert(self.fact2, self.fact2)
        self.store.insert(self.fact3, self.fact3)

        results = self.store.lookup(self.fact3, k=1)
        assert results[0][1] > 0.99

        contra = self._has_contradiction(self.fact3, self.anteater)
        assert not contra, "Direct query of fact3 should not flag contradiction"

    def test_contradiction_specific_to_query(self):
        self.store.insert(self.fact1, self.fact1)
        self.store.insert(self.fact2, self.fact2)
        self.store.insert(self.fact3, self.fact3)

        derived_contra = self._has_contradiction(self.derived, self.anteater)
        fact1_contra = self._has_contradiction(self.fact1, self.mammal)
        fact3_contra = self._has_contradiction(self.fact3, self.anteater)

        assert derived_contra, "Derived query should detect contradiction"
        assert not fact1_contra, "Fact1 (mammal→warm) should not be contradiction"
        assert not fact3_contra, "Direct fact3 query should not be contradiction"
