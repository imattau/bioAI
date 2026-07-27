import torch

from src.vsa import VSA, HopfieldNet, AssociativeStore
from src.vsa.hash_store import VSAHashStore


class TestBrittleRetrieval:
    DIM = 2000
    N_SIMILAR = 100
    N_FLIPS = 20
    N_RANDOM = 10000

    def setup_method(self):
        self.vsa = VSA(dim=self.DIM, device="cpu")
        self.base = self.vsa.make_vector()
        self.similar, self.randoms = self._build_dataset()
        self.store = AssociativeStore(
            dim=self.DIM,
            capacity=self.N_SIMILAR + self.N_RANDOM + 10,
        )

    def _build_dataset(self):
        similar = []
        for _ in range(self.N_SIMILAR):
            s = self.base.clone()
            flip_idx = torch.randperm(self.DIM)[:self.N_FLIPS]
            s[flip_idx] *= -1
            similar.append(s)
        randoms = [self.vsa.make_vector() for _ in range(self.N_RANDOM)]
        return similar, randoms

    def _store_all(self, store_obj):
        for s in self.similar:
            store_obj.insert(s, s)
        for r in self.randoms:
            store_obj.insert(r, r)

    def _precision_assoc(self, store) -> float:
        correct = 0
        for s in self.similar:
            results = store.lookup(s, k=1)
            if results:
                exact = any(
                    self.vsa.similarity(v, s).item() > 0.999
                    for v, _ in results
                )
                if exact:
                    correct += 1
        return correct / self.N_SIMILAR

    def _precision_hash(self, store) -> float:
        correct = 0
        for s in self.similar:
            result = store.lookup(s)
            if result is not None:
                sim = self.vsa.similarity(result, s).item()
                if sim > 0.999:
                    correct += 1
        return correct / self.N_SIMILAR

    def _precision_hopfield(self, hop) -> float:
        correct = 0
        for i, s in enumerate(self.similar):
            recalled = hop.recall(s, steps=30, beta=3.0)
            sims = [self.vsa.similarity(recalled, t).item()
                    for t in self.similar]
            best = max(range(len(sims)), key=lambda j: sims[j])
            if best == i:
                correct += 1
        return correct / self.N_SIMILAR

    def test_associative_store_high_precision(self):
        store = AssociativeStore(
            dim=self.DIM,
            capacity=self.N_SIMILAR + self.N_RANDOM + 10,
        )
        self._store_all(store)
        prec = self._precision_assoc(store)
        print(f"\n  AssociativeStore precision: {prec*100:.1f}%")
        assert prec > 0.95, (
            f"AssociativeStore precision {prec:.2f} < 0.95"
        )

    def test_hash_store_high_precision(self):
        store = VSAHashStore(self.vsa, num_buckets=50000)
        self._store_all(store)
        prec = self._precision_hash(store)
        print(f"\n  VSAHashStore precision: {prec*100:.1f}%")
        assert prec > 0.95, (
            f"VSAHashStore precision {prec:.2f} < 0.95"
        )

    def test_hopfield_net_collapses(self):
        # Explicit classical mode: this test demonstrates classical Hopfield's
        # attractor collapse on correlated patterns. "modern" (the current
        # HopfieldNet default) is specifically designed not to have this
        # failure mode, so it must be requested explicitly here.
        hop = HopfieldNet(dim=self.DIM, retrieval_mode="classical")
        for s in self.similar:
            hop.store(s)
        prec = self._precision_hopfield(hop)
        print(f"\n  HopfieldNet precision: {prec*100:.1f}%")
        assert prec < 0.50, (
            f"HopfieldNet precision {prec:.2f} >= 0.50 — "
            f"expected attractor collapse"
        )

    def test_random_distractors_ignored(self):
        store = VSAHashStore(self.vsa, num_buckets=50000)
        self._store_all(store)

        sims = []
        for _ in range(100):
            r = self.vsa.make_vector()
            result = store.lookup(r)
            if result is not None:
                sims.append(self.vsa.similarity(result, r).item())
        assert len(sims) == 0, (
            f"Random queries returned {len(sims)} false positives"
        )

    def test_hopfield_vs_assoc_gap(self):
        # See test_hopfield_net_collapses: classical mode is required to
        # reproduce the crosstalk this test is measuring against.
        hop = HopfieldNet(dim=self.DIM, retrieval_mode="classical")
        for s in self.similar:
            hop.store(s)
        hop_prec = self._precision_hopfield(hop)

        store = AssociativeStore(
            dim=self.DIM,
            capacity=self.N_SIMILAR + self.N_RANDOM + 10,
        )
        self._store_all(store)
        assoc_prec = self._precision_assoc(store)

        gap = assoc_prec - hop_prec
        print(f"\n  AssociativeStore: {assoc_prec*100:.1f}%")
        print(f"  HopfieldNet:     {hop_prec*100:.1f}%")
        print(f"  Gap:             {gap*100:.1f}%")
        assert gap > 0.45, (
            f"AssociativeStore should outperform HopfieldNet by >45% "
            f"on similar vectors: gap={gap:.2f}"
        )
