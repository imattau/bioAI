import time

import torch

from src.vsa import VSA, AssociativeStore, GridCellPositionalEncoder, PositionalVSAStore


class TestNeedleChain:
    DIM = 10000
    CHAIN_LEN = 200

    def setup_method(self):
        self.vsa = VSA(dim=self.DIM, device="cpu")
        self.encoder = GridCellPositionalEncoder(self.vsa, self.DIM, chunk_size=1)
        self.entities = [self.vsa.make_vector() for _ in range(self.CHAIN_LEN + 1)]

    def _build_positional_store(self, n_distractors: int = 0):
        store = PositionalVSAStore(self.encoder, self.DIM)
        for i in range(self.CHAIN_LEN):
            transition = self.vsa.bind(self.entities[i], self.entities[i + 1])
            store.insert(i, transition)
        for d in range(n_distractors):
            store.insert(1000 + d, self.vsa.make_vector())
        return store

    def _build_assoc_store(self, n_distractors: int = 0):
        store = AssociativeStore(dim=self.DIM, capacity=2000)
        for i in range(self.CHAIN_LEN):
            key = self.vsa.make_vector()
            transition = self.vsa.bind(self.entities[i], self.entities[i + 1])
            store.insert(key, transition)
        for d in range(n_distractors):
            store.insert(self.vsa.make_vector(), self.vsa.make_vector())
        return store

    def _walk_chain(self, store, start_pos=0, steps=None):
        if steps is None:
            steps = self.CHAIN_LEN - start_pos
        current = self.entities[start_pos]
        for i in range(start_pos, start_pos + steps):
            result = store.query(i)
            if result is None:
                return None
            current = self.vsa.bind(result, current)
        return current

    def test_chain_accuracy(self):
        store = self._build_positional_store()
        result = self._walk_chain(store)
        assert result is not None
        sim = self.vsa.similarity(result, self.entities[self.CHAIN_LEN]).item()
        print(f"\n  Final cos_sim @ 200 steps: {sim:.8f}")
        assert sim > 0.999, f"Chain accuracy {sim} < 0.999"

    def test_o1_retrieval_independent(self):
        times = {}
        for n_dist in [0, 500, 1000]:
            store = self._build_positional_store(n_dist)
            store.query(0)
            start = time.perf_counter()
            for _ in range(100):
                store.query(50)
            times[n_dist] = (time.perf_counter() - start) / 100

        ratio = times[1000] / times[0]
        print(f"\n  Time @ 0 distractors:  {times[0]*1e6:.3f} µs")
        print(f"  Time @ 1000 distractors: {times[1000]*1e6:.3f} µs")
        print(f"  Ratio: {ratio:.2f}x")
        assert ratio < 2.0, (
            f"Retrieval time ratio {ratio:.2f}x exceeds 2x — not O(1)"
        )

    def test_chain_vs_assoc_baseline(self):
        n_dist = 500
        pos_store = self._build_positional_store(n_dist)
        assoc_store = self._build_assoc_store(n_dist)

        current = self.entities[0]
        start = time.perf_counter()
        for i in range(self.CHAIN_LEN):
            result = pos_store.query(i)
            if result is not None:
                current = self.vsa.bind(result, current)
        pos_time = time.perf_counter() - start

        current = self.entities[0]
        start = time.perf_counter()
        for i in range(self.CHAIN_LEN):
            results = assoc_store.lookup(assoc_store.keys[i], k=1)
            if results:
                current = self.vsa.bind(results[0][0], current)
        assoc_time = time.perf_counter() - start

        print(f"\n  PositionalVSAStore chain: {pos_time*1e3:.2f} ms")
        print(f"  AssociativeStore chain:   {assoc_time*1e3:.2f} ms")
        print(f"  Speedup: {assoc_time / pos_time:.0f}×")
        assert pos_time < assoc_time, (
            f"Positional store ({pos_time*1e3:.2f} ms) not faster than "
            f"AssociativeStore ({assoc_time*1e3:.2f} ms)"
        )

    def test_partial_chain_correct(self):
        store = self._build_positional_store()
        for mid in [10, 50, 100, 150]:
            result = self._walk_chain(store, start_pos=0, steps=mid)
            assert result is not None
            sim = self.vsa.similarity(result, self.entities[mid]).item()
            assert sim > 0.999, (
                f"Partial chain @ {mid} steps: cos_sim={sim}"
            )

    def test_wrong_starting_entity_fails(self):
        store = self._build_positional_store()
        wrong_start = self.vsa.make_vector()
        current = wrong_start
        for i in range(self.CHAIN_LEN):
            result = store.query(i)
            if result is not None:
                current = self.vsa.bind(result, current)
        sim = self.vsa.similarity(current, self.entities[self.CHAIN_LEN]).item()
        print(f"\n  Wrong start cos_sim to chain end: {sim:.4f}")
        assert sim < 0.5, (
            f"Wrong starting entity produced correct endpoint: sim={sim}"
        )
