import time

import torch

from src.basal import GoNoGoActorCritic
from src.vsa import VSA, AssociativeStore
from src.vsa.hash_store import VSAHashStore


class TestDistractorResistant:
    DIM = 1000
    N_RELEVANT = 10
    N_DISTRACTORS = 90
    N_BUCKETS = 10000

    def setup_method(self):
        self.vsa = VSA(dim=self.DIM, device="cpu")
        self.relevant = [
            (self.vsa.make_vector(), self.vsa.make_vector())
            for _ in range(self.N_RELEVANT)
        ]

    def test_o1_time_independent_of_distractors(self):
        times = {}
        for n_dist in [10, 90]:
            store = VSAHashStore(self.vsa, num_buckets=self.N_BUCKETS)
            dist = [
                (self.vsa.make_vector(), self.vsa.make_vector())
                for _ in range(n_dist)
            ]
            for k, v in self.relevant + dist:
                store.insert(k, v)

            store.lookup(self.relevant[0][0])
            start = time.perf_counter()
            for _ in range(1000):
                store.lookup(self.relevant[0][0])
            times[n_dist] = (time.perf_counter() - start) / 1000

        ratio = times[90] / times[10]
        print(f"\n  Time @ 10 distractors: {times[10]*1e3:.4f} ms")
        print(f"  Time @ 90 distractors: {times[90]*1e3:.4f} ms")
        print(f"  Ratio: {ratio:.2f}x")
        assert ratio < 1.5, (
            f"Lookup time ratio {ratio:.2f}x exceeds 1.5x — "
            f"not O(1)"
        )

    def test_all_relevant_facts_retrievable(self):
        store = VSAHashStore(self.vsa, num_buckets=self.N_BUCKETS)
        dist = [
            (self.vsa.make_vector(), self.vsa.make_vector())
            for _ in range(self.N_DISTRACTORS)
        ]
        for k, v in self.relevant + dist:
            store.insert(k, v)

        for i, (k, v) in enumerate(self.relevant):
            result = store.lookup(k)
            assert result is not None, f"Relevant fact {i} returned None"
            sim = self.vsa.similarity(result, v).item()
            assert sim > 0.99, (
                f"Relevant fact {i}: cos_sim={sim}"
            )

    def test_no_false_positives(self):
        store = VSAHashStore(self.vsa, num_buckets=self.N_BUCKETS)
        dist = [
            (self.vsa.make_vector(), self.vsa.make_vector())
            for _ in range(self.N_DISTRACTORS)
        ]
        for k, v in self.relevant + dist:
            store.insert(k, v)

        for _ in range(10):
            result = store.lookup(self.vsa.make_vector())
            assert result is None, (
                f"Random query returned non-None result"
            )

    def test_go_nogo_selects_from_store(self):
        store = VSAHashStore(self.vsa, num_buckets=self.N_BUCKETS)
        dist = [
            (self.vsa.make_vector(), self.vsa.make_vector())
            for _ in range(self.N_DISTRACTORS)
        ]
        for k, v in self.relevant + dist:
            store.insert(k, v)

        agent = GoNoGoActorCritic(
            input_dim=self.DIM, n_actions=self.N_RELEVANT
        )

        candidates = [v for _, v in self.relevant]
        for query_key, query_val in self.relevant:
            result = store.lookup(query_key)
            assert result is not None

            sims = torch.tensor([
                self.vsa.similarity(result, c).item() for c in candidates
            ])
            best_idx = sims.argmax().item()
            action, _ = agent.act(result)
            assert 0 <= action < self.N_RELEVANT

    def test_collision_handling(self):
        small_store = VSAHashStore(self.vsa, num_buckets=10)
        for k, v in self.relevant:
            small_store.insert(k, v)

        for i, (k, v) in enumerate(self.relevant):
            result = small_store.lookup(k)
            assert result is not None, (
                f"Relevant fact {i} returned None with collisions"
            )
            sim = self.vsa.similarity(result, v).item()
            assert sim > 0.99, (
                f"Relevant fact {i} with collisions: cos_sim={sim}"
            )

    def test_vs_associative_store_speed(self):
        store_hash = VSAHashStore(self.vsa, num_buckets=self.N_BUCKETS)
        store_assoc = AssociativeStore(
            dim=self.DIM, capacity=self.N_RELEVANT + self.N_DISTRACTORS + 10
        )
        all_pairs = self.relevant + [
            (self.vsa.make_vector(), self.vsa.make_vector())
            for _ in range(self.N_DISTRACTORS)
        ]

        for k, v in all_pairs:
            store_hash.insert(k, v)
            store_assoc.insert(k, v)

        qk = self.relevant[0][0]

        store_hash.lookup(qk)
        start = time.perf_counter()
        for _ in range(1000):
            store_hash.lookup(qk)
        t_hash = (time.perf_counter() - start) / 1000

        store_assoc.lookup(qk, k=1)
        start = time.perf_counter()
        for _ in range(1000):
            store_assoc.lookup(qk, k=1)
        t_assoc = (time.perf_counter() - start) / 1000

        print(f"\n  VSAHashStore:     {t_hash*1e3:.4f} ms")
        print(f"  AssociativeStore: {t_assoc*1e3:.4f} ms")
        print(f"  Speedup:          {t_assoc / t_hash:.0f}×")
        assert t_hash <= t_assoc * 1.5, (
            f"VSAHashStore ({t_hash*1e3:.4f} ms) not close to "
            f"AssociativeStore ({t_assoc*1e3:.4f} ms)"
        )

    def test_vs_associative_store_scaling(self):
        times_hash = {}
        times_assoc = {}
        for n_dist in [10, 90]:
            h = VSAHashStore(self.vsa, num_buckets=self.N_BUCKETS)
            a = AssociativeStore(
                dim=self.DIM,
                capacity=self.N_RELEVANT + n_dist + 10,
            )
            pairs = self.relevant + [
                (self.vsa.make_vector(), self.vsa.make_vector())
                for _ in range(n_dist)
            ]
            for k, v in pairs:
                h.insert(k, v)
                a.insert(k, v)

            qk = self.relevant[0][0]

            h.lookup(qk)
            start = time.perf_counter()
            for _ in range(1000):
                h.lookup(qk)
            times_hash[n_dist] = (time.perf_counter() - start) / 1000

            a.lookup(qk, k=1)
            start = time.perf_counter()
            for _ in range(1000):
                a.lookup(qk, k=1)
            times_assoc[n_dist] = (time.perf_counter() - start) / 1000

        hash_ratio = times_hash[90] / times_hash[10]
        assoc_ratio = times_assoc[90] / times_assoc[10]

        print(f"\n  VSAHashStore ratio (90/10): {hash_ratio:.2f}x")
        print(f"  AssociativeStore ratio (90/10): {assoc_ratio:.2f}x")
        assert hash_ratio < assoc_ratio, (
            f"VSAHashStore scales worse than AssociativeStore: "
            f"hash={hash_ratio:.2f}x vs assoc={assoc_ratio:.2f}x"
        )
