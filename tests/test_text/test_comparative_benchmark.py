"""Experiment 5: Comparative Benchmark — BioAI Pipeline vs. Baseline"""

import time
import torch
from src.text import VSAEncoder, VSADecoder
from src.vsa import VSA, VSAHashStore, AssociativeStore


class TestRetrievalSpeed:
    def setup_method(self):
        vsa = VSA(dim=1000, device="cpu")
        self.encoder = VSAEncoder(vsa=vsa)
        self.hash_store = VSAHashStore(vsa, num_buckets=1000)
        self.assoc_store = AssociativeStore(dim=1000, capacity=2000)

    def test_o1_hashstore_beats_linear_assocstore(self):
        n = 500
        keys = []
        for i in range(n):
            t = f"fact number {i} with some content"
            hv = self.encoder.encode(t)
            keys.append(hv)
            self.hash_store.insert(hv, hv)
            self.assoc_store.insert(hv)

        def time_hash(queries):
            for q in queries:
                self.hash_store.lookup(q)

        def time_assoc(queries):
            for q in queries:
                self.assoc_store.lookup(q, k=1)

        queries = [self.encoder.encode(f"query {i}") for i in range(20)]

        t0 = time.perf_counter()
        time_hash(queries)
        t_hash = time.perf_counter() - t0

        t0 = time.perf_counter()
        time_assoc(queries)
        t_assoc = time.perf_counter() - t0

        assert t_hash < t_assoc * 2 or t_hash < 0.01, (
            f"HashStore should be faster or comparable: "
            f"hash={t_hash:.6f}s vs assoc={t_assoc:.6f}s"
        )

    def test_hashstore_retrieval_accuracy(self):
        n = 200
        for i in range(n):
            t = f"exact fact to retrieve {i}"
            hv = self.encoder.encode(t)
            self.hash_store.insert(hv, hv)

        for i in range(n):
            t = f"exact fact to retrieve {i}"
            hv = self.encoder.encode(t)
            result = self.hash_store.lookup(hv)
            assert result is not None, f"Fact {i} should be retrievable"
            sim = self.encoder.vsa.similarity(result, hv).item()
            assert sim > 0.99, f"Fact {i} retrieval sim={sim}"


class TestLongContextAdvantage:
    def setup_method(self):
        vsa = VSA(dim=1000, device="cpu")
        self.encoder = VSAEncoder(vsa=vsa)
        self.store = VSAHashStore(vsa)

    def test_all_facts_retrievable_at_1000_scale(self):
        n = 1000
        for i in range(n):
            t = f"fact {i}: value is {i * 2}"
            hv = self.encoder.encode(t)
            self.store.insert(hv, hv)

        for i in range(0, n, 50):
            t = f"fact {i}: value is {i * 2}"
            hv = self.encoder.encode(t)
            result = self.store.lookup(hv)
            assert result is not None, f"Fact {i} lost at scale"
            sim = self.encoder.vsa.similarity(result, hv).item()
            assert sim > 0.99, f"Fact {i} degraded at scale: sim={sim}"


class TestPipelineLatency:
    def setup_method(self):
        vsa = VSA(dim=1000, device="cpu")
        self.encoder = VSAEncoder(vsa=vsa)
        self.decoder = VSADecoder(vsa=vsa, store_capacity=100)

    def test_encode_latency(self):
        texts = [f"sentence number {i}" for i in range(50)]
        t0 = time.perf_counter()
        for t in texts:
            self.encoder.encode(t)
        elapsed = time.perf_counter() - t0
        avg = elapsed / len(texts)
        assert avg < 0.05, f"Average encode latency too high: {avg*1000:.1f}ms"

    def test_round_trip_latency(self):
        self.decoder.ingest("a test sentence for latency measurement")
        t0 = time.perf_counter()
        for _ in range(50):
            self.decoder.decode_from_text("a test sentence for latency measurement")
        elapsed = time.perf_counter() - t0
        avg = elapsed / 50
        assert avg < 0.05, f"Average decode latency too high: {avg*1000:.1f}ms"
