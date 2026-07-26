"""Experiment 3: Full Pipeline — Encoder + Reasoning Engine + Decoder"""

import torch
from src.text import VSAEncoder, VSADecoder
from src.vsa import VSA, VSAHashStore, AssociativeStore, HopfieldNet
from src.clonal import ClonalPool
from src.immune import SelfMonitor


class TestEncodeStoreRetrieve:
    def setup_method(self):
        vsa = VSA(dim=1000, device="cpu")
        self.encoder = VSAEncoder(vsa=vsa)
        self.decoder = VSADecoder(vsa=vsa, store_capacity=50, encoder=self.encoder)
        self.store = VSAHashStore(vsa)

    def test_encode_store_hashstore_retrieve(self):
        text = "the capital of France is Paris"
        hv = self.encoder.encode(text)
        self.store.insert(hv, hv)
        result = self.store.lookup(hv)
        assert result is not None
        sim = self.encoder.vsa.similarity(result, hv).item()
        assert sim > 0.99, f"HashStore round-trip sim={sim}"

    def test_encode_store_lookup_decode(self):
        text = "Paris is the capital of France"
        self.decoder.ingest(text)
        hv = self.encoder.encode(text)
        self.store.insert(hv, hv)
        retrieved = self.store.lookup(hv)
        assert retrieved is not None
        results = self.decoder.decode(retrieved, k=1)
        assert len(results) == 1
        assert results[0][0] == text


class TestLongContextRetrieval:
    def setup_method(self):
        vsa = VSA(dim=1000, device="cpu")
        self.encoder = VSAEncoder(vsa=vsa)
        self.decoder = VSADecoder(vsa=vsa, store_capacity=200, encoder=self.encoder)
        self.store = VSAHashStore(vsa)
        self.facts = [f"Fact {i}: The value is {i * 2}" for i in range(200)]
        for fact in self.facts:
            hv = self.encoder.encode(fact)
            self.store.insert(hv, hv)
            self.decoder.ingest(fact)

    def test_retrieve_fact_from_middle(self):
        target_idx = 100
        hv = self.encoder.encode(self.facts[target_idx])
        retrieved = self.store.lookup(hv)
        assert retrieved is not None
        results = self.decoder.decode(retrieved, k=1)
        assert len(results) == 1
        assert results[0][0] == self.facts[target_idx]

    def test_retrieve_fact_850_style(self):
        target_idx = 170
        hv = self.encoder.encode(self.facts[target_idx])
        retrieved = self.store.lookup(hv)
        assert retrieved is not None
        results = self.decoder.decode(retrieved, k=1)
        assert results[0][0] == self.facts[target_idx]


class TestDriftDetection:
    def setup_method(self):
        vsa = VSA(dim=256, device="cpu")
        self.encoder = VSAEncoder(vsa=vsa)
        self.hop = HopfieldNet(dim=256)

    def test_monitor_detects_ood_text(self):
        normal_texts = [
            "the cat sat on the mat",
            "dog runs in the park",
            "birds fly in the sky",
        ]
        normal_vecs = [self.encoder.encode(t) for t in normal_texts]
        for v in normal_vecs:
            self.hop.store(v)
        monitor = SelfMonitor(self.hop, energy_threshold=3.0)
        recalls = [self.hop.recall(v + 0.1 * torch.randn(256), steps=10)
                   for v in normal_vecs]
        monitor.calibrate(recalls)

        ood_text = "quantum chromodynamics explains quark confinement"
        ood_vec = self.encoder.encode(ood_text)
        result = monitor.score(self.hop.recall(ood_vec, steps=10))
        assert "is_anomaly" in result

    def test_normal_text_not_flagged(self):
        texts = ["a simple sentence", "another test here"]
        vecs = [self.encoder.encode(t) for t in texts]
        for v in vecs:
            self.hop.store(v)
        monitor = SelfMonitor(self.hop, energy_threshold=3.0)
        recalls = [self.hop.recall(v + 0.1 * torch.randn(256), steps=10)
                   for v in vecs]
        monitor.calibrate(recalls)
        clean = self.hop.recall(vecs[0] + 0.1 * torch.randn(256), steps=10)
        result = monitor.score(clean)
        assert not result["is_anomaly"]


class TestClonalLearning:
    def setup_method(self):
        vsa = VSA(dim=1000, device="cpu")
        self.encoder = VSAEncoder(vsa=vsa)
        self.pool = ClonalPool(input_dim=1000, affinity_threshold=0.4, max_modules=20)

    def test_learns_text_patterns(self):
        texts = [
            "cat sat on mat",
            "dog runs in park",
            "bird flies in sky",
        ]
        for t in texts:
            hv = self.encoder.encode(t)
            self.pool.process(hv, lr=0.01)

        novel = self.encoder.encode("quantum physics is deep")
        _, created = self.pool.process(novel, lr=0.01)
        assert created, "Novel text should trigger pool expansion"

    def test_similar_text_matches(self):
        base = self.encoder.encode("the cat sat on the mat")
        self.pool.process(base, lr=0.01)
        similar = self.encoder.encode("the cat sat on the rug")
        _, created = self.pool.process(similar, lr=0.01)
        affs = [m.affinity(similar) for m in self.pool.modules]
        best_aff = max(affs).item() if affs else 0.0
        assert best_aff > 0.3, f"Similar text should match: best_aff={best_aff:.4f}"
