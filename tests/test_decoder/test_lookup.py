import torch
from src.decoder import LookupDecoder


class TestLookupDecoder:
    def setup_method(self):
        self.decoder = LookupDecoder(vsa_dim=1000, max_seq_len=16, store_capacity=50)

    def test_ingest_and_decode(self):
        sentence = "the cat sat on the mat"
        hv = self.decoder.ingest(sentence)
        results = self.decoder.lookup(hv, k=1)
        assert len(results) == 1
        assert results[0][0] == sentence

    def test_decode_nearest(self):
        sentences = [
            "the dog runs fast in the park",
            "a bird flies high above the trees",
            "the cat sleeps on the warm sofa",
        ]
        for s in sentences:
            self.decoder.ingest(s)

        hv = self.decoder.encode("the dog runs fast in the park")
        results = self.decoder.lookup(hv, k=1)
        assert results[0][0] == "the dog runs fast in the park"

    def test_lookup_from_text(self):
        self.decoder.ingest("the big red ball")
        results = self.decoder.lookup_from_text("the big red ball", k=1)
        assert len(results) == 1
        assert results[0][0] == "the big red ball"

    def test_lookup_multiple_k(self):
        sentences = [
            "the cat sat on the mat",
            "quantum physics is fascinating",
            "pizza tastes really great",
        ]
        for s in sentences:
            self.decoder.ingest(s)

        hv = self.decoder.encode("the cat sat on the mat")
        results = self.decoder.lookup(hv, k=3)
        assert len(results) == 3
        assert results[0][0] == "the cat sat on the mat"

    def test_empty_store(self):
        hv = torch.randn(1000)
        results = self.decoder.lookup(hv, k=1)
        assert results == []

    def test_len(self):
        assert len(self.decoder) == 0
        self.decoder.ingest("hello world")
        assert len(self.decoder) == 1
        self.decoder.ingest("goodbye world")
        assert len(self.decoder) == 2

    def test_ingest_batch(self):
        sentences = ["one", "two", "three"]
        hvs = self.decoder.ingest_batch(sentences)
        assert len(hvs) == 3
        assert len(self.decoder) == 3

    def test_encode_returns_bipolar(self):
        hv = self.decoder.encode("a simple test")
        assert hv.dim() == 1
        assert hv.shape[0] == 1000
        assert torch.all((hv == 1) | (hv == -1))

    def test_ingest_batch_and_decode(self):
        sentences = [
            "the cat sat on the mat",
            "quantum physics explains reality well",
            "pizza with pineapple is delicious",
            "elephants never forget their friends",
            "the mouse eats cheese in the kitchen",
        ]
        self.decoder.ingest_batch(sentences)

        hv = self.decoder.encode("the mouse eats cheese in the kitchen")
        results = self.decoder.lookup(hv, k=1)
        assert results[0][0] == "the mouse eats cheese in the kitchen"

    def test_hopfield_cleaning(self):
        self.decoder.ingest("the cat sat on the mat")
        hv = self.decoder.encode("the cat sat on the mat")
        noisy = hv + torch.randn(1000, device=hv.device) * 0.3
        noisy = noisy.sign()
        results = self.decoder.lookup(noisy, k=1)
        assert len(results) == 1
        assert results[0][0] == "the cat sat on the mat"
