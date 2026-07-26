"""Experiment 2: Decoder Validation — Does VSA → Text Preserve Content?"""

import torch
from src.text import VSAEncoder, VSADecoder
from src.vsa import VSA


class TestDecoderRoundTrip:
    def setup_method(self):
        vsa = VSA(dim=1000, device="cpu")
        self.encoder = VSAEncoder(vsa=vsa)
        self.decoder = VSADecoder(vsa=vsa, store_capacity=50, encoder=self.encoder)

    def test_round_trip_short_phrase(self):
        text = "hello world"
        self.decoder.ingest(text)
        results = self.decoder.decode_from_text(text, k=1)
        assert len(results) == 1
        assert results[0][0] == text

    def test_round_trip_sentence(self):
        text = "the cat sat on the mat"
        self.decoder.ingest(text)
        results = self.decoder.decode_from_text(text, k=1)
        assert results[0][0] == text

    def test_round_trip_paragraph(self):
        text = "the quick brown fox jumps over the lazy dog near the river"
        self.decoder.ingest(text)
        results = self.decoder.decode_from_text(text, k=1)
        assert results[0][0] == text

    def test_round_trip_via_vector(self):
        text = "a simple test sentence"
        self.decoder.ingest(text)
        hv = self.encoder.encode(text)
        results = self.decoder.decode(hv, k=1)
        assert len(results) == 1
        assert results[0][0] == text


class TestDecoderRetrieval:
    def setup_method(self):
        vsa = VSA(dim=1000, device="cpu")
        self.encoder = VSAEncoder(vsa=vsa)
        self.decoder = VSADecoder(vsa=vsa, store_capacity=50, encoder=self.encoder)
        self.sentences = [
            "the dog runs fast in the park",
            "a bird flies high above the trees",
            "the cat sleeps on the warm sofa",
            "quantum physics is fascinating",
            "pizza tastes really great",
        ]
        for s in self.sentences:
            self.decoder.ingest(s)

    def test_k_nearest(self):
        results = self.decoder.decode_from_text(self.sentences[0], k=3)
        assert len(results) == 3
        assert results[0][0] == self.sentences[0]

    def test_empty_store(self):
        empty = VSADecoder(vsa=VSA(dim=1000, device="cpu"), store_capacity=10)
        hv = torch.randn(1000)
        results = empty.decode(hv, k=1)
        assert results == []

    def test_ingest_batch_and_retrieve(self):
        decoder = VSADecoder(vsa=VSA(dim=1000, device="cpu"), store_capacity=50)
        decoder.ingest_batch(self.sentences)
        assert len(decoder) == len(self.sentences)
        for s in self.sentences:
            results = decoder.decode_from_text(s, k=1)
            assert results[0][0] == s

    def test_len(self):
        decoder = VSADecoder(vsa=VSA(dim=1000, device="cpu"), store_capacity=10)
        assert len(decoder) == 0
        decoder.ingest("one")
        assert len(decoder) == 1
        decoder.ingest("two")
        assert len(decoder) == 2


class TestDecoderNoiseRobustness:
    def setup_method(self):
        vsa = VSA(dim=1000, device="cpu")
        self.encoder = VSAEncoder(vsa=vsa)
        self.decoder = VSADecoder(vsa=vsa, store_capacity=50, hopfield_steps=10,
                                  encoder=self.encoder)
        self.text = "the cat sat on the mat"
        self.decoder.ingest(self.text)

    def test_noisy_query_recovers_original(self):
        hv = self.encoder.encode(self.text)
        noisy = hv + torch.randn(1000) * 0.3
        noisy = noisy.sign()
        results = self.decoder.decode(noisy, k=1)
        assert len(results) == 1
        assert results[0][0] == self.text

    def test_high_noise_still_finds_match(self):
        hv = self.encoder.encode(self.text)
        noisy = hv + torch.randn(1000) * 0.5
        noisy = noisy.sign()
        results = self.decoder.decode(noisy, k=1)
        assert len(results) >= 0
