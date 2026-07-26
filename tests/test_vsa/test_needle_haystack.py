import gc
import time

import torch
import torchhd

from src.vsa import VSA, GridCellPositionalEncoder, PositionalVSAStore


class TestNeedleHaystack:
    DIM = 10000
    NUM_ITEMS = 100000
    PASSWORD_POS = 50000

    def setup_method(self):
        self.vsa = VSA(dim=self.DIM)
        self.encoder = GridCellPositionalEncoder(self.vsa, self.DIM, chunk_size=1)
        self.store = PositionalVSAStore(self.encoder, self.DIM)

    def teardown_method(self):
        del self.store
        del self.encoder
        del self.vsa
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_perfect_retrieval_at_100k(self):
        password_value = self.vsa.make_vector()

        with torch.no_grad():
            for pos in range(self.NUM_ITEMS):
                v = password_value if pos == self.PASSWORD_POS else self.vsa.make_vector()
                self.store.insert(pos, v)

        result = self.store.query(self.PASSWORD_POS)
        assert result is not None
        sim = self.vsa.similarity(result, password_value).item()
        print(f"\n  Password retrieval cos_sim = {sim:.8f}")
        assert sim > 0.9999, f"cos_sim={sim} — VSA binding broken at 100k scale"

    def test_retrieval_time_constant(self):
        with torch.no_grad():
            for pos in range(self.NUM_ITEMS):
                self.store.insert(pos, self.vsa.make_vector())

        for _ in range(10):
            self.store.query(self.PASSWORD_POS)

        start = time.perf_counter()
        for _ in range(1000):
            self.store.query(self.PASSWORD_POS)
        t = (time.perf_counter() - start) / 1000
        print(f"\n  PositionalVSAStore query: {t*1e6:.1f} µs")
        assert t < 0.001, f"Query too slow: {t*1e3:.2f} ms"

    def test_linear_scan_baseline(self):
        sample_size = 10000

        with torch.no_grad():
            for pos in range(sample_size):
                self.store.insert(pos, self.vsa.make_vector())

        _, pos_in_chunk = self.encoder.position_to_chunk(0)
        target = self.encoder.encode_fine(pos_in_chunk).cpu()

        items = []
        with torch.no_grad():
            for pos in range(sample_size):
                _, p = self.encoder.position_to_chunk(pos)
                items.append(self.encoder.encode_fine(p).cpu())

        keys = torch.stack(items)

        _ = torchhd.cosine_similarity(target.unsqueeze(0), keys)

        start = time.perf_counter()
        for _ in range(100):
            _ = torchhd.cosine_similarity(target.unsqueeze(0), keys)
        linear_t = (time.perf_counter() - start) / 100

        _ = self.store.query(0)
        start = time.perf_counter()
        for _ in range(10000):
            self.store.query(0)
        pos_t = (time.perf_counter() - start) / 10000

        print(f"\n  Linear scan ({sample_size} items):      {linear_t*1e3:.4f} ms")
        print(f"  PositionalVSA ({sample_size} items):    {pos_t*1e3:.4f} ms")
        print(f"  Speedup:         {linear_t / pos_t:.0f}×")
        assert pos_t < linear_t, "Positional query should be faster than linear scan"

    def test_query_empty_position(self):
        assert self.store.query(999999) is None

    def test_all_items_stored(self):
        with torch.no_grad():
            for pos in range(self.NUM_ITEMS):
                self.store.insert(pos, self.vsa.make_vector())
        assert len(self.store) == self.NUM_ITEMS

    def test_different_positions_different_results(self):
        with torch.no_grad():
            for pos in range(100):
                self.store.insert(pos, self.vsa.make_vector())

        r0 = self.store.query(0)
        r1 = self.store.query(50)
        assert r0 is not None and r1 is not None
        sim = self.vsa.similarity(r0, r1).item()
        assert sim < 0.5, f"Different positions too similar: sim={sim}"

    def test_retrieved_vector_valid(self):
        with torch.no_grad():
            for pos in range(100):
                self.store.insert(pos, self.vsa.make_vector())

        result = self.store.query(50)
        assert result is not None
        assert result.shape == (self.DIM,)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        assert result.norm().item() > 0

    def test_password_wrong_position_fails(self):
        password_value = self.vsa.make_vector()

        with torch.no_grad():
            for pos in range(self.NUM_ITEMS):
                v = password_value if pos == self.PASSWORD_POS else self.vsa.make_vector()
                self.store.insert(pos, v)

        wrong = self.store.query(0)
        assert wrong is not None
        sim = self.vsa.similarity(wrong, password_value).item()
        assert sim < 0.2, f"Wrong position returned password: sim={sim}"

    def test_insert_then_immediate_query(self):
        v = self.vsa.make_vector()
        self.store.insert(0, v)
        result = self.store.query(0)
        assert result is not None
        sim = self.vsa.similarity(result, v).item()
        assert sim > 0.9999, f"Single insert/query: cos_sim={sim}"

    def test_retrieval_of_multiple_known_positions(self):
        known = {}

        with torch.no_grad():
            for pos in range(self.NUM_ITEMS):
                v = self.vsa.make_vector()
                self.store.insert(pos, v)
                if pos % 25000 == 0:
                    known[pos] = v

        for pos, expected in known.items():
            result = self.store.query(pos)
            assert result is not None, f"pos={pos} returned None"
            sim = self.vsa.similarity(result, expected).item()
            assert sim > 0.9999, f"pos={pos}: cos_sim={sim}"
