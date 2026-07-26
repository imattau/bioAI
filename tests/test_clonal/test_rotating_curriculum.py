import time

import torch

from src.clonal import ClonalPool
from src.vsa import VSA


class TestRotatingCurriculum:
    DIM = 1000
    N_CLASSES = 10
    EXAMPLES_PER_CLASS = 1000
    NOISE = 0.9
    MAX_MODULES = 15

    def setup_method(self):
        self.vsa = VSA(dim=self.DIM, device="cpu")
        self.classes = [self.vsa.make_vector() for _ in range(self.N_CLASSES)]
        self.pool = ClonalPool(
            input_dim=self.DIM,
            affinity_threshold=0.3,
            clone_margin=0.5,
            max_modules=self.MAX_MODULES,
        )

    def _sample(self, class_idx: int) -> torch.Tensor:
        return (
            self.classes[class_idx]
            + self.NOISE * torch.randn(self.DIM)
        ).sign()

    def _train_phase(self, start_idx: int, end_idx: int):
        for ci in range(start_idx, end_idx):
            for _ in range(self.EXAMPLES_PER_CLASS):
                self.pool.process(self._sample(ci))

    def _max_affinity(self, class_idx: int) -> float:
        return max(
            m.affinity(self.classes[class_idx]).item()
            for m in self.pool.modules
        )

    def test_old_classes_preserved(self):
        self._train_phase(0, 5)
        aff_before = {ci: self._max_affinity(ci) for ci in range(5)}

        self._train_phase(5, 7)
        self._train_phase(7, 10)

        aff_after = {ci: self._max_affinity(ci) for ci in range(5)}

        max_deg = 0.0
        for ci in range(5):
            deg = (
                max(0, aff_before[ci] - aff_after[ci]) / aff_before[ci] * 100
                if aff_before[ci] > 0.5
                else 0
            )
            max_deg = max(max_deg, deg)

        print(f"\n  Max degradation (classes 0-4): {max_deg:.2f}%")
        assert max_deg < 2.0, (
            f"Degradation {max_deg:.2f}% exceeds 2% threshold"
        )

    def test_pool_stays_bounded(self):
        sizes = []

        self._train_phase(0, 5)
        sizes.append(len(self.pool))

        self._train_phase(5, 7)
        sizes.append(len(self.pool))

        self._train_phase(7, 10)
        sizes.append(len(self.pool))

        print(f"\n  Pool sizes after each phase: {sizes}")
        for i, s in enumerate(sizes):
            assert s <= self.MAX_MODULES, (
                f"Pool size {s} exceeded max {self.MAX_MODULES} "
                f"at phase {i+1}"
            )

    def test_all_classes_learned(self):
        self._train_phase(0, 5)
        self._train_phase(5, 7)
        self._train_phase(7, 10)

        for ci in range(self.N_CLASSES):
            aff = self._max_affinity(ci)
            assert aff > 0.5, (
                f"Class {ci} has affinity {aff:.4f} — below 0.5"
            )

    def test_inference_speed_constant(self):
        def time_n_calls(n: int) -> float:
            start = time.perf_counter()
            for _ in range(n):
                self.pool.process(self._sample(0))
            return (time.perf_counter() - start) / n

        warmup = time_n_calls(50)

        self._train_phase(0, 5)
        mid_time = time_n_calls(100)

        self._train_phase(5, 7)
        self._train_phase(7, 10)
        final_time = time_n_calls(100)

        print(f"\n  Warmup time per call: {warmup*1e3:.4f} ms")
        print(f"  Mid time per call:    {mid_time*1e3:.4f} ms")
        print(f"  Final time per call:  {final_time*1e3:.4f} ms")

        if mid_time > 0 and final_time > 0:
            ratio = final_time / max(mid_time, 1e-9)
            print(f"  Final/Mid ratio: {ratio:.2f}x")
            assert ratio < 3.0, (
                f"Inference speed ratio {ratio:.2f}x exceeds 3x"
            )

    def test_sequential_phase_no_degradation(self):
        self._train_phase(0, 5)
        aff_phase1 = {ci: self._max_affinity(ci) for ci in range(5)}

        self._train_phase(5, 7)
        aff_phase2 = {ci: self._max_affinity(ci) for ci in range(5)}

        self._train_phase(7, 10)
        aff_phase3 = {ci: self._max_affinity(ci) for ci in range(5)}

        max_drop = 0.0
        for ci in range(5):
            affs = [aff_phase1[ci], aff_phase2[ci], aff_phase3[ci]]
            roll = max(affs) - min(affs)
            max_drop = max(max_drop, roll)

        print(f"\n  Max affinity fluctuation (C0-4 across phases): {max_drop:.4f}")
        assert max_drop < 0.05, (
            f"Affinity varied by {max_drop:.4f} across phases"
        )

    def test_pruning_preserves_important(self):
        small_pool = ClonalPool(
            input_dim=self.DIM,
            affinity_threshold=0.3,
            clone_margin=0.5,
            max_modules=7,
        )

        for ci in range(5):
            for _ in range(self.EXAMPLES_PER_CLASS):
                small_pool.process(self._sample(ci))

        for ci in range(5, 10):
            for _ in range(self.EXAMPLES_PER_CLASS):
                small_pool.process(self._sample(ci))

        assert len(small_pool) <= 7, (
            f"Pool size {len(small_pool)} exceeds 7"
        )

        for ci in range(5):
            aff = max(
                m.affinity(self.classes[ci]).item()
                for m in small_pool.modules
            )
            assert aff > 0.5, (
                f"Early class {ci} lost after pruning: affinity={aff:.4f}"
            )
