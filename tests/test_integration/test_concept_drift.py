import torch
from src.vsa import VSA, AssociativeStore, HopfieldNet, \
    GridCellPositionalEncoder, PositionalVSAStore
from src.clonal import ClonalPool
from src.immune import SelfMonitor


class TestGradualConceptDrift:
    DIM = 1000
    NUM_STEPS = 20000
    DRIFT_STEP = 15000

    def setup_method(self):
        self.vsa = VSA(dim=self.DIM, device="cpu")

        self.rule_A = self.vsa.make_vector()
        self.rule_B = self.vsa.make_vector()

        encoder = GridCellPositionalEncoder(self.vsa, self.DIM, chunk_size=1)
        self.time_store = PositionalVSAStore(encoder, self.DIM)
        self.store = AssociativeStore(dim=self.DIM, capacity=self.NUM_STEPS + 1000)

        self.pool = ClonalPool(
            input_dim=self.DIM, affinity_threshold=0.3, max_modules=10,
        )
        self.hop = HopfieldNet(dim=self.DIM)
        self.monitor = SelfMonitor(self.hop, energy_threshold=3.0)

    def run_phase1(self):
        for step in range(self.DRIFT_STEP):
            self.time_store.insert(step, self.rule_A)
            self.store.insert(self.rule_A, self.vsa.make_vector())
            self.pool.process(self.rule_A)

    def run_phase2(self):
        drift_detected = False
        for step in range(self.DRIFT_STEP, self.NUM_STEPS):
            self.time_store.insert(step, self.rule_B)
            self.store.insert(self.rule_B, self.vsa.make_vector())
            _, created = self.pool.process(self.rule_B)
            if created:
                drift_detected = True
        return drift_detected

    def test_drift_detection(self):
        self.run_phase1()
        drift_detected = self.run_phase2()
        assert drift_detected, "ClonalPool did not spawn a new module at rule transition"

    def test_current_step_returns_new_rule(self):
        self.run_phase1()
        self.run_phase2()
        result = self.time_store.query(self.NUM_STEPS - 1)
        assert result is not None
        sim = self.vsa.similarity(result, self.rule_B).item()
        assert sim > 0.99, f"Current step returned rule_A: cos_sim={sim}"

    def test_historical_step_returns_old_rule(self):
        self.run_phase1()
        self.run_phase2()
        result = self.time_store.query(0)
        assert result is not None
        sim = self.vsa.similarity(result, self.rule_A).item()
        assert sim > 0.99, f"Historical step returned wrong rule: cos_sim={sim}"

    def test_drift_point_returns_new_rule(self):
        self.run_phase1()
        self.run_phase2()
        result = self.time_store.query(self.DRIFT_STEP)
        assert result is not None
        sim = self.vsa.similarity(result, self.rule_B).item()
        assert sim > 0.99, f"Drift step returned rule_A: cos_sim={sim}"

    def test_pre_drift_point_returns_old_rule(self):
        self.run_phase1()
        self.run_phase2()
        result = self.time_store.query(self.DRIFT_STEP - 1)
        assert result is not None
        sim = self.vsa.similarity(result, self.rule_A).item()
        assert sim > 0.99, f"Pre-drift step returned rule_B: cos_sim={sim}"

    def test_pool_has_both_rules(self):
        self.run_phase1()
        self.run_phase2()
        assert len(self.pool) >= 2, f"Pool has {len(self.pool)} modules, expected >= 2"

    def test_no_memory_overwrite(self):
        self.run_phase1()
        self.run_phase2()
        assert len(self.time_store) == self.NUM_STEPS, (
            f"Store has {len(self.time_store)} items, expected {self.NUM_STEPS}"
        )

    def test_immune_detects_drift_as_anomaly(self):
        calibrate_patterns = []
        for _ in range(100):
            y = self.vsa.bind(self.vsa.make_vector(), self.rule_A)
            self.hop.store(y)
            calibrate_patterns.append(
                self.hop.recall(y + 0.15 * torch.randn(self.DIM), steps=10)
            )
        self.monitor.calibrate(calibrate_patterns)

        drift_z_scores = []
        for _ in range(20):
            test_B = self.vsa.bind(self.vsa.make_vector(), self.rule_B)
            test_B_recalled = self.hop.recall(
                test_B + 0.15 * torch.randn(self.DIM), steps=10
            )
            score = self.monitor.score(test_B_recalled)
            drift_z_scores.append(abs(score["energy_z"]))

        mean_abs_z = sum(drift_z_scores) / len(drift_z_scores)
        print(f"\n  Mean |energy_z| for rule_B: {mean_abs_z:.2f}")
        assert mean_abs_z > 1.0, (
            f"Rule_B energy_z ({mean_abs_z:.2f}) too close to rule_A baseline"
        )

    def test_immunes_energy_separates_rules(self):
        for _ in range(50):
            y = self.vsa.bind(self.vsa.make_vector(), self.rule_A)
            self.hop.store(y)

        energies_A = []
        for _ in range(20):
            y = self.vsa.bind(self.vsa.make_vector(), self.rule_A)
            rec = self.hop.recall(y + 0.15 * torch.randn(self.DIM), steps=10)
            energies_A.append(self.hop.energy(rec.flatten()).item())

        energies_B = []
        for _ in range(20):
            y = self.vsa.bind(self.vsa.make_vector(), self.rule_B)
            rec = self.hop.recall(y + 0.15 * torch.randn(self.DIM), steps=10)
            energies_B.append(self.hop.energy(rec.flatten()).item())

        mean_A = sum(energies_A) / len(energies_A)
        mean_B = sum(energies_B) / len(energies_B)
        diff = abs(mean_B - mean_A)
        print(f"\n  Mean energy A: {mean_A:.1f}, B: {mean_B:.1f}, diff: {diff:.1f}")
        assert diff > max(mean_A * 0.1, 0.5), f"Energy difference too small: {diff}"
