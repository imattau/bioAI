import torch

from src.vsa import VSA, HopfieldNet
from src.clonal import ClonalPool
from src.immune import SelfMonitor


class MultiEnvGenerator:
    def __init__(self, vsa, n_envs=5, noise=0.9):
        self.vsa = vsa
        self.dim = vsa.dim
        self.n_envs = n_envs
        self.noise = noise
        self.rules = [vsa.make_vector() for _ in range(n_envs)]
        self.current_env = 0
        self.step_count = 0
        self.switches = [(1000, 1), (2000, 2), (3000, 3), (4000, 4), (5000, 0)]

    def _sample(self, env_idx):
        return (self.rules[env_idx] + self.noise * torch.randn(self.dim)).sign()

    def step(self):
        self.step_count += 1
        for switch_step, new_env in self.switches:
            if self.step_count == switch_step:
                self.current_env = new_env
        pattern = self._sample(self.current_env)
        terminated = self.step_count >= 6000
        return pattern, self.current_env, terminated


class TestMultiEnvSwitching:
    DIM = 1000
    N_ENVS = 5
    TOTAL_STEPS = 6000
    MAX_MODULES = 15

    def setup_method(self):
        self.vsa = VSA(dim=self.DIM, device="cpu")

    def test_drift_detected_at_each_switch(self):
        gen = MultiEnvGenerator(self.vsa, n_envs=self.N_ENVS)
        hop = HopfieldNet(dim=self.DIM)
        monitor = SelfMonitor(hop, energy_threshold=3.0)

        for _ in range(100):
            hop.store(gen.rules[0])

        calibrate = []
        for _ in range(100):
            v = gen._sample(0)
            calibrate.append(v)
        monitor.calibrate(calibrate)

        for i in range(1, self.N_ENVS):
            zs = []
            for _ in range(20):
                v = gen._sample(i)
                score = monitor.score(v)
                zs.append(abs(score["energy_z"]))
            mean_z = sum(zs) / len(zs)
            print(f"\n  Env {i}: mean |energy_z| = {mean_z:.2f}")
            assert mean_z > 1.0, (
                f"Env {i} not detected: |energy_z| = {mean_z:.2f}"
            )

    def test_clonal_isolates_environments(self):
        gen = MultiEnvGenerator(self.vsa, n_envs=self.N_ENVS)
        pool = ClonalPool(input_dim=self.DIM, affinity_threshold=0.3,
                          clone_margin=0.5,
                          max_modules=self.MAX_MODULES)

        for _ in range(self.TOTAL_STEPS):
            pattern, _, _ = gen.step()
            pool.process(pattern)

        print(f"\n  Pool size: {len(pool)} (need ≥ {self.N_ENVS})")
        assert len(pool) >= self.N_ENVS, (
            f"Pool size {len(pool)} < {self.N_ENVS} environments"
        )

    def test_environment_recognized_on_return(self):
        gen = MultiEnvGenerator(self.vsa, n_envs=self.N_ENVS)
        pool = ClonalPool(input_dim=self.DIM, affinity_threshold=0.3,
                          clone_margin=0.5,
                          max_modules=self.MAX_MODULES)

        for _ in range(5000):
            pattern, _, _ = gen.step()
            pool.process(pattern)

        pattern, env_id, _ = gen.step()
        assert env_id == 0, (
            f"Expected env 0 at step 5000, got env {env_id}"
        )

        affs = [m.affinity(gen.rules[0]).item() for m in pool.modules]
        max_aff = max(affs)
        print(f"\n  Max affinity to env 0 rule on return: {max_aff:.4f}")
        assert max_aff > 0.5, (
            f"Env 0 not recognized on return: max affinity = {max_aff}"
        )

    def test_old_environment_preserved(self):
        gen = MultiEnvGenerator(self.vsa, n_envs=self.N_ENVS)
        pool = ClonalPool(input_dim=self.DIM, affinity_threshold=0.3,
                          clone_margin=0.5,
                          max_modules=self.MAX_MODULES)

        for _ in range(self.TOTAL_STEPS):
            pattern, _, _ = gen.step()
            pool.process(pattern)

        for i in range(self.N_ENVS):
            max_aff = max(
                m.affinity(gen.rules[i]).item() for m in pool.modules
            )
            print(f"\n  Env {i}: max affinity = {max_aff:.4f}")
            assert max_aff > 0.5, (
                f"Env {i} forgotten: max affinity = {max_aff}"
            )

    def test_pool_stays_bounded(self):
        gen = MultiEnvGenerator(self.vsa, n_envs=self.N_ENVS)
        pool = ClonalPool(input_dim=self.DIM, affinity_threshold=0.3,
                          clone_margin=0.5,
                          max_modules=self.MAX_MODULES)
        pool_sizes = []

        for _ in range(self.TOTAL_STEPS):
            pattern, _, _ = gen.step()
            pool.process(pattern)
            pool_sizes.append(len(pool))

        max_size = max(pool_sizes)
        print(f"\n  Max pool size: {max_size}, limit: {self.MAX_MODULES}")
        assert max_size <= self.MAX_MODULES, (
            f"Pool size {max_size} exceeded limit {self.MAX_MODULES}"
        )

    def test_immune_fingerprint_per_env(self):
        gen = MultiEnvGenerator(self.vsa, n_envs=self.N_ENVS)
        hop = HopfieldNet(dim=self.DIM)

        for _ in range(100):
            hop.store(gen._sample(0))

        energy_means = {}
        for i in range(self.N_ENVS):
            energies = []
            for _ in range(50):
                v = gen._sample(i)
                energies.append(hop.energy(v).item())
            energy_means[i] = sum(energies) / len(energies)
            print(f"\n  Mean energy for env {i}: {energy_means[i]:.2f}")

        for i in range(1, self.N_ENVS):
            diff = abs(energy_means[i] - energy_means[0])
            assert diff > 0.1 * abs(energy_means[0]), (
                f"Env {i} energy ({energy_means[i]:.2f}) too close "
                f"to env 0 ({energy_means[0]:.2f})"
            )
