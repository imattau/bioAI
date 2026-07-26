import time

import torch

from src.basal import GoNoGoActorCritic
from src.clonal import ClonalPool
from src.immune import SelfMonitor
from src.vsa import VSA, HopfieldNet, VSAHashStore


class DriftingBanditChallenge:
    n_arms: int

    def __init__(self, n_arms=4):
        self.n_arms = n_arms
        self.best_arm = 0
        self.step_count = 0
        self.drifts = [(700, 2), (1400, 3), (2100, 0), (3000, 1)]

    def reset(self):
        self.step_count = 0
        self.best_arm = 0
        return self.best_arm

    def step(self, action):
        if not isinstance(action, int):
            action = action.item() if hasattr(action, 'item') else action
        correct = 1.0 if action == self.best_arm else 0.0
        noisy = 1.0 if torch.rand(1).item() < 0.75 else 0.0
        reward = 1.0 if (correct == 1.0 and noisy == 1.0) or (correct == 0.0 and noisy == 0.0) else 0.0
        self.step_count += 1
        for switch_step, new_arm in self.drifts:
            if self.step_count == switch_step:
                self.best_arm = new_arm
        terminated = self.step_count >= 5000
        next_state = self.best_arm
        return next_state, reward, terminated, False, {}


class SimpleBandit:
    n_arms: int

    def __init__(self, n_arms=4):
        self.n_arms = n_arms
        self.best_arm = 0
        self.step_count = 0
        self.drifts = [(2000, 2), (3500, 0)]

    def reset(self):
        self.step_count = 0
        self.best_arm = 0
        return self.best_arm

    def step(self, action):
        if not isinstance(action, int):
            action = action.item() if hasattr(action, 'item') else action
        correct = 1.0 if action == self.best_arm else 0.0
        noisy = 1.0 if torch.rand(1).item() < 0.85 else 0.0
        reward = 1.0 if (correct == 1.0 and noisy == 1.0) or (correct == 0.0 and noisy == 0.0) else 0.0
        self.step_count += 1
        for switch_step, new_arm in self.drifts:
            if self.step_count == switch_step:
                self.best_arm = new_arm
        terminated = self.step_count >= 5000
        next_state = self.best_arm
        return next_state, reward, terminated, False, {}


def _reward_rate(rewards, window=100):
    if len(rewards) < window:
        return sum(rewards) / max(len(rewards), 1)
    return sum(rewards[-window:]) / window


class TestOdysseyArenaChallenge:
    STATE_DIM = 10
    HOPFIELD_DIM = 200
    HASH_DIM = 1000
    N_ARMS = 4
    TOTAL_STEPS = 5000

    def setup_method(self):
        self.vsa_state = VSA(dim=self.STATE_DIM, device="cpu")
        self.vsa_hop = VSA(dim=self.HOPFIELD_DIM, device="cpu")
        self.vsa_hash = VSA(dim=self.HASH_DIM, device="cpu")
        self._state_base = self.vsa_state.make_vector()

        self._hop_vecs = [self.vsa_hop.make_vector() for _ in range(self.N_ARMS)]
        self._hash_base = self.vsa_hash.make_vector()

    def _encode_state(self, arm_idx):
        return self.vsa_state.permute(self._state_base, shifts=arm_idx)

    def _encode_hop(self, arm_idx):
        return self._hop_vecs[arm_idx]

    def _encode_hash(self, arm_idx):
        return self.vsa_hash.permute(self._hash_base, shifts=arm_idx)

    def _run_agent(self, env, agent, optim, pool, hop, monitor, hash_store):
        rewards = []
        state = env.reset()
        state_vec = self._encode_state(state)
        pool_sizes = []

        for step in range(self.TOTAL_STEPS):
            action, _ = agent.act(state_vec)
            next_state, reward, terminated, _, _ = env.step(action)
            rewards.append(reward)
            next_vec = self._encode_state(next_state)
            hop_vec = self._encode_hop(next_state)
            hash_key = self._encode_hash(next_state)

            loss = agent.compute_loss(
                state_vec, action, reward,
                next_vec if not terminated else None,
            )
            optim.zero_grad()
            loss.backward()
            optim.step()

            pool.process(next_vec)
            if monitor.calibrated:
                monitor.score(hop_vec)
            hop.store(hop_vec)
            hash_store.insert(hash_key, hash_key)

            state_vec = next_vec
            pool_sizes.append(len(pool))
            if terminated:
                break

        return rewards, pool_sizes

    def test_reward_recovers_after_each_drift(self):
        env = SimpleBandit(n_arms=self.N_ARMS)
        agent = GoNoGoActorCritic(input_dim=self.STATE_DIM, n_actions=self.N_ARMS)
        optim = torch.optim.AdamW(agent.parameters(), lr=1e-3)
        pool = ClonalPool(input_dim=self.STATE_DIM, affinity_threshold=0.3, max_modules=12)
        hop = HopfieldNet(dim=self.HOPFIELD_DIM)
        monitor = SelfMonitor(hop, energy_threshold=3.0)
        hash_store = VSAHashStore(self.vsa_hash, num_buckets=1000)

        rewards, _ = self._run_agent(env, agent, optim, pool, hop, monitor, hash_store)

        rate_after_first = _reward_rate(rewards[2300:2500], 200)
        rate_after_second = _reward_rate(rewards[4000:], 200)
        print(f"\n  Reward rate after first drift:  {rate_after_first:.3f}")
        print(f"  Reward rate after second drift: {rate_after_second:.3f}")
        assert rate_after_first > 0.4, f"Failed after first drift: {rate_after_first:.3f}"
        assert rate_after_second > 0.4, f"Failed after second drift: {rate_after_second:.3f}"

    def test_clonal_recognizes_returning_arm(self):
        env = DriftingBanditChallenge(n_arms=self.N_ARMS)
        agent = GoNoGoActorCritic(input_dim=self.STATE_DIM, n_actions=self.N_ARMS)
        optim = torch.optim.AdamW(agent.parameters(), lr=1e-3)
        pool = ClonalPool(input_dim=self.STATE_DIM, affinity_threshold=0.3, max_modules=12)
        hop = HopfieldNet(dim=self.HOPFIELD_DIM)
        monitor = SelfMonitor(hop, energy_threshold=3.0)
        hash_store = VSAHashStore(self.vsa_hash, num_buckets=1000)

        self._run_agent(env, agent, optim, pool, hop, monitor, hash_store)

        arm0_vec = self._encode_state(0)
        max_aff = max(m.affinity(arm0_vec).item() for m in pool.modules)
        print(f"\n  Max affinity to arm 0 after full run: {max_aff:.4f}")
        assert max_aff > 0.5, f"Arm 0 forgotten: max affinity = {max_aff}"

    def test_immune_flags_drift_anomaly(self):
        hop = HopfieldNet(dim=self.HOPFIELD_DIM)
        monitor = SelfMonitor(hop, energy_threshold=3.0)

        arm0_vec = self._encode_hop(0)
        calibrate = []
        for _ in range(100):
            noisy = arm0_vec + 0.15 * torch.randn(self.HOPFIELD_DIM)
            hop.store(noisy)
            calibrate.append(noisy)
        monitor.calibrate(calibrate)

        baseline_z = []
        for _ in range(50):
            v = arm0_vec + 0.15 * torch.randn(self.HOPFIELD_DIM)
            score = monitor.score(v)
            baseline_z.append(abs(score["energy_z"]))
        mean_base = sum(baseline_z) / len(baseline_z)

        drift_arms = [2, 3, 1]
        for drift_arm in drift_arms:
            dv = self._encode_hop(drift_arm)
            drift_z = []
            for _ in range(20):
                v = dv + 0.15 * torch.randn(self.HOPFIELD_DIM)
                score = monitor.score(v)
                drift_z.append(abs(score["energy_z"]))
            mean_drift = sum(drift_z) / len(drift_z)
            print(f"\n  Arm {drift_arm}: mean |energy_z| = {mean_drift:.2f} (baseline: {mean_base:.2f})")
            assert mean_drift > mean_base * 1.5 or mean_drift > 1.0, (
                f"Arm {drift_arm} |energy_z| ({mean_drift:.2f}) too close to baseline ({mean_base:.2f})"
            )

    def test_constant_time_selection(self):
        times = {}
        for n in [500, 2500, 4500]:
            hs = VSAHashStore(self.vsa_hash, num_buckets=1000)
            for i in range(n):
                k = self.vsa_hash.permute(self._hash_base, shifts=i % self.N_ARMS)
                hs.insert(k, k)
            q = self.vsa_hash.permute(self._hash_base, shifts=0)
            for _ in range(10):
                hs.lookup(q)
            start = time.perf_counter()
            for _ in range(1000):
                hs.lookup(q)
            times[n] = (time.perf_counter() - start) / 1000

        ratio = times[4500] / times[500]
        print(f"\n  Time @ 500 items:   {times[500]*1e6:.3f} µs")
        print(f"  Time @ 2500 items:  {times[2500]*1e6:.3f} µs")
        print(f"  Time @ 4500 items:  {times[4500]*1e6:.3f} µs")
        print(f"  Ratio (4500/500):   {ratio:.2f}x")
        assert ratio < 2.0

    def test_pool_stays_bounded(self):
        env = DriftingBanditChallenge(n_arms=self.N_ARMS)
        agent = GoNoGoActorCritic(input_dim=self.STATE_DIM, n_actions=self.N_ARMS)
        optim = torch.optim.AdamW(agent.parameters(), lr=1e-3)
        pool = ClonalPool(input_dim=self.STATE_DIM, affinity_threshold=0.3, max_modules=12)
        hop = HopfieldNet(dim=self.HOPFIELD_DIM)
        monitor = SelfMonitor(hop, energy_threshold=3.0)
        hash_store = VSAHashStore(self.vsa_hash, num_buckets=1000)

        _, pool_sizes = self._run_agent(env, agent, optim, pool, hop, monitor, hash_store)
        max_size = max(pool_sizes) if pool_sizes else 0
        print(f"\n  Max pool size: {max_size}, limit: {pool.max_modules}")
        assert max_size <= pool.max_modules

    def test_trained_beats_random(self):
        env = SimpleBandit(n_arms=self.N_ARMS)
        agent = GoNoGoActorCritic(input_dim=self.STATE_DIM, n_actions=self.N_ARMS)
        optim = torch.optim.AdamW(agent.parameters(), lr=1e-3)
        pool = ClonalPool(input_dim=self.STATE_DIM, affinity_threshold=0.3, max_modules=12)
        hop = HopfieldNet(dim=self.HOPFIELD_DIM)
        monitor = SelfMonitor(hop, energy_threshold=3.0)
        hash_store = VSAHashStore(self.vsa_hash, num_buckets=1000)

        rewards, _ = self._run_agent(env, agent, optim, pool, hop, monitor, hash_store)

        trained_rate = _reward_rate(rewards, 1500)

        env2 = SimpleBandit(n_arms=self.N_ARMS)
        env2.reset()
        random_rewards = []
        for _ in range(self.TOTAL_STEPS):
            action = torch.randint(0, self.N_ARMS, ()).item()
            _, reward, term, _, _ = env2.step(action)
            random_rewards.append(reward)
            if term:
                break
        random_rate = _reward_rate(random_rewards, 1500)

        print(f"\n  Trained reward rate (last 1500 steps): {trained_rate:.3f}")
        print(f"  Random reward rate (last 1500 steps):  {random_rate:.3f}")
        assert trained_rate > random_rate * 1.5, (
            f"Trained ({trained_rate:.3f}) not 1.5x random ({random_rate:.3f})"
        )
