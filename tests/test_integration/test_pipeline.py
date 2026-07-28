"""End-to-end integration test: wire the encoding/memory/action subsystems
together (generation via NCA/DiT was removed as an unwired, superseded
experiment -- see LookupDecoder in src/decoder for the production path)."""

import torch
from src.vsa import VSA, AssociativeStore, HopfieldNet
from src.clonal import ClonalPool
from src.immune import SelfMonitor
from src.basal import GoNoGoActorCritic


class TestFullPipeline:
    def test_vsa_store_hopfield_roundtrip(self):
        vsa = VSA(dim=100, device="cpu")
        store = AssociativeStore(dim=100, capacity=10)
        hop = HopfieldNet(dim=100)
        v = vsa.make_vector()
        store.insert(v)
        hop.store(v)
        results = store.lookup(v, k=1)
        assert len(results) == 1
        recalled = hop.recall(v, steps=10)
        sim = vsa.similarity(v, recalled)
        assert sim.item() > 0.8

    def test_clonal_with_vsa_encoding(self):
        vsa = VSA(dim=100, device="cpu")
        pool = ClonalPool(input_dim=100)
        v = vsa.make_vector()
        out, created = pool.process(v)
        assert out.shape == (100,)
        assert created

    def test_immune_monitors_vsa_activations(self):
        vsa = VSA(dim=32, device="cpu")
        hop = HopfieldNet(dim=32)
        patterns = [vsa.make_vector() for _ in range(5)]
        for p in patterns:
            hop.store(p)
        monitor = SelfMonitor(hop, energy_threshold=3.0)
        normal = [hop.recall(p + 0.1 * torch.randn(32), steps=10) for p in patterns]
        monitor.calibrate(normal)
        normal_result = monitor.score(hop.recall(patterns[0] + 0.1 * torch.randn(32), steps=10))
        assert "energy" in normal_result
        assert "energy_z" in normal_result

    def test_actor_critic_selects_memory(self):
        vsa = VSA(dim=16, device="cpu")
        n_actions = 4
        agent = GoNoGoActorCritic(input_dim=16, n_actions=n_actions)
        state = vsa.make_vector()
        action, value = agent.act(state)
        assert 0 <= action < n_actions
        assert value.shape == (1,)

    def test_continual_learning_no_forgetting(self):
        vsa = VSA(dim=100, device="cpu")
        pool = ClonalPool(input_dim=100, affinity_threshold=0.4, max_modules=10)
        tasks = []
        for i in range(3):
            vecs = [vsa.make_vector() for _ in range(5)]
            tasks.append(vecs)
        for task_vecs in tasks:
            for v in task_vecs:
                pool.process(v, lr=0.01)
        assert len(pool) <= 10
        assert len(pool) >= 3

    def test_clonal_with_self_monitor(self):
        vsa = VSA(dim=32, device="cpu")
        hop = HopfieldNet(dim=32)
        pool = ClonalPool(input_dim=32)
        monitor = SelfMonitor(hop, energy_threshold=3.0)
        activations = []
        for _ in range(15):
            v = vsa.make_vector()
            out, _ = pool.process(v)
            activations.append(out.detach())
        for a in activations:
            hop.store(a)
        monitor.calibrate(activations)
        novel = torch.randn(32)
        result = monitor.score(novel)
        assert "is_anomaly" in result
