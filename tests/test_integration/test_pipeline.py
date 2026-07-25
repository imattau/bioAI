"""End-to-end integration test: wire all five subsystems together."""

import torch
from src.vsa import VSA, AssociativeStore, HopfieldNet
from src.clonal import ClonalPool
from src.immune import SelfMonitor
from src.basal import GoNoGoActorCritic
from src.nca import NCACell, NCA, CoarseConditioner


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
        monitor = SelfMonitor(activation_dim=32, n_detectors=2)
        normal = [vsa.make_vector() for _ in range(20)]
        monitor.calibrate(normal)
        normal_result = monitor.score(vsa.make_vector())
        assert "anomaly_score" in normal_result
        assert "drift" in normal_result

    def test_actor_critic_selects_memory(self):
        vsa = VSA(dim=16, device="cpu")
        n_actions = 4
        agent = GoNoGoActorCritic(input_dim=16, n_actions=n_actions)
        state = vsa.make_vector()
        action, value = agent.act(state)
        assert 0 <= action < n_actions
        assert value.shape == (1,)

    def test_nca_conditioned_on_vsa(self):
        vsa = VSA(dim=100, device="cpu")
        cond = CoarseConditioner(vsa_dim=100, grid_h=28, grid_w=28)
        nca = NCA(NCACell(hidden_dim=8), grid_size=(28, 28), channels=8)
        v = vsa.make_vector()
        seed = cond(v)
        assert seed.shape == (1, 1, 28, 28)
        result = nca.generate(seed, steps=10)
        assert result.shape == (1, 8, 28, 28)

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
        pool = ClonalPool(input_dim=32)
        monitor = SelfMonitor(activation_dim=32, n_detectors=2)
        activations = []
        for _ in range(15):
            v = vsa.make_vector()
            out, _ = pool.process(v)
            activations.append(out.detach())
        monitor.calibrate(activations)
        novel = torch.randn(32)
        result = monitor.score(novel)
        assert "is_anomaly" in result
