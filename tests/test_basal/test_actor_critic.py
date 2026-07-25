import torch
from src.basal import GoNoGoActorCritic, MemoryRetrievalEnv
from src.vsa import VSA, AssociativeStore


class TestGoNoGo:
    def setup_method(self):
        self.agent = GoNoGoActorCritic(input_dim=16, n_actions=4)

    def test_forward_shapes(self):
        logits, value, go, nogo = self.agent.forward(torch.randn(16))
        assert logits.shape == (4,)
        assert value.shape == (1,)
        assert go.shape == (4,)
        assert nogo.shape == (4,)

    def test_act_returns_valid_action(self):
        action, value = self.agent.act(torch.randn(16))
        assert 0 <= action < 4


class TestMemoryRetrievalEnv:
    def setup_method(self):
        self.vsa = VSA(dim=100)
        self.store = AssociativeStore(dim=100)
        self.queries = [self.vsa.make_vector() for _ in range(3)]
        for q in self.queries:
            self.store.insert(q)
        self.env = MemoryRetrievalEnv(self.store, self.queries,
                                      correct_indices=[0, 1, 2], dim=100)

    def test_reset(self):
        obs, _ = self.env.reset()
        assert obs.shape == (100,)

    def test_step(self):
        self.env.reset()
        obs, reward, term, trunc, _ = self.env.step(0)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
