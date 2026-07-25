import torch
from src.clonal import ClonalModule, ClonalPool


class TestClonalModule:
    def test_forward_and_affinity(self):
        x = torch.randn(50)
        module = ClonalModule(x, input_dim=50)
        out = module(x)
        assert out.shape == (50,)
        aff = module.affinity(x)
        assert aff.item() > 0.99

    def test_clone_differs(self):
        x = torch.randn(50)
        parent = ClonalModule(x, input_dim=50)
        child = parent.clone(x + 0.1 * torch.randn(50), mutation_strength=0.1)
        p = list(parent.net.parameters())[0]
        c = list(child.net.parameters())[0]
        assert not torch.allclose(p, c)


class TestClonalPool:
    def setup_method(self):
        self.pool = ClonalPool(input_dim=50, affinity_threshold=0.5, max_modules=5)

    def test_first_input_creates_module(self):
        out, created = self.pool.process(torch.randn(50))
        assert len(self.pool) == 1
        assert created

    def test_similar_input_matches(self):
        x = torch.randn(50)
        self.pool.process(x)
        out, created = self.pool.process(x)
        assert created

    def test_pruning_limits_size(self):
        for _ in range(10):
            self.pool.process(torch.randn(50))
        assert len(self.pool) <= 5
