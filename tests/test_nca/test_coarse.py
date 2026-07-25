import torch
from src.nca import CoarseConditioner


class TestCoarseConditioner:
    def test_projection(self):
        cond = CoarseConditioner(vsa_dim=1000, grid_h=28, grid_w=28)
        v = torch.randn(1000)
        seed = cond(v)
        assert seed.shape == (1, 1, 28, 28)
        assert (seed >= 0).all() and (seed <= 1).all()
