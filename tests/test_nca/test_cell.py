import torch
from src.nca import NCACell, NCA


class TestNCACell:
    def setup_method(self):
        self.cell = NCACell(hidden_dim=16)

    def test_cell_shape_preservation(self):
        state = torch.randn(1, 16, 16, 16)
        next_state = self.cell(state)
        assert next_state.shape == state.shape

    def test_cell_updates_state(self):
        state = torch.zeros(1, 16, 16, 16)
        for _ in range(10):
            state = self.cell(state)
        assert state.abs().sum() > 0


class TestNCA:
    def setup_method(self):
        self.nca = NCA(NCACell(hidden_dim=16), grid_size=(16, 16), channels=16)

    def test_generate_produces_output(self):
        seed = torch.randn(1, 1, 16, 16)
        result = self.nca.generate(seed, steps=10)
        assert result.shape == (1, 16, 16, 16)

    def test_generate_changes_over_time(self):
        seed = torch.randn(1, 1, 16, 16)
        out1 = self.nca.generate(seed, steps=5)
        out2 = self.nca.generate(seed, steps=50)
        assert not torch.allclose(out1, out2)
