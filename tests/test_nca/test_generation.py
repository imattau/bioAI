import torch
import torch.nn.functional as F

from src.nca import NCACell, NCA, CoarseConditioner
from src.nca.maze_data import build_dataset
from src.vsa import VSA

GRID_SIZE = 28
HIDDEN_DIM = 16


def _make_nca():
    cell = NCACell(hidden_dim=HIDDEN_DIM, fire_rate=0.5)
    return NCA(cell, grid_size=(GRID_SIZE, GRID_SIZE), channels=HIDDEN_DIM)


def _make_conditioner():
    return CoarseConditioner(vsa_dim=1000, grid_h=GRID_SIZE, grid_w=GRID_SIZE)


def _train_nca(nca, conditioner, dataset, steps=100, lr=1e-3):
    opt = torch.optim.Adam(
        list(nca.parameters()) + list(conditioner.parameters()), lr=lr
    )
    for epoch in range(steps):
        total_loss = 0.0
        for vsa_vec, target in dataset:
            seed = conditioner(vsa_vec)
            output = nca.generate(seed, steps=32, target=seed)
            target_bc = target.expand_as(output[:, :1, :, :])
            loss = F.mse_loss(output[:, :1, :, :], target_bc)
            total_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch == steps - 1:
            return total_loss / len(dataset)
    return 0.0


class TestGeneration:
    DIM = 1000

    def test_no_collapse(self):
        nca = _make_nca()
        seed = torch.randn(1, 1, GRID_SIZE, GRID_SIZE)
        output = nca.generate(seed, steps=200)
        channel = output[:, :1, :, :]
        assert channel.abs().mean().item() > 0.01, "NCA collapsed to zeros"
        uniq = channel.unique().numel()
        assert uniq > 1, "NCA collapsed to constant value"

    def test_output_no_nan(self):
        nca = _make_nca()
        seed = torch.randn(1, 1, GRID_SIZE, GRID_SIZE)
        output = nca.generate(seed, steps=100)
        assert not torch.isnan(output).any(), "NCA output contains NaN"
        assert not torch.isinf(output).any(), "NCA output contains Inf"

    def test_coarse_conditioner_produces_valid_seed(self):
        vsa = VSA(dim=self.DIM, device="cpu")
        cond = _make_conditioner()
        v = vsa.make_vector()
        seed = cond(v)
        assert seed.shape == (1, 1, GRID_SIZE, GRID_SIZE)
        assert (seed >= 0).all() and (seed <= 1).all()

    def test_nca_learns_patterns(self):
        vsa = VSA(dim=self.DIM, device="cpu")
        nca = _make_nca()
        cond = _make_conditioner()
        dataset = build_dataset(vsa, grid_size=GRID_SIZE)[:3]

        loss_before = _train_nca(nca, cond, dataset, steps=50)
        print(f"\n  Final training loss: {loss_before:.6f}")
        assert loss_before < 0.1, (
            f"NCA training failed to converge: loss={loss_before:.4f}"
        )

    def test_different_vsa_different_output(self):
        vsa = VSA(dim=self.DIM, device="cpu")
        nca = _make_nca()
        cond = _make_conditioner()
        dataset = build_dataset(vsa, grid_size=GRID_SIZE)[:3]
        _train_nca(nca, cond, dataset, steps=50)

        v1 = vsa.make_vector()
        v2 = vsa.make_vector()
        s1 = cond(v1)
        s2 = cond(v2)
        o1 = nca.generate(s1, steps=32, target=s1)
        o2 = nca.generate(s2, steps=32, target=s2)
        diff = (o1 - o2).abs().mean().item()
        print(f"\n  Mean abs diff between outputs: {diff:.4f}")
        assert diff > 0.001, (
            f"Different VSA vectors produced nearly identical outputs: "
            f"diff={diff}"
        )

    def test_deterministic_mode_produces_same_output(self):
        vsa = VSA(dim=self.DIM, device="cpu")
        cell = NCACell(hidden_dim=HIDDEN_DIM, fire_rate=1.0)
        nca = NCA(cell, grid_size=(GRID_SIZE, GRID_SIZE), channels=HIDDEN_DIM)
        cond = _make_conditioner()
        dataset = build_dataset(vsa, grid_size=GRID_SIZE)[:3]
        _train_nca(nca, cond, dataset, steps=50)

        v = vsa.make_vector()
        s = cond(v)
        o1 = nca.generate(s, steps=32, target=s)
        o2 = nca.generate(s, steps=32, target=s)
        diff = (o1 - o2).abs().mean().item()
        assert diff < 0.001, (
            f"Deterministic NCA produced different outputs: diff={diff}"
        )
