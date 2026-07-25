import torch
from src.decoder import VSAConditionedDiT, TextDecoder


class TestVSAConditionedDiT:
    def setup_method(self):
        self.model = VSAConditionedDiT(
            vocab_size=20, hidden_size=64, num_heads=2,
            num_layers=3, vsa_dim=100, max_seq_len=12,
        )

    def test_forward_shape(self):
        tokens = torch.randint(0, 20, (2, 8))
        vsa = torch.randn(2, 100)
        logits = self.model(tokens, vsa)
        assert logits.shape == (2, 8, 22)

    def test_forward_different_vsa_after_step(self):
        tokens = torch.randint(0, 20, (1, 8))
        v1 = torch.randn(1, 100)
        v2 = torch.randn(1, 100)
        opt = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        logits = self.model(tokens, v1)
        logits.sum().backward()
        opt.step()
        logits_1 = self.model(tokens, v1)
        logits_2 = self.model(tokens, v2)
        diff = (logits_1 - logits_2).abs().mean().item()
        assert diff > 0.001


class TestTextDecoder:
    def setup_method(self):
        self.decoder = TextDecoder(
            vocab_size=20, hidden_size=64, num_heads=2,
            num_layers=3, vsa_dim=100, max_seq_len=12,
            diffusion_steps=100, sample_steps=10,
        )

    def test_forward_train(self):
        tokens = torch.randint(0, 20, (2, 8))
        vsa = torch.randn(2, 100)
        logits = self.decoder.forward(tokens, vsa)
        assert logits.shape == (2, 8, 22)

    def test_sample(self):
        vsa = torch.randn(1, 100)
        tokens = self.decoder.sample(vsa)
        assert tokens.shape == (1, 12)

    def test_loss_computation(self):
        tokens = torch.randint(0, 20, (2, 8))
        vsa = torch.randn(2, 100)
        t = torch.randint(0, 100, (2,))
        corrupted = self.decoder.encode_corrupt(tokens, t)
        logits = self.decoder.forward(corrupted, vsa)
        loss = self.decoder.compute_loss(logits, tokens, corrupted)
        assert loss.item() > 0
