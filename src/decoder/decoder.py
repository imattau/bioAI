import torch
from .dit import VSAConditionedDiT
from .diffusion import DiscreteDiffusion
from .sampler import iterative_unmasking_sample


class TextDecoder:
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_heads: int = 4,
                 num_layers: int = 6, vsa_dim: int = 10000, max_seq_len: int = 64,
                 cond_dim: int = 256, diffusion_steps: int = 1000,
                 sample_steps: int = 100, sample_temp: float = 1.0):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.sample_steps = sample_steps
        self.sample_temp = sample_temp

        self.diffusion = DiscreteDiffusion(
            mask_token_id=vocab_size,
            num_steps=diffusion_steps,
        )
        self.model = VSAConditionedDiT(
            vocab_size=vocab_size, hidden_size=hidden_size,
            num_heads=num_heads, num_layers=num_layers,
            cond_dim=cond_dim, vsa_dim=vsa_dim,
            max_seq_len=max_seq_len, mask_token_id=vocab_size,
        )

    @torch.no_grad()
    def encode_corrupt(self, token_ids: torch.Tensor,
                        t: torch.Tensor) -> torch.Tensor:
        return self.diffusion.corrupt(token_ids, t)

    def compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor,
                     corrupted_ids: torch.Tensor) -> torch.Tensor:
        return self.diffusion.compute_loss(logits, target_ids, corrupted_ids)

    def forward(self, token_ids: torch.Tensor,
                vsa_vector: torch.Tensor) -> torch.Tensor:
        return self.model(token_ids, vsa_vector)

    @torch.no_grad()
    def sample(self, vsa_vector: torch.Tensor) -> torch.Tensor:
        return iterative_unmasking_sample(
            self.model, vsa_vector,
            mask_token_id=self.vocab_size,
            max_len=self.max_seq_len,
            num_steps=self.sample_steps,
            temp=self.sample_temp,
        )
