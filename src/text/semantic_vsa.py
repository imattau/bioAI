"""Projection of learned text embeddings into deterministic bipolar VSAs."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


class OllamaEmbedder:
    def __init__(self, model: str = "qwen3-embedding:0.6b"):
        self.model = model

    def __call__(self, text: str) -> torch.Tensor:
        import ollama

        response = ollama.embed(model=self.model, input=text)
        embeddings = (
            response["embeddings"]
            if isinstance(response, dict) else response.embeddings
        )
        return torch.tensor(embeddings[0], dtype=torch.float32)


class SemanticVSAEncoder:
    def __init__(
        self,
        embedder: Callable[[str], torch.Tensor],
        vsa_dim: int = 1024,
        seed: int = 42,
        projection: torch.Tensor | None = None,
    ):
        self.embedder = embedder
        self.vsa_dim = vsa_dim
        self.seed = seed
        self.projection = projection

    def _ensure_projection(self, embedding_dim: int):
        if self.projection is None:
            generator = torch.Generator().manual_seed(self.seed)
            self.projection = torch.randn(
                self.vsa_dim, embedding_dim, generator=generator
            ) / embedding_dim ** 0.5
        elif self.projection.shape != (self.vsa_dim, embedding_dim):
            raise ValueError(
                "Embedding dimension does not match persisted semantic projection"
            )

    def encode(self, text: str) -> torch.Tensor:
        embedding = self.embedder(text).detach().float().flatten().cpu()
        self._ensure_projection(len(embedding))
        embedding = F.normalize(embedding, dim=0)
        projected = self.projection @ embedding
        return torch.where(projected >= 0, 1, -1).to(torch.int8)

    def similarity(self, left: torch.Tensor, right: torch.Tensor) -> float:
        return (left.float() @ right.float() / self.vsa_dim).item()

    def get_state(self) -> dict:
        return {
            "vsa_dim": self.vsa_dim,
            "seed": self.seed,
            "projection": self.projection,
        }
