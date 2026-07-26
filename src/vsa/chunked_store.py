import torch

import torchhd

from .grid_cells import GridCellPositionalEncoder


class PositionalVSAStore:
    def __init__(self, encoder: GridCellPositionalEncoder, dim: int):
        self.encoder = encoder
        self.dim = dim
        self.buckets: dict[int, torch.Tensor] = {}

    def insert(self, position: int, value: torch.Tensor):
        pos_in_chunk = position % self.encoder.chunk_size
        fine = self.encoder.encode_fine(pos_in_chunk)
        bound = torchhd.bind(fine, value)
        self.buckets[position] = bound

    def query(self, position: int) -> torch.Tensor | None:
        if position not in self.buckets:
            return None
        pos_in_chunk = position % self.encoder.chunk_size
        fine = self.encoder.encode_fine(pos_in_chunk)
        stored = self.buckets[position]
        return torchhd.bind(stored, fine)

    def __len__(self) -> int:
        return len(self.buckets)
