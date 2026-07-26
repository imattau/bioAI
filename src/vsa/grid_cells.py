import torch

import torchhd


class GridCellPositionalEncoder:
    def __init__(self, vsa, dim: int, num_scales: int = 2, chunk_size: int = 100):
        self.vsa = vsa
        self.dim = dim
        self.chunk_size = chunk_size
        self.num_scales = num_scales

        self.scale_bases = [vsa.make_vector() for _ in range(num_scales)]

    def encode(self, position: int, scale: int = 0) -> torch.Tensor:
        return torchhd.permute(self.scale_bases[scale], shifts=position)

    def encode_coarse(self, chunk_id: int) -> torch.Tensor:
        return torchhd.permute(self.scale_bases[0], shifts=chunk_id)

    def encode_fine(self, pos_in_chunk: int) -> torch.Tensor:
        return torchhd.permute(self.scale_bases[1], shifts=pos_in_chunk)

    def position_to_chunk(self, position: int) -> tuple[int, int]:
        chunk_id = position // self.chunk_size
        pos_in_chunk = position % self.chunk_size
        return chunk_id, pos_in_chunk
