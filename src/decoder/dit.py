import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Rotary(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached: int | None = None
        self.cos_cached: torch.Tensor | None = None
        self.sin_cached: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached, self.sin_cached


@torch.no_grad()
def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    cos = cos[:x.shape[1], :half].unsqueeze(1)
    sin = sin[:x.shape[1], :half].unsqueeze(1)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False) if False else nn.LayerNorm(hidden_size)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.attn_out = nn.Linear(hidden_size, hidden_size, bias=False)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(mlp_ratio * hidden_size)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(mlp_ratio * hidden_size), hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        x_skip = x
        x = self.norm1(x)
        x = modulate(x, shift_msa, scale_msa)

        qkv = self.qkv(x)
        b, s, _ = qkv.shape
        qkv = qkv.reshape(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=None, dropout_p=0.0, is_causal=False,
        )
        x = attn.transpose(1, 2).reshape(b, s, -1)
        x = self.attn_out(x)
        x = x_skip + gate_msa.unsqueeze(1) * x

        x_skip = x
        x = self.norm2(x)
        x = modulate(x, shift_mlp, scale_mlp)
        x = self.mlp(x)
        x = x_skip + gate_mlp.unsqueeze(1) * x
        return x


class VSAConditionedDiT(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 256, num_heads: int = 4,
                 num_layers: int = 6, cond_dim: int = 256, vsa_dim: int = 10000,
                 max_seq_len: int = 64, mask_token_id: int | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.mask_token_id = mask_token_id if mask_token_id is not None else vocab_size

        self.vocab_embed = nn.Embedding(vocab_size + 2, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_size) * 0.02)
        self.vsa_proj = nn.Sequential(
            nn.Linear(vsa_dim, cond_dim, bias=True),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim, bias=True),
        )
        self.rotary = Rotary(hidden_size // num_heads)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, cond_dim) for _ in range(num_layers)
        ])

        self.norm_final = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size + 2)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_final[1].weight)
        nn.init.zeros_(self.adaLN_final[1].bias)

    def forward(self, token_ids: torch.Tensor, vsa_vector: torch.Tensor) -> torch.Tensor:
        x = self.vocab_embed(token_ids)
        x = x + self.pos_embed[:, :x.shape[1], :]

        c = self.vsa_proj(vsa_vector)

        cos, sin = self.rotary(x)
        for block in self.blocks:
            x = block(x, cos, sin, c)

        shift, scale = self.adaLN_final(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = modulate(x, shift, scale)
        logits = self.lm_head(x)
        return logits
