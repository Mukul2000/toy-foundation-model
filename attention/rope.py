import torch
import torch.nn as nn

from config import ModelArgs


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Precomputes frequencies, handles half-rotation, and applies the rotation to tensors.
    """

    def __init__(self, args: ModelArgs, theta: float = 10000.0):
        super().__init__()
        assert args.dim % args.n_heads == 0, "dim must be divisible by n_heads"
        self.dim = args.dim // args.n_heads
        max_seq_len = args.max_seq_len

        # Compute frequency bands: theta_i = 10000^(-2 * (i - 1) / dim)
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute positions t from 0 to max_seq_len - 1
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # shape: (max_seq_len, dim // 2)

        # Duplicate columns to match full head_dim [f1, f2, ..., f1, f2, ...]
        emb = torch.cat((freqs, freqs), dim=-1)  # shape: (max_seq_len, dim)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (B, H, T, head_dim)
        seq_len = x.shape[2]

        # Slices cos and sin caches to the current sequence length
        cos = self.cos_cached[:seq_len].view(1, 1, seq_len, self.dim)
        sin = self.sin_cached[:seq_len].view(1, 1, seq_len, self.dim)

        # Apply rotation
        return self._apply_rope(x, cos, sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Private helper: Splits the final dimension of x in half, negates the second half, and swaps them.
        """
        half_size = x.shape[-1] // 2
        x1 = x[..., :half_size]
        x2 = x[..., half_size:]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Private helper: Applies rotation using the precomputed sine and cosine.
        """
        return (x * cos) + (RotaryEmbedding._rotate_half(x) * sin)
