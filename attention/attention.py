import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelArgs
from .rope import RotaryEmbedding


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0, "dim must be divisible by n_heads"

        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads

        # Query, Key, Value projections
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)

        # Output projection
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

        # Rotary Position Embedding module
        self.rope = RotaryEmbedding(args)

        # Precomputed causal mask cache
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(args.max_seq_len, args.max_seq_len)).view(
                1, 1, args.max_seq_len, args.max_seq_len
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Stage 2: Projections
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # Stage 3: Reshape/Transpose into Heads (B, H, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Stage 4: Apply Rotary Positional Embeddings
        q = self.rope(q)
        k = self.rope(k)

        # Stage 5: Scaled dot-product attention scores calculation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Stage 6: Apply Causal Mask and Softmax
        # Slices causal mask dynamically to current sequence length T
        mask = self.causal_mask[:, :, :T, :T]
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)

        # Stage 7: Weighted sum over Values V
        attn_out = torch.matmul(weights, v)

        # Stage 8: Transpose and concatenate heads back to (B, T, dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.dim)

        # Stage 9: Output Projection
        return self.wo(attn_out)
