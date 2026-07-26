import torch
from torch import nn

from config import ModelArgs
from .attention import MultiHeadSelfAttention
from .feedforward import FeedForward
from .rmsnorm import RMSNorm


class TransformerBlock(nn.Module):
    """
    A single Transformer Block (Decoder Layer) combining Pre-RMSNorm,
    Multi-Head Self-Attention, SwiGLU Feed-Forward, and Residual Connections.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim

        # Attention sub-block components
        self.attention_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.attention = MultiHeadSelfAttention(args)

        # Feed-Forward sub-block components
        self.ffn_norm = RMSNorm(dim=args.dim, eps=args.norm_eps)
        self.feed_forward = FeedForward(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Self-Attention with Pre-RMSNorm and Residual Skip Connection
        x = x + self.attention(self.attention_norm(x))

        # 2. SwiGLU Feed-Forward with Pre-RMSNorm and Residual Skip Connection
        x = x + self.feed_forward(self.ffn_norm(x))

        return x
