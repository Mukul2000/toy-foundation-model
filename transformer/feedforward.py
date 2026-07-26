import torch
import torch.nn.functional as F
from torch import nn

from config import ModelArgs


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network (FFN) block.
    Standard in modern Transformer architectures (LLaMA, Mistral, Gemma).
    """

    def __init__(self, args: ModelArgs):
        super().__init__()

        # Calculate optimal hidden dimension
        self.hidden_dim = self._calculate_hidden_dim(args.dim)

        # Projections
        self.w_gate = nn.Linear(args.dim, self.hidden_dim, bias=False)  # Gate projection
        self.w_down = nn.Linear(self.hidden_dim, args.dim, bias=False)  # Down projection
        self.w_up = nn.Linear(args.dim, self.hidden_dim, bias=False)    # Up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    @staticmethod
    def _calculate_hidden_dim(dim: int, multiple_of: int = 256) -> int:
        """
        Private helper: Computes the SwiGLU hidden dimension scaled by 2/3
        and rounds up to the nearest multiple of 256 for hardware alignment.
        """
        hidden_dim = int(2 * (4 * dim) / 3)
        # Round up to nearest multiple of 256
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        return hidden_dim
