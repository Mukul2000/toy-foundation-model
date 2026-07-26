import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm).
    Simplifies LayerNorm by removing the mean-centering step, making it faster.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Gamma is the learnable scaling parameter
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate Root Mean Square: sqrt( mean(x^2) + eps )
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(variance + self.eps)
        # Normalize and scale with learnable gamma
        return x * rms * self.gamma
