import pytest
import torch

from config import ModelArgs
from transformer import RMSNorm, TransformerBlock


def test_rmsnorm_correctness():
    """
    Verifies that RMSNorm scales inputs correctly.
    Specifically, the output vectors should have a root-mean-square value
    of exactly 1.0 (with a small margin for epsilon) when gamma is 1.0.
    """
    dim = 128
    norm = RMSNorm(dim=dim, eps=1e-5)

    # Generate a random tensor
    x = torch.randn(4, 16, dim) * 10.0  # highly scaled inputs

    out = norm(x)

    # Compute Root Mean Square (RMS) of output: sqrt( mean(out^2) )
    # This must be extremely close to 1.0
    out_rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(out_rms, torch.ones_like(out_rms), rtol=1e-3, atol=1e-3)


def test_transformer_block_shape():
    """
    Verifies that the entire TransformerBlock maintains shape (B, T, dim).
    """
    args = ModelArgs(dim=256, n_heads=4)
    block = TransformerBlock(args)

    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, args.dim)

    out = block(x)
    assert out.shape == (batch_size, seq_len, args.dim)


def test_transformer_block_gradients():
    """
    Verifies that the backward pass propagates gradients correctly through both
    the attention sub-block and feed-forward sub-block weights.
    """
    args = ModelArgs(dim=128, n_heads=2)
    block = TransformerBlock(args)

    x = torch.randn(2, 8, args.dim, requires_grad=True)
    out = block(x)

    loss = out.sum()
    loss.backward()

    # Gradients must flow back to x
    assert x.grad is not None
    assert torch.any(x.grad != 0.0)

    # Gradients must flow to norms, attention weights, and FFN weights
    assert block.attention_norm.gamma.grad is not None
    assert block.attention.wq.weight.grad is not None
    assert block.feed_forward.w_gate.weight.grad is not None
