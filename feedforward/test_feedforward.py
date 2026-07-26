import pytest
import torch

from config import ModelArgs
from feedforward import FeedForward


def test_feedforward_shape():
    """
    Verifies that the FeedForward block preserves the tensor shape (B, T, dim).
    """
    args = ModelArgs(dim=512)
    ffn = FeedForward(args)

    batch_size = 4
    seq_len = 32
    x = torch.randn(batch_size, seq_len, args.dim)

    out = ffn(x)
    assert out.shape == (batch_size, seq_len, args.dim)


def test_hidden_dim_calculation():
    """
    Verifies that the hidden dimension calculation works correctly and rounds up
    to the nearest multiple of 256.
    For dim=512, 2 * (4 * 512) / 3 = 1365.33, which rounds up to 1536.
    """
    # 512 base hidden dimension
    assert FeedForward._calculate_hidden_dim(512) == 1536

    # 128 base hidden dimension: 2 * (4 * 128) / 3 = 341.33, rounds up to 512
    assert FeedForward._calculate_hidden_dim(128) == 512


def test_feedforward_gradients():
    """
    Verifies that the backward pass propagates gradients correctly to all
    three weight projection matrices (w_gate, w_down, w_up).
    """
    args = ModelArgs(dim=256)
    ffn = FeedForward(args)

    x = torch.randn(2, 8, args.dim)
    out = ffn(x)

    loss = out.sum()
    loss.backward()

    # Verify that gradients exist and are non-zero for all projections
    assert ffn.w_gate.weight.grad is not None
    assert ffn.w_down.weight.grad is not None
    assert ffn.w_up.weight.grad is not None

    assert torch.any(ffn.w_gate.weight.grad != 0.0)
    assert torch.any(ffn.w_down.weight.grad != 0.0)
    assert torch.any(ffn.w_up.weight.grad != 0.0)
