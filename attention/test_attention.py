import pytest
import torch

from config import ModelArgs
from attention import MultiHeadSelfAttention


def test_attention_shape():
    """
    Verifies that the attention layer maintains the exact tensor shape (B, T, dim).
    """
    args = ModelArgs(dim=512, n_heads=8, max_seq_len=512)
    attention = MultiHeadSelfAttention(args)

    batch_size = 4
    seq_len = 32
    x = torch.randn(batch_size, seq_len, args.dim)

    out = attention(x)
    assert out.shape == (batch_size, seq_len, args.dim)


def test_attention_causality():
    """
    Verifies the causal masking property of standard self-attention:
    The output at position t must ONLY depend on input states at positions <= t.
    We verify this by taking gradients of the output at position t with respect to
    the input sequence and ensuring they are strictly zero for all positions > t.
    """
    args = ModelArgs(dim=512, n_heads=8, max_seq_len=512)
    attention = MultiHeadSelfAttention(args)

    batch_size = 2
    seq_len = 16
    x = torch.randn(batch_size, seq_len, args.dim, requires_grad=True)

    out = attention(x)

    # We evaluate gradients for output at position t = 5
    t = 5
    loss = out[:, t, :].sum()
    loss.backward()

    # The gradient of the loss with respect to inputs at positions > t must be zero
    assert x.grad is not None
    for i in range(seq_len):
        if i > t:
            assert torch.all(x.grad[:, i, :] == 0.0), f"Leakage detected! Input at pos {i} affected output at pos {t}"
        else:
            # Positions <= t should have non-zero gradients/contributions
            assert torch.any(x.grad[:, i, :] != 0.0), f"No attention contribution from pos {i} to output at pos {t}"


def test_dim_divisibility():
    """
    Asserts that the module raises an error if dim is not divisible by n_heads.
    """
    invalid_args = ModelArgs(dim=512, n_heads=7, max_seq_len=512)
    with pytest.raises(AssertionError, match="dim must be divisible by n_heads"):
        MultiHeadSelfAttention(invalid_args)
