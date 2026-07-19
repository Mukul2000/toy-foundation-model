import pytest
import torch

from config import ModelArgs
from attention import RotaryEmbedding


def test_rotate_half_correctness():
    """
    Verifies that _rotate_half performs the half-dimension swap and negation correctly.
    """
    # Create a vector [1, 2, 3, 4] -> _rotate_half should swap [1, 2] and [3, 4] and negate the new first half: [-3, -4, 1, 2]
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
    res = RotaryEmbedding._rotate_half(x)
    assert torch.allclose(res, expected)


def test_rope_rotation_preserves_norm():
    """
    Verifies that applying Rotary Position Embeddings (RoPE) preserves the L2 norm
    of vectors, since a pure rotation in the complex plane should be norm-preserving.
    """
    head_dim = 64
    seq_len = 16
    args = ModelArgs(dim=head_dim * 8, n_heads=8, max_seq_len=128)
    rope = RotaryEmbedding(args)

    # Mocking x with shape (B, H, T, head_dim)
    x = torch.randn(2, 4, seq_len, head_dim)
    rotated_x = rope(x)

    # Check that norm is preserved for each individual token vector
    orig_norm = torch.norm(x, dim=-1)
    rot_norm = torch.norm(rotated_x, dim=-1)

    assert torch.allclose(orig_norm, rot_norm, rtol=1e-5, atol=1e-5)


def test_rope_dim_divisibility():
    """
    Asserts that the RotaryEmbedding class raises an error if dim is not divisible by n_heads.
    """
    invalid_args = ModelArgs(dim=512, n_heads=7, max_seq_len=512)
    with pytest.raises(AssertionError, match="dim must be divisible by n_heads"):
        RotaryEmbedding(invalid_args)
