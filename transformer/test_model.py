import pytest
import torch

from config import ModelArgs
from transformer.model import Transformer


def test_transformer_logits_and_loss_shape():
    """
    Verifies that the entire Transformer network outputs logits of correct shape
    and returns a scalar loss when targets are provided.
    """
    args = ModelArgs(vocab_size=100, dim=64, n_layers=2, n_heads=2, max_seq_len=32)
    model = Transformer(args)

    batch_size = 2
    seq_len = 16

    # Random integer inputs (token IDs between 0 and 99)
    idx = torch.randint(0, args.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    logits, loss = model(idx, targets)

    # Logits shape must be (Batch, Seq_Len, Vocab_Size)
    assert logits.shape == (batch_size, seq_len, args.vocab_size)

    # Loss must be a scalar (0-dimensional tensor)
    assert loss is not None
    assert loss.dim() == 0


def test_transformer_autoregressive_generation():
    """
    Verifies that the generate method generates exactly the requested number of tokens.
    """
    args = ModelArgs(vocab_size=100, dim=64, n_layers=2, n_heads=2, max_seq_len=32)
    model = Transformer(args)

    batch_size = 1
    seq_len = 8
    idx = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    # Generate 10 new tokens
    generated = model.generate(idx, max_new_tokens=10, temperature=0.7, top_k=5)

    # Total sequence length should be 8 + 10 = 18
    assert generated.shape == (batch_size, seq_len + 10)


def test_transformer_generation_deterministic():
    """
    Verifies that low temperature/top-k generations can execute successfully
    without numerical or runtime exceptions.
    """
    args = ModelArgs(vocab_size=100, dim=64, n_layers=2, n_heads=2, max_seq_len=32)
    model = Transformer(args)

    batch_size = 1
    seq_len = 4
    idx = torch.randint(0, args.vocab_size, (batch_size, seq_len))

    # Run deterministic (temperature = 0.0) generation
    generated = model.generate(idx, max_new_tokens=5, temperature=0.0)
    assert generated.shape == (batch_size, seq_len + 5)


def test_transformer_loss_magnitude():
    """
    Asserts that the initial shifted cross-entropy loss is close to the expected
    uniform random distribution loss: ln(vocab_size).
    """
    args = ModelArgs(vocab_size=100, dim=64, n_layers=1, n_heads=2, max_seq_len=32)
    model = Transformer(args)

    idx = torch.randint(0, args.vocab_size, (2, 8))
    targets = torch.randint(0, args.vocab_size, (2, 8))

    _, loss = model(idx, targets)

    # ln(100) is ~4.6. Since weights are randomly initialized, loss should be close to 4.6.
    expected_loss = torch.log(torch.tensor(args.vocab_size, dtype=torch.float32))
    assert torch.allclose(loss, expected_loss, rtol=0.2, atol=0.2)
