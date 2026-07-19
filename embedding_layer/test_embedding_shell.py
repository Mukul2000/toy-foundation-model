import pytest
import torch

from config import ModelArgs
from embedding_layer.embedding_shell import TransformerEmbeddingShell


def test_embedding_shell_initialization():
    """
    Verifies that the embedding shell initializes correctly with the configuration args.
    """
    args = ModelArgs(vocab_size=1000, dim=256)
    shell = TransformerEmbeddingShell(args, pad_token_id=0)

    assert shell.vocab_size == 1000
    assert shell.hidden_dim == 256
    assert shell.token_embeddings.padding_idx == 0
    assert shell.token_embeddings.weight.shape == (1000, 256)


def test_weight_tying():
    """
    Verifies that weight tying is fully functional.
    Modifying the embedding weights must dynamically affect the projection weights.
    """
    args = ModelArgs(vocab_size=1000, dim=256)
    shell = TransformerEmbeddingShell(args, pad_token_id=0)

    # They should share the exact same underlying storage/memory
    assert shell.token_embeddings.weight is shell.lm_head.weight
    assert torch.equal(shell.token_embeddings.weight, shell.lm_head.weight)

    # In-place modification of token embeddings should directly mutate lm_head weight
    with torch.no_grad():
        shell.token_embeddings.weight.fill_(1.0)
    assert torch.equal(shell.token_embeddings.weight, shell.lm_head.weight)
    assert torch.all(shell.lm_head.weight == 1.0)


def test_padding_token_is_zero():
    """
    Verifies that the padding token embedding is strictly zero upon initialization.
    """
    args = ModelArgs(vocab_size=1000, dim=256)
    shell = TransformerEmbeddingShell(args, pad_token_id=5)

    # The 5th index should be strictly zero
    padding_weight = shell.token_embeddings.weight[5]
    assert torch.all(padding_weight == 0.0)


def test_embed_and_project_shapes():
    """
    Ensures that embed and project return correct output tensor shapes.
    """
    args = ModelArgs(vocab_size=1000, dim=256)
    shell = TransformerEmbeddingShell(args, pad_token_id=0)

    batch_size = 4
    seq_len = 16

    # Input: Token IDs (LongTensor)
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)

    # 1. Embed
    embeddings = shell.embed(input_ids)
    assert embeddings.shape == (batch_size, seq_len, 256)

    # 2. Project
    logits = shell.project(embeddings)
    assert logits.shape == (batch_size, seq_len, 1000)
