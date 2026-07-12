import torch
import torch.nn as nn

from config import ModelArgs


class TransformerEmbeddingShell(nn.Module):
    def __init__(self, args: ModelArgs, pad_token_id: int = 0):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.hidden_dim = args.dim

        # 1. The Input Token Embedding Layer
        self.token_embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_dim,
            padding_idx=pad_token_id,
        )

        # 2. The Output LM Head (projects hidden_dim back to vocab_size)
        # We set bias=False because we are tying weights with the embedding layer
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        # 3. Apply Weight Tying
        # This links the memory of both layers. Updating one updates the other.
        self.lm_head.weight = self.token_embeddings.weight

        # 4. Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Standard initialization for transformer embeddings
        # We use a small standard deviation to keep initial gradients stable
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)

        # Ensure the padding token embedding remains strictly zero
        if self.token_embeddings.padding_idx is not None:
            with torch.no_grad():
                self.token_embeddings.weight[self.token_embeddings.padding_idx].fill_(
                    0.0
                )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Step 1: Map Token IDs to Continuous Vectors
        Input shape:  (batch_size, seq_len)
        Output shape: (batch_size, seq_len, hidden_dim)
        """
        return self.token_embeddings(input_ids)

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Step 3: Map final hidden states back to vocabulary logits
        Input shape:  (batch_size, seq_len, hidden_dim)
        Output shape: (batch_size, seq_len, vocab_size)
        """
        return self.lm_head(hidden_states)
