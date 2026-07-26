import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelArgs
from embedding_layer.embedding_shell import TransformerEmbeddingShell
from .transformer_block import TransformerBlock
from .rmsnorm import RMSNorm


class Transformer(nn.Module):
    """
    The complete Decoder-Only Transformer Foundation Model.
    Combines Tied Embeddings, multiple Transformer Blocks (with MHA, RoPE, SwiGLU FFN),
    a Final RMSNorm, and an autoregressive text generator.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.vocab_size = args.vocab_size
        self.max_seq_len = args.max_seq_len

        # 1. Input/Output embedding layer (with weight tying)
        self.embeddings = TransformerEmbeddingShell(args)

        # 2. Stack of Transformer Decoder layers
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])

        # 3. Final normalization layer
        self.norm = RMSNorm(dim=args.dim, eps=args.norm_eps)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Public Forward Pass:
        idx: Input token IDs of shape (B, T)
        targets: Optional ground-truth next-token targets of shape (B, T)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Cannot process sequence of length {T}, max is {self.max_seq_len}"

        # 1. Project Token IDs to Embeddings
        x = self.embeddings.embed(idx)

        # 2. Pass through the stack of Transformer Blocks
        for layer in self.layers:
            x = layer(x)

        # 3. Apply the final normalization
        x = self.norm(x)

        # 4. Project hidden states back to vocabulary space (Logits)
        logits = self.embeddings.project(x)

        # 5. If targets are provided, compute shifted Cross-Entropy Loss
        loss = None
        if targets is not None:
            loss = self._calculate_loss(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        Public Autoregressive Text Generation:
        idx: Starting prompt token IDs of shape (B, T)
        max_new_tokens: Number of new tokens to generate
        temperature: Controls randomness (lower = more deterministic, higher = more creative)
        top_k: Optional constraint to only sample from the top K most likely tokens
        """
        for _ in range(max_new_tokens):
            # If the sequence grows larger than max_seq_len, crop it to fit our cached mask
            idx_cond = idx if idx.shape[1] <= self.max_seq_len else idx[:, -self.max_seq_len:]

            # Forward pass to get raw logits
            logits, _ = self(idx_cond)

            # Focus only on the logits of the last token in the sequence
            logits = logits[:, -1, :]  # Shape: (B, vocab_size)

            # Apply Temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

            # Apply optional Top-K constraint
            if top_k is not None:
                logits = self._apply_top_k(logits, top_k)

            # Apply Softmax to convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token ID from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)  # Shape: (B, 1)

            # Append the newly generated token to the running sequence
            idx = torch.cat((idx, next_token), dim=-1)

        return idx

    @staticmethod
    def _calculate_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Private helper: Computes next-token Cross-Entropy loss by shifting
        logits and targets by 1 position so the model learns to predict the future.
        """
        # Shift logits and targets:
        # logits[B, T-1, V] predicts targets[B, 1 to T-1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()

        # Flatten tensors for PyTorch's cross_entropy
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1)
        )

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """
        Private helper: Limits sampling to the top K most probable tokens
        by setting all other token logits to -inf.
        """
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        # Set all values below the top-k threshold to -inf
        logits[logits < v[:, [-1]]] = float("-inf")
        return logits
