import argparse
import math
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

from config import ModelArgs
from train_config import TrainArgs
from transformer.model import Transformer
from tokenizer.tokenizer import Tokenizer


class TokenizedIterableDataset(IterableDataset):
    """
    An IterableDataset that streams stories, tokenizes them,
    and packs them into constant-length sequences of max_seq_len.
    This avoids padding overhead.
    """
    def __init__(self, hf_dataset, tokenizer, max_seq_len):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __iter__(self):
        buffer = []
        for item in self.hf_dataset:
            # Encode text, wrapping with <s> and </s>
            ids = self.tokenizer.encode(item["text"], add_special_tokens=True)
            buffer.extend(ids)
            
            # Yield full chunks of size max_seq_len + 1
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[:self.max_seq_len + 1]
                # inputs = x[:-1], targets = x[1:]
                yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
                buffer = buffer[self.max_seq_len:]


def get_lr_multiplier(step, warmup_steps, max_steps):
    """
    Helper for LambdaLR: computes linear warmup followed by cosine decay.
    """
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    
    # Progress from 0 to 1 after warmup
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    # Keep it capped at 1.0 just in case
    progress = min(1.0, progress)
    
    # Cosine decay down to 10% of peak learning rate
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_loader, eval_steps, device):
    """
    Computes average loss on a limited number of validation batches.
    """
    model.eval()
    val_loss = 0.0
    steps = 0
    
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, loss = model(inputs, targets)
        val_loss += loss.item()
        steps += 1
        if steps >= eval_steps:
            break
            
    model.train()
    return val_loss / max(1, steps)


def main():
    default_args = TrainArgs()
    parser = argparse.ArgumentParser(description="Pre-train the toy foundation model")
    parser.add_argument("--batch-size", type=int, default=default_args.batch_size, help="Batch size for training")
    parser.add_argument("--max-steps", type=int, default=default_args.max_steps, help="Total training steps")
    parser.add_argument("--lr", type=float, default=default_args.lr, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=default_args.warmup_steps, help="LR linear warmup steps")
    parser.add_argument("--weight-decay", type=float, default=default_args.weight_decay, help="Weight decay factor")
    parser.add_argument("--grad-clip", type=float, default=default_args.grad_clip, help="Gradient clipping max norm")
    parser.add_argument("--eval-interval", type=int, default=default_args.eval_interval, help="Steps between evaluations")
    parser.add_argument("--eval-steps", type=int, default=default_args.eval_steps, help="Batches to use for evaluation")
    parser.add_argument("--save-interval", type=int, default=default_args.save_interval, help="Steps between checkpoint saves")
    parser.add_argument("--checkpoint-dir", type=str, default=default_args.checkpoint_dir, help="Directory to save checkpoints")
    parser.add_argument("--tokenizer-dir", type=str, default=default_args.tokenizer_dir, help="Directory containing trained tokenizer")
    parser.add_argument("--tiny-model", action="store_true", help="Override config with a tiny 2-layer model for local prototyping")
    parsed = parser.parse_args()

    args = TrainArgs(
        batch_size=parsed.batch_size,
        max_steps=parsed.max_steps,
        lr=parsed.lr,
        warmup_steps=parsed.warmup_steps,
        weight_decay=parsed.weight_decay,
        grad_clip=parsed.grad_clip,
        eval_interval=parsed.eval_interval,
        eval_steps=parsed.eval_steps,
        save_interval=parsed.save_interval,
        checkpoint_dir=parsed.checkpoint_dir,
        tokenizer_dir=parsed.tokenizer_dir,
        tiny_model=parsed.tiny_model,
    )

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Tokenizer
    print(f"Loading tokenizer from: {args.tokenizer_dir}")
    tokenizer = Tokenizer(args.tokenizer_dir)

    # Initialize Model Configuration
    model_args = ModelArgs()
    if args.tiny_model:
        print("Configuring a tiny model for CPU prototyping...")
        model_args = ModelArgs(
            dim=128,
            n_layers=2,
            n_heads=2,
            max_seq_len=256,
            vocab_size=ModelArgs.vocab_size
        )
    
    print(f"Model Configuration: {model_args}")
    model = Transformer(model_args).to(device)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load TinyStories dataset (using streaming)
    print("Loading TinyStories dataset (streaming mode)...")
    train_dataset_raw = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    val_dataset_raw = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    # Wrap HF datasets with our tokenized generator
    train_dataset = TokenizedIterableDataset(train_dataset_raw, tokenizer, model_args.max_seq_len)
    val_dataset = TokenizedIterableDataset(val_dataset_raw, tokenizer, model_args.max_seq_len)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Optimizer (AdamW with weight decay)
    # We filter out parameters that shouldn't get weight decay (like biases and layer norms)
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # Learning rate scheduler (Warmup + Cosine)
    lr_lambda = lambda step: get_lr_multiplier(step, args.warmup_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create checkpoint directory (using absolute path for robustness)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training state
    step = 0
    start_time = time.time()
    last_step_time = time.time()
    
    # Iterate over the training stream
    train_iter = iter(train_loader)
    
    model.train()
    print("Starting training loop...")
    while step < args.max_steps:
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            # Reinitialize the stream if we run out of data (unlikely with streaming datasets)
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)

        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        logits, loss = model(inputs, targets)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()
        scheduler.step()

        step += 1

        # Print training progress
        if step % 20 == 0 or step == 1:
            current_lr = scheduler.get_last_lr()[0]
            step_time = time.time() - last_step_time
            tokens_per_sec = (args.batch_size * model_args.max_seq_len) / step_time
            print(
                f"Step {step}/{args.max_steps} | "
                f"Loss: {loss.item():.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Step Time: {step_time*1000:.0f}ms | "
                f"Tokens/sec: {tokens_per_sec:.0f}"
            )
            last_step_time = time.time()

        # Run periodic evaluations and print sample generation
        if step % args.eval_interval == 0:
            print("\n" + "-" * 50)
            print(f"Running evaluation at step {step}...")
            val_loss = evaluate(model, val_loader, args.eval_steps, device)
            val_perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")
            print(f"Validation Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.2f}")

            # Generate a sample story to visualize progress
            print("Generating demo story...")
            model.eval()
            prompt = "Once upon a time, a little girl"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            # Remove the closing token </s> so the model can continue generating
            if prompt_ids and prompt_ids[-1] == tokenizer.tokenizer.eos_token_id:
                prompt_ids = prompt_ids[:-1]
                
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated_tensor = model.generate(prompt_tensor, max_new_tokens=60, temperature=0.7, top_k=10)
            generated_ids = generated_tensor[0].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            
            print(f"Prompt: {prompt}")
            print(f"Generated text:\n{generated_text}")
            print("-" * 50 + "\n")
            model.train()
            last_step_time = time.time()

        # Save model checkpoint
        if step % args.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")
            print(f"Saving checkpoint to: {checkpoint_path}")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
                "args": model_args,
            }, checkpoint_path)
            last_step_time = time.time()

    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes.")

    # Save final model
    final_path = os.path.join(checkpoint_dir, "model_final.pt")
    print(f"Saving final model to: {final_path}")
    torch.save(model.state_dict(), final_path)


if __name__ == "__main__":
    main()
