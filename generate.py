import argparse
import os
import torch

from config import ModelArgs
from transformer.model import Transformer
from tokenizer.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained model checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="The start of the story/prompt to generate from",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (lower = more deterministic, higher = more creative)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Limit sampling to the top K most likely tokens (None to disable)",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default="./tokenizer/tokenizer_config",
        help="Directory containing the trained tokenizer",
    )
    args = parser.parse_args()

    # Determine execution device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    if not os.path.exists(args.tokenizer_dir):
        raise FileNotFoundError(f"Tokenizer directory '{args.tokenizer_dir}' not found.")
    tokenizer = Tokenizer(args.tokenizer_dir)

    # 2. Load Checkpoint file
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file '{args.checkpoint}' not found.")
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    # Register custom config classes as safe for PyTorch's unpickler
    torch.serialization.add_safe_globals([ModelArgs])
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Determine if this is a training checkpoint (dict with metadata) or raw state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("Detected intermediate training checkpoint...")
        state_dict = checkpoint["model_state_dict"]
        # Retrieve the configuration used to train this specific checkpoint
        model_args = checkpoint.get("args", ModelArgs())
        step = checkpoint.get("step", "unknown")
        print(f"Checkpoint step: {step}")
    else:
        print("Detected raw state dictionary (final model weights)...")
        state_dict = checkpoint
        # Fall back to default model configuration
        model_args = ModelArgs()

    print(f"Model architecture config: {model_args}")

    # 3. Instantiate model and load state weights
    model = Transformer(model_args)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode

    # 4. Process the prompt
    print(f"\nPrompt: {args.prompt}")
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    
    # Remove the closing token </s> if present so the model can continue generating naturally
    if prompt_ids and prompt_ids[-1] == tokenizer.tokenizer.eos_token_id:
        prompt_ids = prompt_ids[:-1]
        
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 5. Generate output autoregressively
    print("Generating...")
    with torch.no_grad():
        generated_tensor = model.generate(
            prompt_tensor,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        
    # 6. Decode and display the result
    generated_ids = generated_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    
    print("\n" + "=" * 50)
    print("                 GENERATED TEXT")
    print("=" * 50)
    print(generated_text)
    print("=" * 50)


if __name__ == "__main__":
    main()
