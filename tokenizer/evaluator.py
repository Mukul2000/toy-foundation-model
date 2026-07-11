import argparse
import os
from typing import List

from datasets import load_dataset

from tokenizer import Tokenizer


def evaluate_tokenizer(tokenizer: Tokenizer, texts: List[str]):
    """
    Evaluates the tokenizer on a list of texts and returns key metrics.
    """
    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    total_words = 0
    single_char_or_byte_tokens = 0

    print("Evaluating tokenizer metrics...")

    for text in texts:
        if not text.strip():
            continue

        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))

        # Split by whitespace to approximate raw words
        words = text.split()
        total_words += len(words)

        # Encode (excluding special tokens to avoid skewing standard compression/fertility)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)

        # Count single-character/byte tokens robustly by decoding individually
        for tok_id in tokens:
            try:
                decoded_tok = tokenizer.decode([tok_id], skip_special_tokens=True)
                if len(decoded_tok) <= 1:
                    single_char_or_byte_tokens += 1
            except Exception:
                single_char_or_byte_tokens += 1

    compression_bytes_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    compression_chars_ratio = total_chars / total_tokens if total_tokens > 0 else 0
    fertility = total_tokens / total_words if total_words > 0 else 0
    fallback_rate = (
        (single_char_or_byte_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    )

    return {
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "total_words": total_words,
        "compression_bytes_ratio": compression_bytes_ratio,
        "compression_chars_ratio": compression_chars_ratio,
        "fertility": fertility,
        "fallback_rate": fallback_rate,
        "avg_token_length_chars": total_chars / total_tokens if total_tokens > 0 else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained tokenizer")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./english_tokenizer",
        help="Directory of the trained tokenizer",
    )
    parser.add_argument(
        "--num-eval-samples",
        type=int,
        default=500,
        help="Number of Wikipedia articles to use for evaluation",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(
            f"Error: Trained tokenizer not found at '{args.model_dir}'. Run tokenizer/tokenizer.py first."
        )
        exit(1)

    # 1. Load the trained tokenizer
    tokenizer = Tokenizer(args.model_dir)

    # 2. Load validation dataset (using the end of the Wikipedia train split to avoid re-downloads)
    dataset_name = "wikimedia/wikipedia"
    dataset_config = "20231101.en"
    print(f"Loading evaluation dataset from '{dataset_name}'...")
    dataset = load_dataset(dataset_name, dataset_config, split="train")

    # Pick validation articles from the end of the dataset
    start_idx = len(dataset) - args.num_eval_samples
    eval_texts = [dataset[i]["text"] for i in range(start_idx, len(dataset))]

    # 3. Compute Metrics
    metrics = evaluate_tokenizer(tokenizer, eval_texts)

    # 4. Print Results
    print("\n" + "=" * 40)
    print("        TOKENIZER EVALUATION METRICS")
    print("=" * 40)
    print(f"Dataset:                  {dataset_name} ({dataset_config})")
    print(f"Evaluation Samples:       {args.num_eval_samples:,} articles")
    print("-" * 40)
    print(f"Total Characters:         {metrics['total_chars']:,}")
    print(f"Total Bytes (UTF-8):      {metrics['total_bytes']:,}")
    print(f"Total Words (Whitespace): {metrics['total_words']:,}")
    print(f"Total Tokens Generated:   {metrics['total_tokens']:,}")
    print("-" * 40)
    print(f"Compression Ratio (Bytes): {metrics['compression_bytes_ratio']:.3f}x")
    print(f"Compression Ratio (Chars): {metrics['compression_chars_ratio']:.3f}x")
    print(f"Fertility (Tokens/Word):   {metrics['fertility']:.3f}")
    print(f"Byte Fallback Rate:        {metrics['fallback_rate']:.2f}%")
    print(f"Avg Token Length (Chars):  {metrics['avg_token_length_chars']:.2f}")
    print("=" * 40)
    print("\nExplanation of Metrics:")
    print(
        "- Compression Ratio (Bytes): Average number of bytes represented by each token."
    )
    print("  Higher is better. Values around 3.0x - 4.0x are typical for good BPE.")
    print(
        "- Fertility: Average number of tokens produced per whitespace-separated word."
    )
    print("  Closer to 1.0 is better (lower fragmentation of words).")
    print(
        "- Byte Fallback Rate: Percentage of tokens that fallback to single characters or bytes."
    )
    print("  Lower is better (means BPE is capturing larger merged combinations).")
