import argparse
import os
import sys
from typing import Iterable, List, Optional, Union

from datasets import load_dataset
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

# Add parent directory to sys.path to resolve imports when running as a direct script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelArgs


class Tokenizer:
    def __init__(self, model_dir: Optional[str] = None):
        """
        A wrapper class around Hugging Face's tokenizers for easy BPE training,
        encoding, and decoding.
        """
        self.tokenizer = None
        if model_dir and os.path.exists(model_dir):
            self.load(model_dir)

    def train(
        self,
        texts: Union[Iterable[str], List[str]],
        vocab_size: int = ModelArgs.vocab_size,
        save_dir: Optional[str] = None,
    ):
        """
        Trains a Byte-Level BPE tokenizer on the given iterable of texts
        (or a list of file paths).

        Args:
            texts: An iterable of strings (e.g. generator, dataset, list of strings)
                   or a list of file paths on disk.
            vocab_size: Total vocabulary size to learn.
            save_dir: Optional directory to save the trained tokenizer.
        """
        print(f"Training BBPE Tokenizer (Vocab Size: {vocab_size})...")

        # 1. Initialize BPE Model
        base_tokenizer = HFTokenizer(BPE())

        # 2. Pre-tokenization: Splits text by spaces and punctuation, converting to bytes
        base_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

        # 3. Configure Trainer with special tokens:
        # <pad> = Padding (ID 0)
        # <s>   = Start of Text / BOS (ID 1)
        # </s>  = End of Text / EOS (ID 2)
        # <unk> = Unknown (ID 3)
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
            initial_alphabet=ByteLevel.alphabet(),
        )

        # 4. Train the tokenizer
        # Check if texts is a list of existing files
        if isinstance(texts, list) and all(
            isinstance(x, str) and os.path.exists(x) for x in texts
        ):
            base_tokenizer.train(texts, trainer)
        else:
            base_tokenizer.train_from_iterator(texts, trainer)

        # 5. Set up the Decoder to convert bytes back to readable strings
        base_tokenizer.decoder = ByteLevelDecoder()

        # 6. Post-processing: Automatically wrap sequences with <s> and </s>
        # TemplateProcessing automatically wraps the encoded sequence
        base_tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            special_tokens=[
                ("<s>", base_tokenizer.token_to_id("<s>")),
                ("</s>", base_tokenizer.token_to_id("</s>")),
            ],
        )

        # 7. Wrap in Hugging Face's PreTrainedTokenizerFast for PyTorch compatibility
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            clean_up_tokenization_spaces=False,
        )

        # 8. Save the tokenizer if a save directory is provided
        if save_dir:
            self.save(save_dir)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encodes text into a list of token IDs.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been trained or loaded yet.")

        # PreTrainedTokenizerFast's encode method returns list of ints
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decodes a list of token IDs back into a string.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been trained or loaded yet.")

        decoded = self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        return str(decoded)

    def save(self, save_dir: str):
        """
        Saves the tokenizer files (tokenizer.json, config, special tokens) to a directory.
        """
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save.")

        os.makedirs(save_dir, exist_ok=True)
        self.tokenizer.save_pretrained(save_dir)
        print(f"Tokenizer files successfully saved to: {save_dir}")

    def load(self, model_dir: str):
        """
        Loads the tokenizer from the specified directory.
        """
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Directory {model_dir} does not exist.")

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
        print(f"Tokenizer successfully loaded from: {model_dir}")

    def save_vocabulary(self, file_path: str):
        """
        Saves the human-readable vocabulary mapping (ID -> Token representation) to a file.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer has not been trained or loaded yet.")

        vocab = self.tokenizer.get_vocab()
        # Sort by token ID
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for token, token_id in sorted_vocab:
                f.write(f"{token_id}\t{token}\n")
        print(f"Vocabulary successfully saved to: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train custom BPE tokenizer on Wikipedia"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50000,
        help="Number of Wikipedia articles to train on (None for all)",
    )
    args = parser.parse_args()

    save_dir = "./tokenizer/tokenizer_config"
    vocab_file = "./tokenizer/tokenizer_config/vocab.txt"

    dataset_name = "wikimedia/wikipedia"
    dataset_config = "20231101.en"

    print(f"Loading Hugging Face '{dataset_name}' ({dataset_config}) dataset...")
    # Load the train split
    dataset = load_dataset(dataset_name, dataset_config, split="train")

    # Define a generator that yields text from the dataset
    def iterator():
        limit = (
            min(args.num_samples, len(dataset))
            if args.num_samples is not None
            else len(dataset)
        )
        print(f"Streaming {limit:,} articles from the dataset for training...")
        for i in range(limit):
            yield dataset[i]["text"]

    # Instantiate our Custom Tokenizer
    tok = Tokenizer()

    # Train it!
    tok.train(iterator(), vocab_size=ModelArgs.vocab_size, save_dir=save_dir)

    # Save vocabulary file
    tok.save_vocabulary(vocab_file)

    # Simple demonstration of encode / decode
    print("\n--- Tokenizer Demo ---")
    test_text = (
        "Once upon a time, there was a small model that wanted to learn English."
    )
    encoded = tok.encode(test_text)
    decoded = tok.decode(encoded)

    print(f"Original text: {test_text}")
    print(f"Encoded IDs:   {encoded}")
    print(f"Decoded text:  {decoded}")
