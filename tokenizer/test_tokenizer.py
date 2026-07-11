import os
import shutil
import tempfile

import pytest

from tokenizer import Tokenizer


@pytest.fixture(scope="module")
def trained_tokenizer():
    """
    Returns a tokenizer trained on a small, synthetic corpus for testing.
    """
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "To be, or not to be, that is the question.",
        "Hello world! This is a robust tokenizer test suite.",
        "Testing UTF-8 characters: 🦄 🐼 ⚡ 🚀.",
    ]
    tok = Tokenizer()
    tok.train(corpus, vocab_size=1000)
    return tok


def test_encode_decode_roundtrip(trained_tokenizer):
    """
    Verifies that encoding and then decoding any text recovers the original string exactly.
    """
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Simple english text.",
        "Numbers: 1234567890, punctuation: !@#$%^&*()_+.",
        "Unicode emojis: 🚀 🦄 🐼.",
        "Multi-line\ntext\nwith\ttabs.",
    ]

    for sentence in test_sentences:
        encoded = trained_tokenizer.encode(sentence, add_special_tokens=False)
        decoded = trained_tokenizer.decode(encoded, skip_special_tokens=True)
        assert decoded == sentence, f"Failed roundtrip for: '{sentence}'"


def test_special_tokens(trained_tokenizer):
    """
    Verifies that the special tokens are positioned correctly when add_special_tokens=True.
    """
    text = "Hello world."
    encoded_with_special = trained_tokenizer.encode(text, add_special_tokens=True)
    encoded_without_special = trained_tokenizer.encode(text, add_special_tokens=False)

    # <s> is at the beginning, </s> is at the end
    assert encoded_with_special[0] == 1, "BOS token <s> (ID 1) should be at the start"
    assert encoded_with_special[-1] == 2, "EOS token </s> (ID 2) should be at the end"
    assert encoded_with_special[1:-1] == encoded_without_special, (
        "Inner tokens should match exactly"
    )


def test_special_token_mapping(trained_tokenizer):
    """
    Asserts that special tokens are correctly mapped to their designated invariant IDs.
    """
    fast_tok = trained_tokenizer.tokenizer
    assert fast_tok is not None

    assert fast_tok.pad_token_id == 0
    assert fast_tok.bos_token_id == 1
    assert fast_tok.eos_token_id == 2
    assert fast_tok.unk_token_id == 3

    assert fast_tok.decode([0], skip_special_tokens=False) == "<pad>"
    assert fast_tok.decode([1], skip_special_tokens=False) == "<s>"
    assert fast_tok.decode([2], skip_special_tokens=False) == "</s>"
    assert fast_tok.decode([3], skip_special_tokens=False) == "<unk>"


def test_save_and_load(trained_tokenizer):
    """
    Verifies that a tokenizer can be serialized to disk and loaded back with identical behavior.
    """
    test_text = "Checking serialization of our tokenizer model."
    original_encoded = trained_tokenizer.encode(test_text, add_special_tokens=True)

    # Use a secure temporary directory for serialization test
    temp_dir = tempfile.mkdtemp()
    try:
        # Save
        trained_tokenizer.save(temp_dir)

        # Assert files are actually written
        assert os.path.exists(os.path.join(temp_dir, "tokenizer.json"))
        assert os.path.exists(os.path.join(temp_dir, "tokenizer_config.json"))

        # Load into a new Tokenizer instance
        loaded_tokenizer = Tokenizer(temp_dir)
        loaded_encoded = loaded_tokenizer.encode(test_text, add_special_tokens=True)

        assert original_encoded == loaded_encoded, (
            "Loaded tokenizer encoding does not match original"
        )

        # Check vocabulary text export
        vocab_file = os.path.join(temp_dir, "vocab.txt")
        loaded_tokenizer.save_vocabulary(vocab_file)
        assert os.path.exists(vocab_file)

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def test_empty_input(trained_tokenizer):
    """
    Ensures that empty strings or whitespace-only strings are handled gracefully.
    """
    assert trained_tokenizer.encode("", add_special_tokens=False) == []
    assert trained_tokenizer.decode([], skip_special_tokens=True) == ""

    # With special tokens, it should only contain BOS and EOS
    assert trained_tokenizer.encode("", add_special_tokens=True) == [1, 2]
