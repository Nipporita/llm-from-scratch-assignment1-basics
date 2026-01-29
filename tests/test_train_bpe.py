import json
import time

if __name__ == "__main__":
    from adapters import run_train_bpe
    from common import FIXTURES_PATH, gpt2_bytes_to_unicode
else:
    from .adapters import run_train_bpe
    from .common import FIXTURES_PATH, gpt2_bytes_to_unicode


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )

if __name__ == "__main__":
    # test_train_bpe()
    
    import pickle
    from pathlib import Path
    import regex as re
    
    PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    PAT = re.compile(PATTERN)

    snapshot_path = Path(r"/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/tests/_snapshots/test_train_bpe_special_tokens.pkl")

    with open(snapshot_path, "rb") as f:
        snapshot_data = pickle.load(f)

    reference_merges = snapshot_data["merges"]
    
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    alter_path = FIXTURES_PATH / "tinystories_sample_5M_alter.txt"
    
    alt_special_token = "<|endoftext|>"
    
    with open(input_path, "r", encoding="utf-8") as f, open(alter_path, "w", encoding="utf-8") as f_out:
        data = f.read()
        data.replace("<|endoftext|>", alt_special_token)
        f_out.write(data)
    
    vocab, merges = run_train_bpe(
        input_path=alter_path,
        vocab_size=1000,
        special_tokens=[alt_special_token],
    )
    
    for i in range(max(len(merges), len(reference_merges))):
        merges_i = merges[i] if i < len(merges) else None
        reference_merges_i = reference_merges[i] if i < len(reference_merges) else None
        
        print(f"Merge {i}: Computed: {merges_i}, Reference: {reference_merges_i}")
        if merges_i != reference_merges_i:
            print("Mismatch found!")
