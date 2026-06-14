"""Build a word-level vocabulary from TinyStories data."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import pickle


def build_vocab(filepath, max_vocab=5000, min_freq=2):
    """Build a word-level tokenizer from a text file (one story per line).

    Returns:
        (word2idx, idx2word, encode_fn, decode_fn)
    """
    from collections import Counter

    word_counts = Counter()
    total_lines = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Simple word tokenization: split on whitespace, lowercase
            words = re.findall(r"[a-zA-Z0-9'']+(?:[''][a-zA-Z]+)?", line.lower())
            word_counts.update(words)
            total_lines += 1

    print(f"  Read {total_lines} stories, {len(word_counts)} unique words")

    # Filter by min_freq and cap vocab
    vocab = [(w, c) for w, c in word_counts.most_common(max_vocab)
             if c >= min_freq]

    # Build mappings
    word2idx = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3,
    }
    idx2word = {v: k for k, v in word2idx.items()}

    for i, (word, _) in enumerate(vocab):
        idx = i + 4
        word2idx[word] = idx
        idx2word[idx] = word

    print(f"  Vocabulary: {len(word2idx)} tokens ({len(vocab)} words + 4 special)")

    def encode(text):
        words = re.findall(r"[a-zA-Z0-9'']+(?:[''][a-zA-Z]+)?", text.lower())
        return [word2idx.get(w, word2idx['<UNK>']) for w in words]

    def decode(ids):
        return ' '.join(idx2word.get(i, '<UNK>') for i in ids)

    return word2idx, idx2word, encode, decode


if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'tinystories_5k.txt')
    vocab_path = os.path.join(os.path.dirname(data_path), 'vocab.pkl')

    print("Building vocabulary...")
    word2idx, idx2word, encode, decode = build_vocab(data_path, max_vocab=4000, min_freq=2)
    print(f"Vocab size: {len(word2idx)}")

    # Save
    with open(vocab_path, 'wb') as f:
        pickle.dump({'word2idx': word2idx, 'idx2word': idx2word}, f)
    print(f"Saved to {vocab_path}")

    # Test
    test_text = "Lily found a needle in her room"
    ids = encode(test_text)
    print(f"Encode: {test_text!r} -> {ids}")
    print(f"Decode: {ids} -> {decode(ids)}")
