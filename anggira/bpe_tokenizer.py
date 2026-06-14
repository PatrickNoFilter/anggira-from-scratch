"""Pure NumPy BPE Tokenizer — train, encode, decode.

Algorithm: Byte Pair Encoding
- Start with individual chars (plus special tokens)
- Iteratively merge most frequent adjacent pair
- Vocab size target: ~5000 (matching current model scale)

Usage:
    from anggira.bpe_tokenizer import BPETokenizer
    tok = BPETokenizer()
    tok.train(texts, vocab_size=5000)
    ids = tok.encode("text")
    text = tok.decode(ids)
"""

import re
from collections import Counter, defaultdict
import pickle


class BPETokenizer:
    """Byte-Pair Encoding tokenizer with special tokens."""

    def __init__(self):
        self.vocab = {}       # id -> bytes representation
        self.merges = {}      # pair -> id
        self.idx2token = {}   # id -> printable token
        self.token2idx = {}   # printable token -> id
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self.next_id = 4
        self.vocab_size = 0

    def _get_stats(self, ids):
        """Count adjacent pairs."""
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

    def _merge(self, ids, pair, new_id):
        """Replace all occurrences of pair with new_id."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, texts, vocab_size=5000, verbose=True):
        """Train BPE on a list of text strings."""
        # Build initial char-level vocabulary
        # First add special tokens
        for tok, idx in self.special_tokens.items():
            self.vocab[idx] = bytes(tok, 'utf-8')
            self.idx2token[idx] = tok
            self.token2idx[tok] = idx

        # Get all unique characters from the text
        chars = set()
        for text in texts:
            for ch in text:
                chars.add(ch)

        # Add characters to vocab
        char_ids = {}
        for ch in sorted(chars):
            if ch not in self.token2idx:
                tid = self.next_id
                self.next_id = tid + 1
                self.vocab[tid] = bytes(ch, 'utf-8')
                self.idx2token[tid] = ch
                self.token2idx[ch] = tid
                char_ids[ch] = tid

        if verbose:
            print(f"  Base chars: {len(chars)} unique")

        # Convert all texts to IDs (char level)
        all_ids = []
        for text in texts:
            ids = [char_ids.get(c, self.special_tokens['<UNK>']) for c in text]
            all_ids.append(ids)

        # Also add a corpus-level ID stream for pair stats
        corpus_ids = []
        for ids in all_ids:
            corpus_ids.extend(ids)

        current_vocab = len(self.token2idx)
        target_merges = vocab_size - current_vocab

        if verbose:
            print(f"  Starting merges: need {target_merges} to reach vocab_size={vocab_size}")

        merged = 0
        merge_list = []
        while len(self.token2idx) < vocab_size and merged < target_merges:
            # Compute pair frequencies across entire corpus
            # (re-sample each iteration since IDs change)
            stats = self._get_stats(corpus_ids)
            if not stats:
                break
            most_common = max(stats.items(), key=lambda x: x[1])
            pair, freq = most_common
            if freq < 2:
                break

            # Create new token
            new_id = self.next_id
            self.next_id = new_id + 1

            # The merged token's printable representation
            t1 = self.idx2token[pair[0]]
            t2 = self.idx2token[pair[1]]
            merged_token = t1 + t2

            # Store merge and vocab
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.idx2token[new_id] = merged_token
            self.token2idx[merged_token] = new_id

            # Apply merge in corpus
            corpus_ids = self._merge(corpus_ids, pair, new_id)

            merged += 1
            merge_list.append((merged_token, freq))

            if verbose and (merged % 500 == 0 or merged == target_merges or merged < 5):
                print(f"  Merge {merged:4d}/{target_merges}: '{merged_token}' freq={freq}")

        self.vocab_size = len(self.token2idx)
        if verbose:
            print(f"  Vocab: {self.vocab_size} tokens ({merged} merges)")

    def encode(self, text, bos=False, eos=False):
        """Encode text to token IDs using the learned merges.

        Args:
            text: input string
            bos: prepend <BOS> token
            eos: append <EOS> token
        Returns:
            list of token IDs
        """
        ids = []
        if bos:
            ids.append(self.special_tokens['<BOS>'])

        # Convert to char IDs first
        char_ids = []
        for ch in text:
            if ch in self.token2idx:
                char_ids.append(self.token2idx[ch])
            else:
                char_ids.append(self.special_tokens['<UNK>'])

        # Apply merges in the order they were learned
        # We need the merges sorted by priority (first learned has highest priority)
        # Actually BPE applies higher-frequency merges first, but the merge order
        # matters: earlier merges get applied before later merges
        # For encoding, we repeatedly scan for merges
        while True:
            stats = self._get_stats(char_ids)
            # Find the lowest-id merge that exists in this text
            best_pair = None
            best_merge_id = None
            for pair in stats:
                if pair in self.merges:
                    merge_id = self.merges[pair]
                    if best_merge_id is None or merge_id < best_merge_id:
                        best_pair = pair
                        best_merge_id = merge_id

            if best_pair is None:
                break
            char_ids = self._merge(char_ids, best_pair, best_merge_id)

        ids.extend(char_ids)
        if eos:
            ids.append(self.special_tokens['<EOS>'])
        return ids

    def decode(self, ids):
        """Decode token IDs back to text."""
        tokens = []
        for tid in ids:
            if tid in self.vocab:
                try:
                    tokens.append(self.vocab[tid].decode('utf-8', errors='replace'))
                except:
                    tokens.append('<UNK>')
            elif tid in self.special_tokens.values():
                tokens.append('')
            else:
                tokens.append('<UNK>')
        return ''.join(tokens)

    def save(self, path):
        """Save tokenizer to pickle."""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'idx2token': self.idx2token,
            'token2idx': self.token2idx,
            'special_tokens': self.special_tokens,
            'next_id': self.next_id,
            'vocab_size': self.vocab_size,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        """Load tokenizer from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vocab = data['vocab']
        self.merges = data['merges']
        self.idx2token = data['idx2token']
        self.token2idx = data['token2idx']
        self.special_tokens = data['special_tokens']
        self.next_id = data['next_id']
        self.vocab_size = data['vocab_size']
