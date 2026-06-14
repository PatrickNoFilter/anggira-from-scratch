"""
Anggira NLP — Subword Tokenization, Word Embeddings, Seq2Seq, and Attention

Phase 11: NLP from Scratch
- BPE (Byte-Pair Encoding) tokenizer with train() and encode()
- Word2Vec (Skip-gram with negative sampling)
- Sequence-to-sequence with fixed context vector
- Bahdanau (additive) and Luong (dot-product) attention mechanisms
"""

import math
import random
from collections import Counter, defaultdict


# ═══════════════════════════════════════════════
# BPE TOKENIZER
# ═══════════════════════════════════════════════

class BPETokenizer:
    """Byte-Pair Encoding subword tokenizer.

    Trains merge rules from a corpus and encodes new text
    by greedily applying the learned merges.

    Attributes:
        merges: list of (left, right) pairs in merge order
        vocab: set of all subword tokens known to the tokenizer
        special_tokens: dict mapping name -> token string
    """

    def __init__(self, special_tokens=None):
        self.merges = []          # list of (left, right) in merge order
        self.vocab = set()        # all known subword tokens
        self.special_tokens = special_tokens or {}
        self._merge_rank = {}     # (left, right) -> rank (lower = learned earlier)

    def _get_stats(self, word_freqs):
        """Count adjacent character/subword pairs in the corpus.

        Args:
            word_freqs: dict of word (list of tokens) -> frequency

        Returns:
            Counter of (token_i, token_j) -> count
        """
        stats = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                stats[(word[i], word[i + 1])] += freq
        return stats

    def _merge_pair(self, pair, word_freqs):
        """Merge a specific pair across all words in the corpus.

        Args:
            pair: (left, right) tokens to merge
            word_freqs: dict of word (list of tokens) -> frequency (modified in-place)

        Returns:
            updated word_freqs dict
        """
        result = {}
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if (i < len(word) - 1 and word[i] == pair[0]
                        and word[i + 1] == pair[1]):
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            result[tuple(new_word)] = freq
        return result

    def train(self, corpus, num_merges=100, min_frequency=2):
        """Learn BPE merge rules from a text corpus.

        Args:
            corpus: iterable of strings (sentences/documents)
            num_merges: maximum number of merge operations to learn
            min_frequency: minimum frequency for a pair to be merged
        """
        # Build initial word frequencies (character-level)
        word_freqs = Counter()
        for text in corpus:
            # Split into words, then to characters + special end token
            for word in text.strip().split():
                tokenized = list(word) + ["</w>"]
                word_freqs[tuple(tokenized)] += 1

        # Initial vocab = all unique characters + special tokens
        self.vocab = set()
        for word in word_freqs:
            for token in word:
                self.vocab.add(token)
        for token in self.special_tokens.values():
            self.vocab.add(token)

        # Learn merges
        self.merges = []
        self._merge_rank = {}
        for rank in range(num_merges):
            stats = self._get_stats(word_freqs)

            if not stats:
                break

            # Find most frequent pair
            best_pair = max(stats, key=lambda p: stats[p])

            if stats[best_pair] < min_frequency:
                break

            # Record the merge
            self.merges.append(best_pair)
            self._merge_rank[best_pair] = rank
            new_token = best_pair[0] + best_pair[1]
            self.vocab.add(new_token)

            # Apply the merge
            word_freqs = self._merge_pair(best_pair, word_freqs)

    def encode(self, text):
        """Tokenize text using learned BPE merges.

        Args:
            text: string to tokenize

        Returns:
            list of subword token strings
        """
        tokens = []
        for word in text.strip().split():
            # Start with character-level representation
            word_tokens = list(word) + ["</w>"]
            # Greedily apply merges from earliest to latest
            changed = True
            while changed:
                changed = False
                i = 0
                while i < len(word_tokens) - 1:
                    pair = (word_tokens[i], word_tokens[i + 1])
                    if pair in self._merge_rank:
                        word_tokens[i] = word_tokens[i] + word_tokens[i + 1]
                        del word_tokens[i + 1]
                        changed = True
                        # Restart from beginning after each merge
                        # to allow longer merges to take effect
                        break
                    i += 1
            tokens.extend(word_tokens)
        return tokens

    def decode(self, tokens):
        """Convert token list back to string.

        Args:
            tokens: list of subword token strings

        Returns:
            reconstructed string
        """
        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()


# ═══════════════════════════════════════════════
# WORD2VEC (SKIP-GRAM WITH NEGATIVE SAMPLING)
# ═══════════════════════════════════════════════

class Word2Vec:
    """Skip-gram Word2Vec with negative sampling.

    Learns dense vector representations for words by predicting
    context words from a target word, using negative sampling
    to make training efficient.

    Attributes:
        vocab_size: number of unique words
        embedding_dim: dimensionality of word vectors
        learning_rate: step size for gradient updates
        word_to_idx: dict mapping word -> index
        idx_to_word: dict mapping index -> word
        in_vectors: list of input (target) word vectors
        out_vectors: list of output (context) word vectors
    """

    def __init__(self, embedding_dim=50, learning_rate=0.01):
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.vocab_size = 0
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.in_vectors = []   # input embeddings  (vocab_size × dim)
        self.out_vectors = []  # output embeddings (vocab_size × dim)

    def _sigmoid(self, x):
        if x > 20:
            return 1.0
        if x < -20:
            return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _build_vocab(self, corpus):
        """Build vocabulary from corpus.

        Args:
            corpus: list of tokenized sentences (list of list of str)
        """
        word_counts = Counter()
        for sentence in corpus:
            word_counts.update(sentence)

        self.vocab_size = len(word_counts)
        self.word_to_idx = {word: idx for idx, (word, _)
                            in enumerate(word_counts.most_common())}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        # Initialize vectors with small random values
        self.in_vectors = [
            [random.gauss(0, 0.01) for _ in range(self.embedding_dim)]
            for _ in range(self.vocab_size)
        ]
        self.out_vectors = [
            [random.gauss(0, 0.01) for _ in range(self.embedding_dim)]
            for _ in range(self.vocab_size)
        ]

    def _negative_samples(self, target_idx, num_samples=5):
        """Sample negative context words (not the target word).

        Args:
            target_idx: index of the target word to exclude
            num_samples: number of negative samples to draw

        Returns:
            list of word indices for negative sampling
        """
        samples = []
        while len(samples) < num_samples:
            idx = random.randrange(0, self.vocab_size)
            if idx != target_idx:
                samples.append(idx)
        return samples

    def train(self, corpus, epochs=5, window_size=2, negative_samples=5):
        """Train word vectors using skip-gram with negative sampling.

        Args:
            corpus: list of tokenized sentences (list of list of str)
            epochs: number of full passes over the corpus
            window_size: context window radius (2*target words per position)
            negative_samples: number of negative samples per positive pair
        """
        self._build_vocab(corpus)

        for epoch in range(epochs):
            total_loss = 0.0
            pairs_processed = 0

            for sentence in corpus:
                sent_indices = [self.word_to_idx[w] for w in sentence
                                if w in self.word_to_idx]
                n = len(sent_indices)

                for i, target_idx in enumerate(sent_indices):
                    # Define context window
                    start = max(0, i - window_size)
                    end = min(n, i + window_size + 1)

                    for j in range(start, end):
                        if j == i:
                            continue
                        context_idx = sent_indices[j]

                        # Positive sample loss + gradients
                        loss, d_in, d_out = self._train_pair(
                            target_idx, context_idx, 1.0, negative_samples
                        )
                        total_loss += loss
                        pairs_processed += 1

                        # Update input vector
                        for d in range(self.embedding_dim):
                            self.in_vectors[target_idx][d] -= (
                                self.learning_rate * d_in[d]
                            )

            avg_loss = total_loss / max(pairs_processed, 1)
            # Print progress
            if (epoch + 1) % 1 == 0:
                pass  # demo() will report

    def _train_pair(self, target_idx, context_idx, label, negative_samples):
        """Train on one (target, context) pair with negative sampling.

        Args:
            target_idx: index of the target word
            context_idx: index of the positive context word
            label: 1.0 for positive, 0.0 for negative (for loss computation)
            negative_samples: number of negative samples to draw

        Returns:
            (loss, grad_input, grad_output) tuple
        """
        # Embeddings
        v_target = self.in_vectors[target_idx]

        # Positive context
        v_context = self.out_vectors[context_idx]
        score = sum(v_target[d] * v_context[d] for d in range(self.embedding_dim))
        prob = self._sigmoid(score)
        loss = -label * math.log(max(prob, 1e-15)) - (1 - label) * math.log(max(1 - prob, 1e-15))

        grad = prob - label  # = sigmoid(score) - label

        # Gradient for input vector (from positive sample)
        grad_in = [grad * v_context[d] for d in range(self.embedding_dim)]
        grad_out_pos = [grad * v_target[d] for d in range(self.embedding_dim)]

        # Update output vector for positive context
        for d in range(self.embedding_dim):
            self.out_vectors[context_idx][d] -= self.learning_rate * grad_out_pos[d]

        # Negative samples
        neg_indices = self._negative_samples(target_idx, negative_samples)
        for neg_idx in neg_indices:
            v_neg = self.out_vectors[neg_idx]
            score_neg = sum(v_target[d] * v_neg[d] for d in range(self.embedding_dim))
            prob_neg = self._sigmoid(score_neg)
            loss_neg = -0 * math.log(max(prob_neg, 1e-15)) - (1 - 0) * math.log(max(1 - prob_neg, 1e-15))
            loss += loss_neg

            grad_neg = prob_neg - 0  # label = 0 for negative

            # Gradient accumulation for input vector
            for d in range(self.embedding_dim):
                grad_in[d] += grad_neg * v_neg[d]

            # Update output vector for negative context
            for d in range(self.embedding_dim):
                self.out_vectors[neg_idx][d] -= self.learning_rate * grad_neg * v_target[d]

        return loss, grad_in, [0.0] * self.embedding_dim

    def get_vector(self, word):
        """Return the learned embedding vector for a word.

        Args:
            word: the word to look up

        Returns:
            list of floats (embedding vector), or None if not found
        """
        if word not in self.word_to_idx:
            return None
        idx = self.word_to_idx[word]
        return self.in_vectors[idx][:]

    def most_similar(self, word, top_n=5):
        """Find the most similar words by cosine similarity.

        Args:
            word: the query word
            top_n: number of nearest neighbors to return

        Returns:
            list of (word, similarity) tuples
        """
        if word not in self.word_to_idx:
            return []

        target_vec = self.get_vector(word)
        target_norm = math.sqrt(sum(x**2 for x in target_vec)) + 1e-10

        similarities = []
        for other_word, idx in self.word_to_idx.items():
            if other_word == word:
                continue
            other_vec = self.in_vectors[idx]
            dot = sum(target_vec[d] * other_vec[d] for d in range(self.embedding_dim))
            other_norm = math.sqrt(sum(x**2 for x in other_vec)) + 1e-10
            sim = dot / (target_norm * other_norm)
            similarities.append((other_word, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]


# ═══════════════════════════════════════════════
# RNN CELL (shared by encoder and decoder)
# ═══════════════════════════════════════════════

class RNNCeil:
    """A single recurrent cell (Elman RNN).

    h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b_h)

    Args:
        input_dim: size of input vectors
        hidden_dim: size of hidden state vectors
    """

    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input -> hidden weights
        self.W_ih = [[random.gauss(0, 0.1) for _ in range(input_dim)]
                     for _ in range(hidden_dim)]
        # Hidden -> hidden weights
        self.W_hh = [[random.gauss(0, 0.1) for _ in range(hidden_dim)]
                     for _ in range(hidden_dim)]
        # Bias
        self.b_h = [0.0 for _ in range(hidden_dim)]

    def forward(self, x, h_prev=None):
        """Run one step of the RNN.

        Args:
            x: input vector (list of floats, length input_dim)
            h_prev: previous hidden state (list of floats, length hidden_dim).
                     If None, initializes to zeros.

        Returns:
            h: new hidden state (list of floats, length hidden_dim)
        """
        if h_prev is None:
            h_prev = [0.0 for _ in range(self.hidden_dim)]

        # h = tanh(W_ih @ x + W_hh @ h_prev + b)
        h = []
        for i in range(self.hidden_dim):
            # W_ih[i] dot x
            val = sum(self.W_ih[i][j] * x[j] for j in range(self.input_dim))
            # W_hh[i] dot h_prev
            val += sum(self.W_hh[i][j] * h_prev[j] for j in range(self.hidden_dim))
            val += self.b_h[i]
            h.append(math.tanh(val))

        return h


# ═══════════════════════════════════════════════
# SEQUENCE-TO-SEQUENCE (TOY ENCODER-DECODER)
# ═══════════════════════════════════════════════

class Seq2Seq:
    """Toy sequence-to-sequence model with fixed context vector.

    Encoder processes an input sequence into a single context vector
    (the final encoder hidden state). Decoder generates the output
    sequence conditioned on this context vector.

    Uses simple RNN cells for both encoder and decoder.

    Attributes:
        encoder: RNNCeil for encoding the input
        decoder: RNNCeil for decoding the output
        enc_output_dim: hidden size of the encoder
        dec_input_dim: input size to the decoder (embeddings)
        dec_hidden_dim: hidden size of the decoder (= enc_output_dim)
        vocab_size: number of output tokens
        output_proj: weight matrix (vocab_size × hidden_dim) for generating logits
    """

    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=32):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layer for both encoder and decoder input tokens
        self.embeddings = [
            [random.gauss(0, 0.1) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]

        # Encoder and decoder RNN cells
        self.encoder = RNNCeil(embedding_dim, hidden_dim)
        self.decoder = RNNCeil(embedding_dim, hidden_dim)

        # Output projection: hidden_dim -> vocab_size
        self.output_proj = [
            [random.gauss(0, 0.1) for _ in range(hidden_dim)]
            for _ in range(vocab_size)
        ]
        self.output_bias = [0.0 for _ in range(vocab_size)]

    def _embed(self, token_id):
        """Get embedding vector for a token id."""
        if token_id < 0 or token_id >= self.vocab_size:
            return [0.0 for _ in range(self.embedding_dim)]
        return self.embeddings[token_id][:]

    def _softmax(self, logits):
        max_l = max(logits)
        exps = [math.exp(z - max_l) for z in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def encode(self, input_ids):
        """Encode an input sequence into a context vector.

        Args:
            input_ids: list of token IDs (integers)

        Returns:
            final hidden state (context vector) as list of floats
        """
        h = None
        for token_id in input_ids:
            x = self._embed(token_id)
            h = self.encoder.forward(x, h)
        return h

    def decode_step(self, token_id, hidden_state):
        """Decode one token given the current decoder hidden state.

        Args:
            token_id: previous output token ID
            hidden_state: current decoder hidden state

        Returns:
            (logits, new_hidden_state): output logits and updated hidden state
        """
        x = self._embed(token_id)
        h_new = self.decoder.forward(x, hidden_state)

        # Project hidden state to vocabulary logits
        logits = []
        for i in range(self.vocab_size):
            val = sum(self.output_proj[i][j] * h_new[j] for j in range(self.hidden_dim))
            val += self.output_bias[i]
            logits.append(val)

        return logits, h_new

    def forward(self, input_ids, target_ids=None, max_len=20):
        """Full forward pass: encode then decode.

        Args:
            input_ids: list of input token IDs
            target_ids: optional ground-truth output IDs for teacher forcing
            max_len: maximum decoding length when target_ids is None

        Returns:
            List of (token_id, logits) for each decoding step.
            During inference (no target_ids), token_id is the argmax.
        """
        # Encode
        context = self.encode(input_ids)
        h = context  # decoder starts from encoder's final hidden state

        # Start with <SOS> token (index 0)
        prev_token = 0
        outputs = []

        if target_ids is not None:
            # Teacher forcing
            for target_id in target_ids:
                logits, h = self.decode_step(prev_token, h)
                outputs.append((target_id, logits))
                prev_token = target_id
        else:
            # Autoregressive generation
            for _ in range(max_len):
                logits, h = self.decode_step(prev_token, h)
                probs = self._softmax(logits)
                next_token = probs.index(max(probs))  # greedy decoding
                outputs.append((next_token, logits))
                prev_token = next_token
                if next_token == 1:  # <EOS> token
                    break

        return outputs


# ═══════════════════════════════════════════════
# ATTENTION MECHANISMS
# ═══════════════════════════════════════════════

class BahdanauAttention:
    """Bahdanau (additive) attention mechanism.

    Computes attention scores as:
        score(h_t, h_s) = v_a^T @ tanh(W_1 @ h_t + W_2 @ h_s)

    where h_t is the current decoder hidden state and
    h_s are the encoder hidden states.

    Attributes:
        hidden_dim: dimensionality of hidden states
        W1: weight matrix for decoder hidden state (hidden_dim × hidden_dim)
        W2: weight matrix for encoder hidden states (hidden_dim × hidden_dim)
        v_a: attention vector (length hidden_dim)
    """

    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

        # W1 transforms decoder hidden state
        self.W1 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)]
                   for _ in range(hidden_dim)]
        # W2 transforms encoder hidden states
        self.W2 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)]
                   for _ in range(hidden_dim)]
        # Attention vector v_a
        self.va = [random.gauss(0, 0.1) for _ in range(hidden_dim)]

    def score(self, decoder_hidden, encoder_hidden):
        """Compute a single attention score.

        Args:
            decoder_hidden: current decoder hidden state (list, length hidden_dim)
            encoder_hidden: single encoder hidden state (list, length hidden_dim)

        Returns:
            scalar attention score
        """
        # W1 @ decoder_hidden
        w1_h = [sum(self.W1[i][j] * decoder_hidden[j]
                    for j in range(self.hidden_dim))
                for i in range(self.hidden_dim)]
        # W2 @ encoder_hidden
        w2_s = [sum(self.W2[i][j] * encoder_hidden[j]
                    for j in range(self.hidden_dim))
                for i in range(self.hidden_dim)]
        # v_a^T @ tanh(W1 @ h_t + W2 @ h_s)
        combined = [math.tanh(w1_h[i] + w2_s[i]) for i in range(self.hidden_dim)]
        score_val = sum(self.va[i] * combined[i] for i in range(self.hidden_dim))
        return score_val

    def forward(self, decoder_hidden, encoder_states):
        """Compute attention weights and context vector.

        Args:
            decoder_hidden: current decoder hidden state (list, length hidden_dim)
            encoder_states: list of encoder hidden states, each length hidden_dim

        Returns:
            (weights, context_vector):
                weights: list of attention weights (sum to 1)
                context_vector: weighted sum of encoder states (list, length hidden_dim)
        """
        # Compute scores for all encoder states
        scores = [self.score(decoder_hidden, h) for h in encoder_states]

        # Softmax over scores
        max_s = max(scores) if scores else 0.0
        exp_scores = [math.exp(s - max_s) for s in scores]
        total = sum(exp_scores)
        weights = [e / total for e in exp_scores] if total > 0 else (
            [1.0 / len(scores)] * len(scores)
        )

        # Weighted sum of encoder states
        context = [0.0 for _ in range(self.hidden_dim)]
        for i, h in enumerate(encoder_states):
            for d in range(self.hidden_dim):
                context[d] += weights[i] * h[d]

        return weights, context


class LuongAttention:
    """Luong (dot-product / multiplicative) attention mechanism.

    Supports three scoring methods:
        - 'dot':       score(h_t, h_s) = h_t^T @ h_s
        - 'general':   score(h_t, h_s) = h_t^T @ W @ h_s
        - 'concat':    score(h_t, h_s) = v_a^T @ tanh(W @ [h_t; h_s])

    Attributes:
        method: scoring method ('dot', 'general', or 'concat')
        hidden_dim: dimensionality of hidden states
        W: weight matrix for 'general' method (hidden_dim × hidden_dim)
        va: attention vector for 'concat' method
    """

    def __init__(self, hidden_dim, method='dot'):
        self.method = method
        self.hidden_dim = hidden_dim

        # General method: W matrix
        self.W = [[random.gauss(0, 0.1) for _ in range(hidden_dim)]
                  for _ in range(hidden_dim)]

        # Concat method: v_a vector
        self.va = [random.gauss(0, 0.1) for _ in range(hidden_dim)]

    def score(self, decoder_hidden, encoder_hidden):
        """Compute a single attention score.

        Args:
            decoder_hidden: current decoder hidden state (list, length hidden_dim)
            encoder_hidden: single encoder hidden state (list, length hidden_dim)

        Returns:
            scalar attention score
        """
        if self.method == 'dot':
            # h_t^T @ h_s
            return sum(decoder_hidden[d] * encoder_hidden[d]
                       for d in range(self.hidden_dim))

        elif self.method == 'general':
            # h_t^T @ W @ h_s
            w_hs = [sum(self.W[i][j] * encoder_hidden[j]
                        for j in range(self.hidden_dim))
                    for i in range(self.hidden_dim)]
            return sum(decoder_hidden[d] * w_hs[d] for d in range(self.hidden_dim))

        elif self.method == 'concat':
            # v_a^T @ tanh(W @ [h_t; h_s])
            # Simplified: v_a^T @ tanh(W_1 @ h_t + W_2 @ h_s)
            # We approximate as v_a^T @ tanh(W_concat @ h_t + W_concat @ h_s)
            combined = [math.tanh(decoder_hidden[d] + encoder_hidden[d])
                        for d in range(self.hidden_dim)]
            return sum(self.va[d] * combined[d] for d in range(self.hidden_dim))

        else:
            raise ValueError(f"Unknown attention method: {self.method}")

    def forward(self, decoder_hidden, encoder_states):
        """Compute attention weights and context vector.

        Args:
            decoder_hidden: current decoder hidden state (list, length hidden_dim)
            encoder_states: list of encoder hidden states, each length hidden_dim

        Returns:
            (weights, context_vector):
                weights: list of attention weights (sum to 1)
                context_vector: weighted sum of encoder states (list, length hidden_dim)
        """
        # Compute scores
        scores = [self.score(decoder_hidden, h) for h in encoder_states]

        # Softmax
        max_s = max(scores) if scores else 0.0
        exp_scores = [math.exp(s - max_s) for s in scores]
        total = sum(exp_scores)
        weights = [e / total for e in exp_scores] if total > 0 else (
            [1.0 / len(scores)] * len(scores)
        )

        # Weighted context vector
        context = [0.0 for _ in range(self.hidden_dim)]
        for i, h in enumerate(encoder_states):
            for d in range(self.hidden_dim):
                context[d] += weights[i] * h[d]

        return weights, context


# ═══════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════

def demo():
    """Run all NLP components and demonstrate functionality."""
    print("=" * 60)
    print("ANGGIRA NLP MODULE — DEMO")
    print("=" * 60)

    # ── BPE TOKENIZER ──
    print("\n" + "─" * 60)
    print("1. BPE TOKENIZER")
    print("─" * 60)

    corpus = [
        "low low low low low low low low low low low",
        "lowest lowest lowest lowest lowest lowest",
        "newer newer newer newer newer newer",
        "wider wider wider wider wider",
        "new new new new new new new new new",
        "newest newest newest newest newest newest",
    ]

    bpe = BPETokenizer()
    bpe.train(corpus, num_merges=20)

    print(f"Learned {len(bpe.merges)} merge rules:")
    for i, (l, r) in enumerate(bpe.merges[:10]):
        print(f"  Merge {i + 1}: '{l}' + '{r}' -> '{l}{r}'")
    if len(bpe.merges) > 10:
        print(f"  ... and {len(bpe.merges) - 10} more")

    test_words = ["low", "lower", "newest", "newer", "wider", "lowest"]
    print("\nEncoding test words:")
    for w in test_words:
        encoded = bpe.encode(w)
        decoded = bpe.decode(encoded)
        print(f"  '{w}' -> {encoded} -> '{decoded}'")

    # ── WORD2VEC ──
    print("\n" + "─" * 60)
    print("2. WORD2VEC (SKIP-GRAM)")
    print("─" * 60)

    # Tiny corpus
    sentences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "chased", "the", "cat"],
        ["a", "mouse", "ran", "under", "the", "table"],
        ["the", "cat", "chased", "the", "mouse"],
        ["the", "dog", "sat", "on", "the", "floor"],
        ["a", "bird", "sat", "on", "the", "window"],
        ["the", "cat", "sat", "under", "the", "table"],
        ["the", "bird", "chased", "the", "cat"],
    ]

    w2v = Word2Vec(embedding_dim=20, learning_rate=0.05)
    w2v.train(sentences, epochs=30, window_size=2, negative_samples=3)

    print(f"Vocabulary size: {w2v.vocab_size}")
    print(f"Embedding dimension: {w2v.embedding_dim}")

    # Show most similar words
    query_word = "cat"
    similar = w2v.most_similar(query_word, top_n=4)
    print(f"\nWords most similar to '{query_word}':")
    for word, sim in similar:
        print(f"  {word}: {sim:.4f}")

    # Show vector for a word
    vec = w2v.get_vector("dog")
    print(f"\nVector for 'dog' (first 6 dims): "
          f"{[round(v, 3) for v in vec[:6]]}...")

    # ── SEQUENCE-TO-SEQUENCE ──
    print("\n" + "─" * 60)
    print("3. SEQUENCE-TO-SEQUENCE MODEL")
    print("─" * 60)

    seq2seq = Seq2Seq(vocab_size=10, embedding_dim=16, hidden_dim=16)
    input_ids = [2, 3, 4, 5]  # dummy sequence
    target_ids = [6, 7, 8, 1]  # includes <EOS> (1)

    # Forward with teacher forcing
    outputs_tf = seq2seq.forward(input_ids, target_ids=target_ids)
    print("Teacher-forced forward pass:")
    for step, (tok, logits) in enumerate(outputs_tf):
        probs = seq2seq._softmax(logits)
        predicted = probs.index(max(probs))
        print(f"  Step {step}: target={tok}, predicted={predicted}")

    # Inference (autoregressive)
    outputs_infer = seq2seq.forward(input_ids, max_len=8)
    print("\nAutoregressive inference:")
    generated = [tok for tok, _ in outputs_infer]
    print(f"  Generated token IDs: {generated}")

    # ── ATTENTION ──
    print("\n" + "─" * 60)
    print("4. ATTENTION MECHANISMS")
    print("─" * 60)

    hidden_dim = 8
    n_encoder_states = 5

    # Generate random encoder states and decoder hidden
    encoder_states = [
        [random.gauss(0, 1) for _ in range(hidden_dim)]
        for _ in range(n_encoder_states)
    ]
    decoder_hidden = [random.gauss(0, 1) for _ in range(hidden_dim)]

    # Bahdanau attention
    bahdanau = BahdanauAttention(hidden_dim)
    weights_bd, context_bd = bahdanau.forward(decoder_hidden, encoder_states)
    print("Bahdanau (additive) attention:")
    print(f"  Attention weights: {[round(w, 3) for w in weights_bd]}")
    print(f"  Sum of weights: {sum(weights_bd):.3f}")
    print(f"  Context vector (first 4 dims): "
          f"{[round(c, 3) for c in context_bd[:4]]}")

    # Luong attention (dot product)
    luong_dot = LuongAttention(hidden_dim, method='dot')
    weights_ld, context_ld = luong_dot.forward(decoder_hidden, encoder_states)
    print("\nLuong (dot-product) attention:")
    print(f"  Attention weights: {[round(w, 3) for w in weights_ld]}")
    print(f"  Sum of weights: {sum(weights_ld):.3f}")
    print(f"  Context vector (first 4 dims): "
          f"{[round(c, 3) for c in context_ld[:4]]}")

    # Luong attention (general)
    luong_gen = LuongAttention(hidden_dim, method='general')
    weights_lg, context_lg = luong_gen.forward(decoder_hidden, encoder_states)
    print("\nLuong (general) attention:")
    print(f"  Attention weights: {[round(w, 3) for w in weights_lg]}")

    # Luong attention (concat)
    luong_con = LuongAttention(hidden_dim, method='concat')
    weights_lc, context_lc = luong_con.forward(decoder_hidden, encoder_states)
    print("\nLuong (concat) attention:")
    print(f"  Attention weights: {[round(w, 3) for w in weights_lc]}")

    print("\n" + "=" * 60)
    print("NLP DEMO COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    demo()
