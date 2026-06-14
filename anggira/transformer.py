"""
Anggira Transformer — LLM Architecture from Scratch

Phase 10: LLMs from Scratch
- Multi-Head Self-Attention, Transformer Blocks
- GPT-style Causal Language Model
- Training loop, text generation
"""

import math
import random


# ═══════════════════════════════════════════════
# ACTIVATIONS
# ═══════════════════════════════════════════════

def relu(x):
    return max(0.0, x)


def gelu(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


def softmax(x):
    max_val = max(x)
    exps = [math.exp(v - max_val) for v in x]
    total = sum(exps)
    return [e / total for e in exps]


# ═══════════════════════════════════════════════
# TINY LINEAR ALGEBRA (numpy-free)
# ═══════════════════════════════════════════════

def matmul(A, B):
    """Matrix multiply A (m×k) @ B (k×n)."""
    m, k = len(A), len(A[0])
    n = len(B[0])
    return [[sum(A[i][p] * B[p][j] for p in range(k))
             for j in range(n)] for i in range(m)]


def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


def mat_vec_mul(M, v):
    return [sum(row[j] * v[j] for j in range(len(v))) for row in M]


def vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]


def vec_mul(a, b):
    return [ai * bi for ai, bi in zip(a, b)]


def vec_scale(v, s):
    return [vi * s for vi in v]


def vec_sum(v):
    return sum(v)


# ═══════════════════════════════════════════════
# EMBEDDING / LAYERNORM
# ═══════════════════════════════════════════════

class Embedding:
    """Token + positional embeddings."""

    def __init__(self, vocab_size, dim, max_seq_len):
        self.dim = dim
        self.token_embed = [[random.gauss(0, 0.02) for _ in range(dim)]
                            for _ in range(vocab_size)]
        self.pos_embed = [[random.gauss(0, 0.02) for _ in range(dim)]
                          for _ in range(max_seq_len)]

    def forward(self, token_ids):
        """token_ids: list of ints (shape [seq_len]). Returns seq_len × dim."""
        seq_len = min(len(token_ids), len(self.pos_embed))
        token_ids = token_ids[:seq_len]
        result = []
        for i, tok_id in enumerate(token_ids):
            emb = self.token_embed[tok_id][:]
            for d in range(self.dim):
                emb[d] += self.pos_embed[i][d]
            result.append(emb)
        return result


class LayerNorm:
    """Layer normalization."""

    def __init__(self, dim, eps=1e-5):
        self.gamma = [1.0] * dim
        self.beta = [0.0] * dim
        self.eps = eps

    def forward(self, x):
        """x: list of vectors (seq_len × dim). Returns same shape."""
        result = []
        for vec in x:
            mean = sum(vec) / len(vec)
            var = sum((v - mean) ** 2 for v in vec) / len(vec)
            inv_std = 1.0 / math.sqrt(var + self.eps)
            normalized = [(v - mean) * inv_std for v in vec]
            result.append([g * n + b for g, n, b in zip(self.gamma, normalized, self.beta)])
        return result


# ═══════════════════════════════════════════════
# MULTI-HEAD ATTENTION
# ═══════════════════════════════════════════════

class MultiHeadAttention:
    """Scaled dot-product multi-head self-attention with causal masking."""

    def __init__(self, dim, num_heads):
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Initialize weight matrices
        self.W_q = [[random.gauss(0, 0.02) for _ in range(dim)] for _ in range(dim)]
        self.W_k = [[random.gauss(0, 0.02) for _ in range(dim)] for _ in range(dim)]
        self.W_v = [[random.gauss(0, 0.02) for _ in range(dim)] for _ in range(dim)]
        self.W_o = [[random.gauss(0, 0.02) for _ in range(dim)] for _ in range(dim)]

    def forward(self, x, causal=True):
        """
        x: seq_len × dim
        Returns: seq_len × dim
        """
        seq_len = len(x)
        d = self.dim
        nh = self.num_heads
        hd = self.head_dim

        # Project to Q, K, V
        Q = matmul(x, transpose(self.W_q))  # seq_len × dim
        K = matmul(x, transpose(self.W_k))
        V = matmul(x, transpose(self.W_v))

        # Reshape to multi-head: [seq_len, nh, hd]
        def split_heads(M):
            return [[M[i][h * hd:(h + 1) * hd] for h in range(nh)]
                    for i in range(seq_len)]

        Q_h = split_heads(Q)
        K_h = split_heads(K)
        V_h = split_heads(V)

        # Per-head attention with causal masking
        outputs = [[0.0] * d for _ in range(seq_len)]

        for h in range(nh):
            # Compute attention scores for this head
            for i in range(seq_len):
                for j in range(seq_len):
                    if causal and j > i:
                        continue  # causal mask
                    # Score = Q[i,h] · K[j,h] / sqrt(head_dim)
                    score = sum(Q_h[i][h][d] * K_h[j][h][d] for d in range(hd))
                    score /= math.sqrt(hd)
                    # We'll compute softmax per row

            # Apply softmax per row and weighted sum
            for i in range(seq_len):
                scores = []
                for j in range(seq_len):
                    if causal and j > i:
                        scores.append(-1e9)
                    else:
                        s = sum(Q_h[i][h][d] * K_h[j][h][d] for d in range(hd))
                        s /= math.sqrt(hd)
                        scores.append(s)

                attn_weights = softmax(scores)
                # Weighted sum of V values
                for d_out in range(hd):
                    val = sum(attn_weights[j] * V_h[j][h][d_out] for j in range(seq_len))
                    outputs[i][h * hd + d_out] += val

        # Final projection
        output = matmul(outputs, transpose(self.W_o))
        return output


# ═══════════════════════════════════════════════
# FEED FORWARD NETWORK
# ═══════════════════════════════════════════════

class FeedForward:
    """Two-layer MLP with GELU activation."""

    def __init__(self, dim, ff_dim):
        self.W1 = [[random.gauss(0, 0.02) for _ in range(dim)] for _ in range(ff_dim)]
        self.b1 = [0.0] * ff_dim
        self.W2 = [[random.gauss(0, 0.02) for _ in range(ff_dim)] for _ in range(dim)]
        self.b2 = [0.0] * dim

    def forward(self, x):
        """
        x: seq_len × dim
        Returns: seq_len × dim
        """
        result = []
        for vec in x:
            # W1 @ x + b1
            h = [sum(self.W1[j][k] * vec[k] for k in range(len(vec)))
                 + self.b1[j] for j in range(len(self.W1))]
            # GELU activation
            h = [gelu(v) for v in h]
            # W2 @ h + b2
            out = [sum(self.W2[k][j] * h[j] for j in range(len(h)))
                   + self.b2[k] for k in range(len(self.W2))]
            result.append(out)
        return result


# ═══════════════════════════════════════════════
# TRANSFORMER BLOCK
# ═══════════════════════════════════════════════

class TransformerBlock:
    """Pre-norm Transformer block with residual connections."""

    def __init__(self, dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.ln2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_dim)

    def forward(self, x, causal=True):
        # Self-attention with residual
        attn_out = self.attn.forward(self.ln1.forward(x), causal)
        x = [vec_add(x[i], attn_out[i]) for i in range(len(x))]

        # FFN with residual
        ffn_out = self.ffn.forward(self.ln2.forward(x))
        x = [vec_add(x[i], ffn_out[i]) for i in range(len(x))]
        return x


# ═══════════════════════════════════════════════
# CAUSAL LANGUAGE MODEL (GPT-style)
# ═══════════════════════════════════════════════

class AnggiraGPT:
    """Causal language model using decoder-only Transformer."""

    def __init__(self, vocab_size=256, dim=128, num_heads=4,
                 num_layers=4, max_seq_len=64, ff_dim=None):
        if ff_dim is None:
            ff_dim = dim * 4
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding = Embedding(vocab_size, dim, max_seq_len)
        self.blocks = [TransformerBlock(dim, num_heads, ff_dim)
                       for _ in range(num_layers)]
        self.ln_f = LayerNorm(dim)

    def forward(self, token_ids):
        """
        token_ids: list of ints [seq_len]
        Returns logits: seq_len × vocab_size
        """
        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, causal=True)
        x = self.ln_f.forward(x)

        # Project to vocab (weight tying with token embeddings)
        logits = []
        for vec in x:
            logits.append(mat_vec_mul(self.embedding.token_embed, vec))
        return logits

    def generate(self, prompt_ids, max_new_tokens=50, temperature=0.8, top_k=40):
        """Autoregressive generation."""
        tokens = list(prompt_ids)
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len
            context = tokens[-min(len(tokens), self.max_seq_len):]
            logits = self.forward(context)
            next_logits = logits[-1]

            # Apply temperature
            next_logits = [l / temperature for l in next_logits]

            # Softmax
            probs = softmax(next_logits)

            # Top-k filtering
            if top_k < len(probs):
                indices = list(range(len(probs)))
                sorted_idx = [i for i, _ in sorted(zip(indices, probs), key=lambda x: -x[1])]
                for i in sorted_idx[top_k:]:
                    probs[i] = 0.0
                total = sum(probs)
                probs = [p / total for p in probs]

            # Sample
            r = random.random()
            cumulative = 0.0
            next_token = 0
            for i, p in enumerate(probs):
                cumulative += p
                if r < cumulative:
                    next_token = i
                    break

            tokens.append(next_token)
        return tokens

    def count_parameters(self):
        total = 0
        # Embeddings
        total += len(self.embedding.token_embed) * len(self.embedding.token_embed[0])
        total += len(self.embedding.pos_embed) * len(self.embedding.pos_embed[0])
        # Blocks
        for block in self.blocks:
            total += len(block.attn.W_q) * len(block.attn.W_q[0])  # Q
            total += len(block.attn.W_k) * len(block.attn.W_k[0])  # K
            total += len(block.attn.W_v) * len(block.attn.W_v[0])  # V
            total += len(block.attn.W_o) * len(block.attn.W_o[0])  # O
            total += len(block.ffn.W1) * len(block.ffn.W1[0])  # W1
            total += len(block.ffn.b1)  # b1
            total += len(block.ffn.W2) * len(block.ffn.W2[0])  # W2
            total += len(block.ffn.b2)  # b2
            total += len(block.ln1.gamma) + len(block.ln1.beta)
            total += len(block.ln2.gamma) + len(block.ln2.beta)
        # Final LN
        total += len(self.ln_f.gamma) + len(self.ln_f.beta)
        return total


def cross_entropy_loss(logits, targets):
    """Cross-entropy loss for language modeling."""
    seq_len = len(logits)
    total_loss = 0.0
    for i in range(seq_len):
        # Softmax
        probs = softmax(logits[i])
        # Cross-entropy for target token
        total_loss += -math.log(max(probs[targets[i]], 1e-15))
    return total_loss / seq_len


# ═══════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════

def demo():
    random.seed(42)
    print("🤖 AnggiraGPT — Transformer from Scratch")
    print("=" * 60)

    # Tiny model for demo
    model = AnggiraGPT(vocab_size=256, dim=64, num_heads=4,
                       num_layers=3, max_seq_len=32, ff_dim=256)
    params = model.count_parameters()
    print(f"\n  Model parameters: {params:,}")
    print(f"  Architecture: 3 layers, 4 heads, 64-dim")

    # Training corpus
    corpus = (
        "The transformer architecture has revolutionized natural language processing. "
        "Attention mechanisms allow the model to focus on relevant parts of the input. "
        "Self-attention computes relationships between all pairs of positions. "
        "Multi-head attention splits the representation into multiple subspaces. "
        "Each attention head can learn different types of relationships. "
        "The feedforward network provides nonlinear transformations at each position. "
        "Residual connections enable gradient flow through deep networks."
    )

    tokens = list(corpus.encode("utf-8"))
    print(f"\n  Training tokens: {len(tokens):,}")
    print(f"  Corpus: \"{corpus[:60]}...\"")

    # Training loop
    seq_len = 16
    print(f"\n  Training ({len(tokens) // seq_len} batches)...")
    lr = 0.001

    for epoch in range(50):
        total_loss = 0.0
        batches = 0

        for start in range(0, len(tokens) - seq_len - 1, seq_len // 2):
            batch = tokens[start:start + seq_len + 1]
            if len(batch) < seq_len + 1:
                break

            input_ids = batch[:-1]
            target_ids = batch[1:]

            # Forward
            logits = model.forward(input_ids)
            loss = cross_entropy_loss(logits, target_ids)
            total_loss += loss
            batches += 1

            # Simple SGD update via finite differences (conceptual)
            # In a real training loop we'd use backprop
            # Here we just report the loss trajectory

        avg_loss = total_loss / batches if batches > 0 else 0
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:3d}: loss = {avg_loss:.4f}")

    # Generate text
    print(f"\n  Generating...")
    prompt = list("The transformer".encode("utf-8"))
    output = model.generate(prompt, max_new_tokens=30, temperature=0.8)
    generated = bytes(output).decode("utf-8", errors="replace")
    print(f"  Prompt: \"The transformer\"")
    print(f"  Output: \"{generated}\"")

    # Parameter breakdown for real models
    print(f"\n  GPT-2 Family Parameter Counts")
    print(f"  {'Model':<20} {'Params':>14}")
    print(f"  {'-'*34}")
    configs = [
        ("GPT-2 Small (124M)", 50257, 768, 12, 12, 1024),
        ("GPT-2 Medium (355M)", 50257, 1024, 16, 24, 1024),
        ("GPT-2 Large (774M)", 50257, 1280, 20, 36, 1024),
        ("GPT-2 XL (1.5B)", 50257, 1600, 25, 48, 1024),
    ]
    for name, vocab, dim, heads, layers, seq in configs:
        params = vocab * dim + seq * dim
        params += layers * (4 * dim * dim + 2 * dim * (dim * 4) + dim + dim * 4 + 4 * dim)
        params += 2 * dim
        print(f"  {name:<20} {params:>14,}")


if __name__ == "__main__":
    demo()
