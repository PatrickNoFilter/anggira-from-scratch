"""
AnggiraGPT — Fast Transformer using NumPy

Phase 10: LLMs from Scratch
Mini GPT-style language model.

Upgrades from Raschka's LLMs-from-scratch:
  - GELU activation (tanh approx) instead of ReLU
  - Dropout (0.1) after attention and FFN
  - Weight-tying gradient merged into embedding for single AdamW update
  - Training/eval mode flag
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Activation functions
# ═══════════════════════════════════════════════════════════════

def gelu(x):
    """GELU activation: tanh approximation (Raschka / GPT-2 style).

    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    a = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(a * (x + 0.044715 * x ** 3)))


def gelu_grad(x):
    """Gradient of the GELU tanh-approximation.

    Let  a = sqrt(2/pi),  b = 0.044715
    Let  f(x) = a * (x + b * x^3)
    Let  t = tanh(f(x))
    GELU(x)  = 0.5 * x * (1 + t)
    GELU'(x) = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * a * (1 + 3b * x^2)
    """
    a = np.sqrt(2.0 / np.pi)
    b = 0.044715
    f = a * (x + b * x ** 3)
    t = np.tanh(f)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * a * (1.0 + 3.0 * b * x ** 2)


# ═══════════════════════════════════════════════════════════════
# Dropout
# ═══════════════════════════════════════════════════════════════

class Dropout:
    """Inverted dropout: activations are scaled by 1/(1-rate) during training."""

    def __init__(self, rate=0.1):
        self.rate = rate
        self.mask = None

    def forward(self, x, training=True):
        if not training or self.rate <= 0.0:
            return x
        self.mask = np.random.binomial(1, 1.0 - self.rate, size=x.shape).astype(x.dtype)
        self.mask /= (1.0 - self.rate)  # inverted scaling
        return x * self.mask

    def backward(self, dout):
        if self.mask is None:
            return dout
        return dout * self.mask


# ═══════════════════════════════════════════════════════════════
# Layers
# ═══════════════════════════════════════════════════════════════

class Embedding:
    def __init__(self, vocab_size, dim, max_seq_len):
        self.token_embed = np.random.randn(vocab_size, dim) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, dim) * 0.02
        # Gradient buffers (populated by backward)
        self._grad_token_embed = np.zeros_like(self.token_embed)
        self._grad_pos_embed = np.zeros_like(self.pos_embed)

    def forward(self, token_ids):
        seq_len = token_ids.shape[-1]
        self._cache_ids = token_ids
        self._seq_len = seq_len
        return self.token_embed[token_ids] + self.pos_embed[:seq_len]

    def backward(self, dout):
        """Backprop: accumulate gradients into token_embed and pos_embed."""
        token_ids = self._cache_ids
        seq_len = self._seq_len
        B, T = token_ids.shape
        # d_token_embed: scatter-add gradients back to each token position
        flat_ids = token_ids.reshape(-1)
        flat_dout = dout.reshape(-1, dout.shape[-1])
        self._grad_token_embed[:] = 0.0  # zero in-place, preserve array ref
        np.add.at(self._grad_token_embed, flat_ids, flat_dout)
        # d_pos_embed
        self._grad_pos_embed[:] = 0.0
        self._grad_pos_embed[:seq_len] = dout.sum(axis=0)
        return None


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps
        self._grad_gamma = np.zeros_like(self.gamma)
        self._grad_beta = np.zeros_like(self.beta)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self._cache_x = x
        self._cache_mean = mean
        self._cache_inv_std = 1.0 / np.sqrt(var + self.eps)
        return self.gamma * (x - mean) * self._cache_inv_std + self.beta

    def backward(self, dout):
        """LayerNorm backward via the standard gradient path."""
        x = self._cache_x
        mean = self._cache_mean
        inv_std = self._cache_inv_std
        N = x.shape[-1]

        x_hat = (x - mean) * inv_std

        # Gradient for gamma and beta
        self._grad_gamma[:] = (dout * x_hat).sum(axis=tuple(range(dout.ndim - 1)))
        self._grad_beta[:] = dout.sum(axis=tuple(range(dout.ndim - 1)))

        # Gradient for input
        dx_hat = dout * self.gamma
        dvar = (dx_hat * (x - mean) * -0.5 * inv_std ** 3).sum(axis=-1, keepdims=True)
        dmean = (dx_hat * -inv_std).sum(axis=-1, keepdims=True) \
                + dvar * (-2.0 * (x - mean)).sum(axis=-1, keepdims=True) / N
        dx = dx_hat * inv_std + dvar * 2.0 * (x - mean) / N + dmean / N
        return dx


class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.02
        self.W_o = np.random.randn(embed_dim, embed_dim) * 0.02
        # Gradient buffers
        self._grad_W_q = np.zeros_like(self.W_q)
        self._grad_W_k = np.zeros_like(self.W_k)
        self._grad_W_v = np.zeros_like(self.W_v)
        self._grad_W_o = np.zeros_like(self.W_o)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        nh = self.num_heads
        hd = self.head_dim

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape to multi-head
        Qr = Q.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        Kr = K.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)
        Vr = V.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)

        scores = Qr @ Kr.transpose(0, 1, 3, 2) / np.sqrt(hd)
        if mask is not None:
            scores = scores + mask

        # Stable softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        scores_exp = np.exp(scores - scores_max)
        attn = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

        context = attn @ Vr
        context_t = context.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = context_t @ self.W_o

        # Cache for backward
        self._cache = (x, Q, K, V, Qr, Kr, Vr, scores, scores_max, attn, context, context_t, mask)
        return out

    def backward(self, dout):
        """Multi-Head Attention backward pass."""
        x, Q, K, V, Qr, Kr, Vr, scores, scores_max, attn, context, context_t, mask = self._cache
        B, T, D = x.shape
        nh = self.num_heads
        hd = self.head_dim

        # Gradient for W_o
        self._grad_W_o[:] = context_t.reshape(-1, D).T @ dout.reshape(-1, D)
        d_context_t = dout @ self.W_o.T

        # Reshape d_context_t to multi-head
        d_context = d_context_t.reshape(B, T, nh, hd).transpose(0, 2, 1, 3)

        # Gradient for V: context = attn @ V
        d_Vr = attn.transpose(0, 1, 3, 2) @ d_context

        # Gradient for attn
        d_attn = d_context @ Vr.transpose(0, 1, 3, 2)

        # Softmax backward
        dscores = attn * (d_attn - (attn * d_attn).sum(axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(hd)

        # Gradients for Q, K
        d_Qr = dscores @ Kr
        d_Kr = dscores.transpose(0, 1, 3, 2) @ Qr

        # Reshape gradients back to (B, T, D)
        d_Q = d_Qr.transpose(0, 2, 1, 3).reshape(B, T, D)
        d_K = d_Kr.transpose(0, 2, 1, 3).reshape(B, T, D)
        d_V = d_Vr.transpose(0, 2, 1, 3).reshape(B, T, D)

        # Gradients for W_q, W_k, W_v: Q = x @ W_q
        self._grad_W_q[:] = x.reshape(-1, D).T @ d_Q.reshape(-1, D)
        self._grad_W_k[:] = x.reshape(-1, D).T @ d_K.reshape(-1, D)
        self._grad_W_v[:] = x.reshape(-1, D).T @ d_V.reshape(-1, D)

        # Gradient for input x
        dx = d_Q @ self.W_q.T + d_K @ self.W_k.T + d_V @ self.W_v.T
        return dx


class FeedForward:
    def __init__(self, dim, ff_dim):
        self.W1 = np.random.randn(dim, ff_dim) * 0.02
        self.b1 = np.zeros(ff_dim)
        self.W2 = np.random.randn(ff_dim, dim) * 0.02
        self.b2 = np.zeros(dim)
        # Gradient buffers
        self._grad_W1 = np.zeros_like(self.W1)
        self._grad_b1 = np.zeros_like(self.b1)
        self._grad_W2 = np.zeros_like(self.W2)
        self._grad_b2 = np.zeros_like(self.b2)

    def forward(self, x):
        self._cache_x = x
        self._cache_pre = x @ self.W1 + self.b1
        self._cache_act = gelu(self._cache_pre)  # GELU instead of ReLU
        return self._cache_act @ self.W2 + self.b2

    def backward(self, dout):
        """FeedForward backward: GELU + two linear layers."""
        x = self._cache_x
        pre = self._cache_pre
        act = self._cache_act

        # Gradients for W2, b2
        self._grad_W2[:] = act.reshape(-1, act.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])
        self._grad_b2[:] = dout.sum(axis=tuple(range(dout.ndim - 1)))

        # Backprop through W2
        d_act = dout @ self.W2.T

        # GELU backward
        d_pre = d_act * gelu_grad(pre)

        # Gradients for W1, b1
        self._grad_W1[:] = x.reshape(-1, x.shape[-1]).T @ d_pre.reshape(-1, d_pre.shape[-1])
        self._grad_b1[:] = d_pre.sum(axis=tuple(range(d_pre.ndim - 1)))

        # Backprop through W1
        dx = d_pre @ self.W1.T
        return dx


class TransformerBlock:
    def __init__(self, dim, num_heads, ff_dim, dropout_rate=0.1):
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.dropout1 = Dropout(dropout_rate)
        self.ln2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_dim)
        self.dropout2 = Dropout(dropout_rate)

    def forward(self, x, mask=None, training=True):
        # Pre-LN + attention + dropout + residual
        ln1_out = self.ln1.forward(x)
        attn_out = self.attn.forward(ln1_out, mask)
        attn_out = self.dropout1.forward(attn_out, training)
        x2 = x + attn_out

        # Pre-LN + FFN + dropout + residual
        ln2_out = self.ln2.forward(x2)
        ff_out = self.ffn.forward(ln2_out)
        ff_out = self.dropout2.forward(ff_out, training)
        return x2 + ff_out

    def backward(self, dout):
        """Backprop through residual connections, attention, FFN, and dropouts."""
        # Last residual: output = x2 + ff_out
        d_ffn_out = dout
        d_x2_res = dout

        # Dropout2 backward
        d_ln2_out = self.dropout2.backward(self.ffn.backward(d_ffn_out))
        d_x2 = self.ln2.backward(d_ln2_out)
        d_x2 = d_x2 + d_x2_res

        # Attention residual: x2 = x1 + attn_out
        d_attn_out = d_x2
        d_x1_res = d_x2

        # Dropout1 backward + attention backward
        d_ln1_out = self.dropout1.backward(self.attn.backward(d_attn_out))
        d_x1 = self.ln1.backward(d_ln1_out)
        dx = d_x1 + d_x1_res
        return dx


class AnggiraGPT:
    """GPT-style causal language model with GELU, Dropout, and Pre-LN."""

    def __init__(self, vocab_size=50257, dim=768, num_heads=12,
                 num_layers=12, max_seq_len=1024, ff_dim=3072, dropout_rate=0.1):
        self.embedding = Embedding(vocab_size, dim, max_seq_len)
        self.blocks = [TransformerBlock(dim, num_heads, ff_dim, dropout_rate)
                       for _ in range(num_layers)]
        self.ln_f = LayerNorm(dim)
        self.training = True
        self._vocab_size = vocab_size
        self._dim = dim
        self._mask_cache = {}  # Cache causal masks by sequence length

    def _get_mask(self, T):
        """Return cached causal mask for sequence length T."""
        if T not in self._mask_cache:
            self._mask_cache[T] = np.triu(np.full((T, T), -1e9), k=1)
        return self._mask_cache[T]

    def forward(self, token_ids):
        T = token_ids.shape[-1]
        mask = self._get_mask(T)

        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x, mask, training=self.training)
        x = self.ln_f.forward(x)

        # Weight tying: project to vocab using token_embed
        logits = x @ self.embedding.token_embed.T
        return logits

    def backward(self, dlogits, token_ids):
        """Full backprop through all layers.

        Also merges the weight-tying gradient into embedding._grad_token_embed
        so a single optimizer can update everything.

        Args:
            dlogits: gradient of loss w.r.t. logits, shape (B, T, V)
            token_ids: input token IDs, shape (B, T)
        """
        x = self.ln_f._cache_x  # input to ln_f from the forward pass
        B, T, D = x.shape
        V = dlogits.shape[-1]

        # Gradient through weight-tying projection: logits = x @ token_embed.T
        # d(token_embed) from weight tying = x.T @ dlogits  (D, V)
        flat_x = x.reshape(-1, D)
        flat_dlogits = dlogits.reshape(-1, V)
        grad_embed_tying = flat_x.T @ flat_dlogits  # shape (D, V)

        # Gradient for x from weight tying
        d_x = dlogits @ self.embedding.token_embed  # (B, T, D)

        # Backward through final LayerNorm
        d_x = self.ln_f.backward(d_x)

        # Backward through blocks in reverse
        for block in reversed(self.blocks):
            d_x = block.backward(d_x)

        # Backward through embedding (fills _grad_token_embed, _grad_pos_embed)
        self.embedding.backward(d_x)

        # Merge weight-tying gradient into token_embed gradient
        # grad_embed_tying is (D, V) -> transpose to (V, D) to match _grad_token_embed
        self.embedding._grad_token_embed += grad_embed_tying.T

    def collect_params(self):
        """Return list of (param, grad) tuples for the AdamW optimizer.

        Order is deterministic: embedding, final LN, then blocks (attn, ff, ln).
        """
        params = []

        # ── Embedding ──
        params.append((self.embedding.token_embed, self.embedding._grad_token_embed))
        params.append((self.embedding.pos_embed, self.embedding._grad_pos_embed))

        # ── Final LayerNorm ──
        params.append((self.ln_f.gamma, self.ln_f._grad_gamma))
        params.append((self.ln_f.beta, self.ln_f._grad_beta))

        # ── Each block ──
        for b in self.blocks:
            # Attention
            params.append((b.attn.W_q, b.attn._grad_W_q))
            params.append((b.attn.W_k, b.attn._grad_W_k))
            params.append((b.attn.W_v, b.attn._grad_W_v))
            params.append((b.attn.W_o, b.attn._grad_W_o))
            # FFN
            params.append((b.ffn.W1, b.ffn._grad_W1))
            params.append((b.ffn.b1, b.ffn._grad_b1))
            params.append((b.ffn.W2, b.ffn._grad_W2))
            params.append((b.ffn.b2, b.ffn._grad_b2))
            # LayerNorms
            params.append((b.ln1.gamma, b.ln1._grad_gamma))
            params.append((b.ln1.beta, b.ln1._grad_beta))
            params.append((b.ln2.gamma, b.ln2._grad_gamma))
            params.append((b.ln2.beta, b.ln2._grad_beta))

        return params

    def train_step(self, inp, tgt, optimizer):
        """Single training step using the provided optimizer.

        Args:
            inp: input token IDs, shape (B, T)
            tgt: target token IDs, shape (B, T)
            optimizer: AdamW instance (created from model.collect_params())
        Returns:
            loss value (float)
        """
        logits = self.forward(inp)

        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        tgt_flat = tgt.reshape(-1)

        # Cross-entropy loss (stable)
        max_l = logits_flat.max(axis=-1, keepdims=True)
        log_softmax = logits_flat - max_l - np.log(
            np.exp(logits_flat - max_l).sum(axis=-1, keepdims=True))
        loss = -log_softmax[np.arange(len(tgt_flat)), tgt_flat].mean()

        # Gradient of CE loss w.r.t. logits
        dlogits = np.exp(logits_flat - max_l)
        dlogits = dlogits / dlogits.sum(axis=-1, keepdims=True)
        dlogits[np.arange(len(tgt_flat)), tgt_flat] -= 1.0
        dlogits = dlogits.reshape(B, T, V) / (B * T)

        # Full backprop (fills all _grad_* buffers)
        self.backward(dlogits, inp)

        # Optimizer step (applies AdamW to all params)
        optimizer.step()

        return float(loss)

    def train_on_data(self, tokens, seq_len=64, num_steps=1000, lr=3e-4,
                      lr_warmup=100, log_every=50, batch_size=4):
        """Train the model on a 1D array of token IDs.

        Args:
            tokens: 1D numpy array of token IDs
            seq_len: context window length
            num_steps: number of training steps
            lr: peak learning rate
            lr_warmup: steps of linear warmup
            log_every: print loss every N steps
            batch_size: number of sequences per step
        """
        from anggira.optimizer import AdamW

        print(f"  Training on {len(tokens)} tokens")
        print(f"  Params: {self.count_params():,}")
        print(f"  Steps: {num_steps}, seq_len: {seq_len}, lr: {lr}, batch: {batch_size}")

        optimizer = AdamW(self.collect_params(), lr=lr)

        losses = []
        for step in range(num_steps):
            # LR schedule: linear warmup then constant
            if step < lr_warmup:
                current_lr = lr * (step + 1) / lr_warmup
                optimizer.lr = current_lr
            else:
                optimizer.lr = lr

            # Sample batch_size random segments
            inputs = []
            targets = []
            for _ in range(batch_size):
                i = np.random.randint(0, max(1, len(tokens) - seq_len - 1))
                batch = tokens[i:i + seq_len + 1]
                inputs.append(batch[:-1])
                targets.append(batch[1:])

            inp = np.stack(inputs, axis=0)
            tgt = np.stack(targets, axis=0)

            loss = self.train_step(inp, tgt, optimizer)
            losses.append(loss)

            if step % log_every == 0 or step == num_steps - 1:
                print(f"  Step {step:5d} | loss = {loss:.4f} | lr = {optimizer.lr:.6f}")

        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Avg loss (last 100): {np.mean(losses[-100:]):.4f}")
        return losses

    def generate(self, prompt_tokens, max_new=100, temp=0.8, top_k=None):
        tokens = list(prompt_tokens)
        max_seq = self.embedding.pos_embed.shape[0]
        was_training = self.training
        self.training = False  # disable dropout during generation
        try:
            for _ in range(max_new):
                ctx = np.array(tokens[-max_seq:]).reshape(1, -1)
                logits = self.forward(ctx)
                next_logits = logits[0, -1, :] / temp

                # Top-k filtering
                if top_k is not None and top_k < len(next_logits):
                    indices = np.argpartition(-next_logits, top_k)[:top_k]
                    mask = np.zeros_like(next_logits)
                    mask[indices] = 1.0
                    next_logits = next_logits * mask - 1e9 * (1 - mask)

                probs = np.exp(next_logits - next_logits.max())
                probs = probs / probs.sum()
                next_token = np.random.choice(len(probs), p=probs)
                tokens.append(int(next_token))
        finally:
            self.training = was_training
        return tokens

    def count_params(self):
        total = 0
        total += self.embedding.token_embed.size
        total += self.embedding.pos_embed.size
        for b in self.blocks:
            total += b.attn.W_q.size + b.attn.W_k.size
            total += b.attn.W_v.size + b.attn.W_o.size
            total += b.ffn.W1.size + b.ffn.b1.size
            total += b.ffn.W2.size + b.ffn.b2.size
            total += b.ln1.gamma.size + b.ln1.beta.size
            total += b.ln2.gamma.size + b.ln2.beta.size
        total += self.ln_f.gamma.size + self.ln_f.beta.size
        return total

    def train(self, mode=True):
        """Set model to training/eval mode."""
        self.training = mode

    def eval(self):
        """Set model to evaluation mode (disables dropout)."""
        self.training = False
