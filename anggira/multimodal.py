"""
Anggira Multimodal — ViT, CLIP, LLaVA-style Projector, Multimodal RAG, Visual Prompting

Phase 12: Multimodal AI from Scratch
- ViT (Vision Transformer): image→patches→projection→positional encodings→transformer→CLS→classification
- CLIP: dual-encoder (image + text) with contrastive (InfoNCE) loss, zero-shot classification
- LLaVA-style: MLP projector mapping visual → language embedding space, simple transformer LM
- Multimodal RAG: embed text & image features into shared space, cross-modal retrieval
- Visual Prompting: in-context visual prompting via overlay markers

Pure Python — no numpy, only stdlib (math, random).
"""

import math
import random


# ═══════════════════════════════════════════════════════════
# HELPER: tiny linear algebra (mirrors transformer.py)
# ═══════════════════════════════════════════════════════════

def relu(x):
    return max(0.0, x)


def gelu(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


def softmax(x):
    max_val = max(x)
    exps = [math.exp(v - max_val) for v in x]
    total = sum(exps)
    return [e / total for e in exps]


def matmul(A, B):
    """A (m×k) @ B (k×n)."""
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


def vec_sub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]


def vec_mul(a, b):
    return [ai * bi for ai, bi in zip(a, b)]


def vec_scale(v, s):
    return [vi * s for vi in v]


def vec_dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    dot = vec_dot(a, b)
    norm_a = vec_norm(a)
    norm_b = vec_norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


# ═══════════════════════════════════════════════════════════
# HELPER: LayerNorm (from transformer.py)
# ═══════════════════════════════════════════════════════════

class LayerNorm:
    """Layer normalization."""

    def __init__(self, dim, eps=1e-5):
        self.gamma = [1.0] * dim
        self.beta = [0.0] * dim
        self.eps = eps

    def forward(self, x):
        """x: seq_len × dim. Returns same shape."""
        result = []
        for vec in x:
            mean = sum(vec) / len(vec)
            var = sum((v - mean) ** 2 for v in vec) / len(vec)
            inv_std = 1.0 / math.sqrt(var + self.eps)
            normalized = [(v - mean) * inv_std for v in vec]
            result.append([g * n + b for g, n, b in zip(self.gamma, normalized, self.beta)])
        return result


# ═══════════════════════════════════════════════════════════
# HELPER: Multi-Head Self-Attention (from transformer.py)
# ═══════════════════════════════════════════════════════════

class MultiHeadSelfAttention:
    """Scaled dot-product multi-head self-attention (no causal mask — bidirectional)."""

    def __init__(self, dim, num_heads):
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_q = [[random.gauss(0, 0.02) for _ in range(dim)] for _ in range(dim)]
        self.W_k = [[random.gauss(0, 0.02) for _ in range(dim)] for _ in range(dim)]
        self.W_v = [[random.gauss(0, 0.02) for _ in range(dim)] for _ in range(dim)]
        self.W_o = [[random.gauss(0, 0.02) for _ in range(dim)] for _ in range(dim)]

    def forward(self, x):
        """x: seq_len × dim. Returns seq_len × dim (bidirectional)."""
        seq_len = len(x)
        nh = self.num_heads
        hd = self.head_dim

        Q = matmul(x, transpose(self.W_q))
        K = matmul(x, transpose(self.W_k))
        V = matmul(x, transpose(self.W_v))

        # Split heads: seq_len × nh × hd
        Q_h = [[Q[i][h * hd:(h + 1) * hd] for h in range(nh)] for i in range(seq_len)]
        K_h = [[K[i][h * hd:(h + 1) * hd] for h in range(nh)] for i in range(seq_len)]
        V_h = [[V[i][h * hd:(h + 1) * hd] for h in range(nh)] for i in range(seq_len)]

        outputs = [[0.0] * self.dim for _ in range(seq_len)]

        for h in range(nh):
            # Attention scores: seq_len × seq_len
            for i in range(seq_len):
                # Compute scores (no causal mask — bidirectional)
                scores = []
                for j in range(seq_len):
                    s = vec_dot(Q_h[i][h], K_h[j][h]) / math.sqrt(hd)
                    scores.append(s)

                attn_weights = softmax(scores)

                # Weighted sum of V
                for d_out in range(hd):
                    val = sum(attn_weights[j] * V_h[j][h][d_out] for j in range(seq_len))
                    outputs[i][h * hd + d_out] += val

        output = matmul(outputs, transpose(self.W_o))
        return output


# ═══════════════════════════════════════════════════════════
# HELPER: FeedForward (from transformer.py)
# ═══════════════════════════════════════════════════════════

class FeedForward:
    """Two-layer MLP with GELU activation."""

    def __init__(self, dim, ff_dim):
        self.W1 = [[random.gauss(0, 0.02) for _ in range(dim)] for _ in range(ff_dim)]
        self.b1 = [0.0] * ff_dim
        self.W2 = [[random.gauss(0, 0.02) for _ in range(ff_dim)] for _ in range(dim)]
        self.b2 = [0.0] * dim

    def forward(self, x):
        """x: seq_len × dim. Returns seq_len × dim."""
        result = []
        for vec in x:
            h = [sum(self.W1[j][k] * vec[k] for k in range(len(vec)))
                 + self.b1[j] for j in range(len(self.W1))]
            h = [gelu(v) for v in h]
            out = [sum(self.W2[k][j] * h[j] for j in range(len(h)))
                   + self.b2[k] for k in range(len(self.W2))]
            result.append(out)
        return result


# ═══════════════════════════════════════════════════════════
# HELPER: TransformerBlock (bidirectional, no causal mask)
# ═══════════════════════════════════════════════════════════

class TransformerBlock:
    """Pre-norm Transformer block with residual connections (bidirectional)."""

    def __init__(self, dim, num_heads, ff_dim):
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.ln2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_dim)

    def forward(self, x):
        attn_out = self.attn.forward(self.ln1.forward(x))
        x = [vec_add(x[i], attn_out[i]) for i in range(len(x))]
        ffn_out = self.ffn.forward(self.ln2.forward(x))
        x = [vec_add(x[i], ffn_out[i]) for i in range(len(x))]
        return x


# ═══════════════════════════════════════════════════════════
# HELPER: PositionalEncoding (learned)
# ═══════════════════════════════════════════════════════════

class PositionalEncoding:
    """Learned positional encoding."""

    def __init__(self, max_seq_len, dim):
        self.pos_embed = [[random.gauss(0, 0.02) for _ in range(dim)]
                          for _ in range(max_seq_len)]

    def forward(self, x):
        """Add positional encodings to x. x: seq_len × dim."""
        seq_len = len(x)
        return [vec_add(x[i], self.pos_embed[i]) for i in range(seq_len)]


# ═══════════════════════════════════════════════════════════
# 1. VISION TRANSFORMER (ViT)
# ═══════════════════════════════════════════════════════════
#
# ViT splits an image into patches, linearly projects each patch,
# adds positional encodings, prepends a CLS token, passes through
# a Transformer encoder, then classifies using the CLS token output.

class ViT:
    """Vision Transformer — image classification from scratch.

    Processes images as sequences of patches through a transformer
    encoder. Uses a CLS token for classification.

    Args:
        patch_size: height/width of each patch (image is patch_size × patch_size).
        image_channels: number of input channels (default 3 for RGB).
        hidden_dim: transformer dimension.
        num_heads: number of attention heads.
        num_layers: number of transformer blocks.
        ff_dim: feed-forward hidden dimension (default 4 * hidden_dim).
        num_classes: number of output classes.
        max_patches: maximum sequence length (num patches + CLS).
    """

    def __init__(self, patch_size=4, image_channels=3, hidden_dim=64,
                 num_heads=4, num_layers=4, ff_dim=None, num_classes=10,
                 max_patches=None):
        if ff_dim is None:
            ff_dim = hidden_dim * 4
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.image_channels = image_channels

        # Linear projection: (patch_size * patch_size * channels) -> hidden_dim
        patch_vec_dim = patch_size * patch_size * image_channels
        self.patch_proj = [[random.gauss(0, 0.02) for _ in range(patch_vec_dim)]
                           for _ in range(hidden_dim)]
        self.patch_bias = [0.0] * hidden_dim

        # Maximum patches = image_size/patch_size squared
        self.max_patches = max_patches or (32 * 32) // (patch_size * patch_size) + 1

        # CLS token (learnable)
        self.cls_token = [random.gauss(0, 0.02) for _ in range(hidden_dim)]

        # Positional encoding
        self.pos_enc = PositionalEncoding(self.max_patches + 1, hidden_dim)

        # Transformer encoder
        self.blocks = [TransformerBlock(hidden_dim, num_heads, ff_dim)
                       for _ in range(num_layers)]
        self.ln_f = LayerNorm(hidden_dim)

        # Classification head
        self.class_head = [[random.gauss(0, 0.02) for _ in range(hidden_dim)]
                           for _ in range(num_classes)]
        self.class_bias = [0.0] * num_classes
        self.num_classes = num_classes

    def _image_to_patches(self, image):
        """Convert image tensor to patch vectors.

        Args:
            image: list of lists of lists — [channels][height][width]
                   where height == width == patch_size * sqrt(num_patches)

        Returns:
            list of patch vectors, each of length patch_size*patch_size*channels
        """
        channels = len(image)
        H = len(image[0])
        W = len(image[0][0])
        ps = self.patch_size
        num_patches_h = H // ps
        num_patches_w = W // ps
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch_vec = []
                for c in range(channels):
                    for pi in range(ps):
                        for pj in range(ps):
                            patch_vec.append(image[c][i * ps + pi][j * ps + pj])
                patches.append(patch_vec)
        return patches

    def _project_patches(self, patches):
        """Linear projection of patch vectors to hidden_dim."""
        result = []
        for patch_vec in patches:
            proj = [sum(self.patch_proj[k][d] * patch_vec[d]
                        for d in range(len(patch_vec)))
                    + self.patch_bias[k] for k in range(self.hidden_dim)]
            result.append(proj)
        return result

    def forward(self, image):
        """Forward pass through ViT.

        Args:
            image: list of lists of lists — [channels][height][width]

        Returns:
            logits: list of length num_classes
            cls_output: the final CLS token representation
            all_outputs: all patch + CLS outputs (cls_output, ...patch_outputs)
        """
        # 1. Patchify
        patches = self._image_to_patches(image)

        # 2. Linear projection
        patch_embs = self._project_patches(patches)

        # 3. Prepend CLS token
        seq = [self.cls_token] + patch_embs

        # 4. Add positional encodings
        seq = self.pos_enc.forward(seq)

        # 5. Transformer encoder
        for block in self.blocks:
            seq = block.forward(seq)

        # 6. Final layer norm
        seq = self.ln_f.forward(seq)

        # 7. CLS token for classification
        cls_output = seq[0]

        # 8. Classification head
        logits = [vec_dot(self.class_head[c], cls_output) + self.class_bias[c]
                  for c in range(self.num_classes)]

        return logits, cls_output, seq

    def classify(self, image):
        """Return predicted class index."""
        logits, _, _ = self.forward(image)
        return logits.index(max(logits))

    def count_parameters(self):
        total = 0
        # Patch projection
        total += len(self.patch_proj) * len(self.patch_proj[0])
        total += len(self.patch_bias)
        # CLS token
        total += len(self.cls_token)
        # Positional encoding
        total += len(self.pos_enc.pos_embed) * len(self.pos_enc.pos_embed[0])
        # Transformer blocks
        for block in self.blocks:
            total += len(block.attn.W_q) * len(block.attn.W_q[0])
            total += len(block.attn.W_k) * len(block.attn.W_k[0])
            total += len(block.attn.W_v) * len(block.attn.W_v[0])
            total += len(block.attn.W_o) * len(block.attn.W_o[0])
            total += len(block.ffn.W1) * len(block.ffn.W1[0])
            total += len(block.ffn.b1)
            total += len(block.ffn.W2) * len(block.ffn.W2[0])
            total += len(block.ffn.b2)
            total += len(block.ln1.gamma) + len(block.ln1.beta)
            total += len(block.ln2.gamma) + len(block.ln2.beta)
        # Final LN
        total += len(self.ln_f.gamma) + len(self.ln_f.beta)
        # Classification head
        total += len(self.class_head) * len(self.class_head[0])
        total += len(self.class_bias)
        return total


# ═══════════════════════════════════════════════════════════
# 2. CLIP — DUAL-ENCODER WITH CONTRASTIVE LOSS
# ═══════════════════════════════════════════════════════════
#
# CLIP uses two encoders (image + text) that map to a shared
# embedding space. Training uses InfoNCE (contrastive) loss:
#   - Cosine similarity between all image-text pairs in a batch
#   - Cross-entropy over similarities (image→text, text→image)

class ImageEncoder:
    """Image encoder for CLIP — simplified ViT-like encoder.

    Uses a lightweight transformer to produce a single image embedding.
    """

    def __init__(self, patch_size=4, image_channels=3, hidden_dim=64,
                 num_heads=4, num_layers=3, ff_dim=None):
        if ff_dim is None:
            ff_dim = hidden_dim * 4
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.image_channels = image_channels

        patch_vec_dim = patch_size * patch_size * image_channels
        self.patch_proj = [[random.gauss(0, 0.02) for _ in range(patch_vec_dim)]
                           for _ in range(hidden_dim)]
        self.patch_bias = [0.0] * hidden_dim

        self.cls_token = [random.gauss(0, 0.02) for _ in range(hidden_dim)]
        self.pos_enc = PositionalEncoding(256, hidden_dim)

        self.blocks = [TransformerBlock(hidden_dim, num_heads, ff_dim)
                       for _ in range(num_layers)]
        self.ln_f = LayerNorm(hidden_dim)

        # Projection head to shared embedding space
        self.proj = [[random.gauss(0, 0.02) for _ in range(hidden_dim)]
                     for _ in range(hidden_dim)]
        self.proj_bias = [0.0] * hidden_dim

    def encode(self, image):
        """Encode image into a single embedding vector.

        Args:
            image: [channels][height][width]

        Returns:
            embedding vector of length hidden_dim
        """
        # Patchify
        H, W = len(image[0]), len(image[0][0])
        ps = self.patch_size
        num_patches_h = H // ps
        num_patches_w = W // ps
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch_vec = []
                for c in range(self.image_channels):
                    for pi in range(ps):
                        for pj in range(ps):
                            patch_vec.append(image[c][i * ps + pi][j * ps + pj])
                patches.append(patch_vec)

        # Project patches
        embs = []
        for pv in patches:
            emb = [sum(self.patch_proj[k][d] * pv[d] for d in range(len(pv)))
                   + self.patch_bias[k] for k in range(self.hidden_dim)]
            embs.append(emb)

        # Prepend CLS
        seq = [self.cls_token] + embs
        seq = self.pos_enc.forward(seq)

        for block in self.blocks:
            seq = block.forward(seq)

        seq = self.ln_f.forward(seq)
        cls_out = seq[0]

        # Project to shared space
        emb = [sum(self.proj[k][d] * cls_out[d] for d in range(self.hidden_dim))
               + self.proj_bias[k] for k in range(self.hidden_dim)]

        # L2-normalize
        norm = vec_norm(emb)
        if norm > 1e-12:
            emb = [v / norm for v in emb]
        return emb


class TextEncoder:
    """Text encoder for CLIP — simplified transformer encoder.

    Encodes text sequences into a single embedding vector.
    """

    def __init__(self, vocab_size=256, hidden_dim=64, num_heads=4,
                 num_layers=3, max_seq_len=64, ff_dim=None):
        if ff_dim is None:
            ff_dim = hidden_dim * 4
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_embed = [[random.gauss(0, 0.02) for _ in range(hidden_dim)]
                            for _ in range(vocab_size)]
        self.pos_enc = PositionalEncoding(max_seq_len, hidden_dim)

        self.blocks = [TransformerBlock(hidden_dim, num_heads, ff_dim)
                       for _ in range(num_layers)]
        self.ln_f = LayerNorm(hidden_dim)

        # Projection head to shared embedding space
        self.proj = [[random.gauss(0, 0.02) for _ in range(hidden_dim)]
                     for _ in range(hidden_dim)]
        self.proj_bias = [0.0] * hidden_dim

    def encode(self, token_ids):
        """Encode token sequence into a single embedding vector.

        Args:
            token_ids: list of int token ids

        Returns:
            embedding vector of length hidden_dim (L2-normalized)
        """
        seq_len = min(len(token_ids), self.max_seq_len)
        token_ids = token_ids[:seq_len]

        # Token embeddings
        x = [self.token_embed[tid][:] for tid in token_ids]
        x = self.pos_enc.forward(x)

        for block in self.blocks:
            x = block.forward(x)

        x = self.ln_f.forward(x)

        # Mean pooling over all tokens
        emb = [0.0] * self.hidden_dim
        for vec in x:
            emb = vec_add(emb, vec)
        emb = vec_scale(emb, 1.0 / len(x))

        # Project to shared space
        emb = [sum(self.proj[k][d] * emb[d] for d in range(self.hidden_dim))
               + self.proj_bias[k] for k in range(self.hidden_dim)]

        # L2-normalize
        norm = vec_norm(emb)
        if norm > 1e-12:
            emb = [v / norm for v in emb]
        return emb


def contrastive_loss(image_embs, text_embs, temperature=0.07):
    """InfoNCE contrastive loss for CLIP.

    Args:
        image_embs: list of image embedding vectors (batch_size × d)
        text_embs: list of text embedding vectors (batch_size × d)
        temperature: scaling temperature for logits

    Returns:
        loss: scalar (average cross-entropy loss)
        accuracy_i2t: image→text retrieval accuracy
        accuracy_t2i: text→image retrieval accuracy
    """
    batch_size = len(image_embs)
    d = len(image_embs[0])

    # Compute similarity matrix: batch_size × batch_size
    logits = [[0.0] * batch_size for _ in range(batch_size)]
    for i in range(batch_size):
        for j in range(batch_size):
            sim = cosine_similarity(image_embs[i], text_embs[j])
            logits[i][j] = sim / temperature

    # Loss: cross-entropy over rows (image→text) and columns (text→image)
    total_loss = 0.0
    correct_i2t = 0
    correct_t2i = 0

    for i in range(batch_size):
        # Image→Text: correct pair is (i, i)
        probs_i = softmax(logits[i])
        total_loss += -math.log(max(probs_i[i], 1e-15))
        if probs_i.index(max(probs_i)) == i:
            correct_i2t += 1

        # Text→Image: correct pair is (i, i) — column-wise softmax
        col = [logits[j][i] for j in range(batch_size)]
        probs_t = softmax(col)
        total_loss += -math.log(max(probs_t[i], 1e-15))
        if probs_t.index(max(probs_t)) == i:
            correct_t2i += 1

    loss = total_loss / (2 * batch_size)
    acc_i2t = correct_i2t / batch_size
    acc_t2i = correct_t2i / batch_size

    return loss, acc_i2t, acc_t2i


class CLIP:
    """CLIP dual-encoder model.

    Combines an ImageEncoder and TextEncoder with a contrastive
    training objective. Supports zero-shot classification by
    comparing image embeddings to text prompt embeddings.
    """

    def __init__(self, patch_size=4, image_channels=3, image_hidden_dim=64,
                 text_hidden_dim=64, text_vocab_size=256, num_heads=4,
                 num_layers=3, max_seq_len=64, embed_dim=64):
        self.embed_dim = embed_dim

        self.image_encoder = ImageEncoder(
            patch_size=patch_size, image_channels=image_channels,
            hidden_dim=image_hidden_dim, num_heads=num_heads,
            num_layers=num_layers
        )
        self.text_encoder = TextEncoder(
            vocab_size=text_vocab_size, hidden_dim=text_hidden_dim,
            num_heads=num_heads, num_layers=num_layers,
            max_seq_len=max_seq_len
        )

    def encode_image(self, image):
        """Encode image to shared embedding space."""
        return self.image_encoder.encode(image)

    def encode_text(self, token_ids):
        """Encode text to shared embedding space."""
        return self.text_encoder.encode(token_ids)

    def forward(self, images, text_batch):
        """Forward pass: encode a batch of images and texts.

        Args:
            images: list of image tensors (batch_size × [C][H][W])
            text_batch: list of token id lists (batch_size × [seq_len])

        Returns:
            image_embs, text_embs, loss, acc_i2t, acc_t2i
        """
        image_embs = [self.encode_image(img) for img in images]
        text_embs = [self.encode_text(tokens) for tokens in text_batch]
        loss, acc_i2t, acc_t2i = contrastive_loss(image_embs, text_embs)
        return image_embs, text_embs, loss, acc_i2t, acc_t2i

    def zero_shot_classify(self, image, class_prompts):
        """Zero-shot classification by text-prompt matching.

        Args:
            image: image tensor [C][H][W]
            class_prompts: list of text prompts (strings), one per class

        Returns:
            predicted_class_index, scores (list of similarities)
        """
        img_emb = self.encode_image(image)
        scores = []
        for prompt_ids in class_prompts:
            txt_emb = self.encode_text(prompt_ids)
            scores.append(cosine_similarity(img_emb, txt_emb))
        return scores.index(max(scores)), scores


# ═══════════════════════════════════════════════════════════
# 3. LLAVA-STYLE PROJECTOR
# ═══════════════════════════════════════════════════════════
#
# LLaVA uses a simple MLP to project visual features into the
# language model's embedding space. The projector is a 2-layer MLP
# (or sometimes a single linear layer). We then use a small
# transformer (from our existing architecture) as the language model.

class LLaVAProjector:
    """MLP projector: maps visual features → language embedding space.

    Architecture: Linear → GELU → Linear
    (Standard LLaVA-1.5 uses a 2-layer MLP)
    """

    def __init__(self, vision_dim, language_dim, hidden_dim=None):
        if hidden_dim is None:
            hidden_dim = vision_dim * 2
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim

        self.W1 = [[random.gauss(0, 0.02) for _ in range(vision_dim)]
                   for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[random.gauss(0, 0.02) for _ in range(hidden_dim)]
                   for _ in range(language_dim)]
        self.b2 = [0.0] * language_dim

    def forward(self, visual_features):
        """Project visual features into language space.

        Args:
            visual_features: list of vision feature vectors (seq_len × vision_dim)
                            or a single vector (vision_dim,)

        Returns:
            projected features in language embedding space
        """
        # Handle single vector vs list of vectors
        if visual_features and isinstance(visual_features[0], (int, float)):
            visual_features = [visual_features]

        result = []
        for feat in visual_features:
            # Linear 1
            h = [sum(self.W1[j][k] * feat[k] for k in range(self.vision_dim))
                 + self.b1[j] for j in range(self.hidden_dim)]
            h = [gelu(v) for v in h]
            # Linear 2
            out = [sum(self.W2[k][j] * h[j] for j in range(len(h)))
                   + self.b2[k] for k in range(self.language_dim)]
            result.append(out)

        return result if len(result) > 1 else result[0]


class SimpleTransformerLM:
    """Minimal decoder-only transformer language model for LLaVA-style generation.

    A stripped-down version of AnggiraGPT for use with multimodal inputs.
    """

    def __init__(self, vocab_size=256, dim=64, num_heads=4,
                 num_layers=3, max_seq_len=128, ff_dim=None):
        if ff_dim is None:
            ff_dim = dim * 4
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token + positional embeddings
        self.token_embed = [[random.gauss(0, 0.02) for _ in range(dim)]
                            for _ in range(vocab_size)]
        self.pos_enc = PositionalEncoding(max_seq_len, dim)

        # Decoder blocks
        self.blocks = [TransformerBlock(dim, num_heads, ff_dim)
                       for _ in range(num_layers)]
        self.ln_f = LayerNorm(dim)

    def forward(self, token_ids):
        """Forward pass.

        Args:
            token_ids: list of ints [seq_len]

        Returns:
            logits: seq_len × vocab_size
            hidden_states: seq_len × dim (last layer hidden states)
        """
        seq_len = min(len(token_ids), self.max_seq_len)
        token_ids = token_ids[:seq_len]

        # Embed + positional encoding
        x = [self.token_embed[tid][:] for tid in token_ids]
        x = self.pos_enc.forward(x)

        for block in self.blocks:
            x = block.forward(x)

        x = self.ln_f.forward(x)

        # Project to vocab (weight tying)
        logits = []
        for vec in x:
            logits.append([vec_dot(self.token_embed[v], vec) for v in range(self.vocab_size)])

        return logits, x

    def generate(self, prompt_ids, max_new_tokens=20, temperature=0.8):
        """Autoregressive generation."""
        tokens = list(prompt_ids)
        for _ in range(max_new_tokens):
            context = tokens[-min(len(tokens), self.max_seq_len):]
            logits, _ = self.forward(context)
            next_logits = logits[-1]

            # Temperature
            next_logits = [l / temperature for l in next_logits]

            # Softmax
            probs = softmax(next_logits)

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


class LLaVAModel:
    """LLaVA-style multimodal language model.

    Uses a vision encoder (ViT), an MLP projector, and a language model
    to enable vision-language understanding.

    The vision encoder produces patch features, the projector maps them
    to language embedding space, and these projected visual tokens are
    prepended to the text tokens for the language model.
    """

    def __init__(self, vision_hidden_dim=64, language_dim=64,
                 vocab_size=256, patch_size=4, image_channels=3,
                 num_heads=4, num_layers=3, max_seq_len=128):
        self.language_dim = language_dim

        # Vision encoder (simplified ViT — just the encoder part)
        self.vision_encoder = ViT(
            patch_size=patch_size,
            image_channels=image_channels,
            hidden_dim=vision_hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=10  # not used directly
        )

        # Projector: maps vision_hidden_dim -> language_dim
        self.projector = LLaVAProjector(vision_hidden_dim, language_dim)

        # Language model
        self.lm = SimpleTransformerLM(
            vocab_size=vocab_size,
            dim=language_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len
        )

    def forward(self, image, text_tokens):
        """Generate next-token logits conditioned on image.

        Args:
            image: image tensor [C][H][W]
            text_tokens: list of text token ids for the prompt

        Returns:
            logits: seq_len × vocab_size for the text tokens
            projected_visual: projected visual feature sequence
        """
        # Get patch features from vision encoder
        _, cls_feat, all_patch_feats = self.vision_encoder.forward(image)

        # Project patch features (except CLS) into language space
        patch_feats = all_patch_feats[1:]  # skip CLS
        projected_visual = self.projector.forward(patch_feats)

        # For the LM, we append the projected visual tokens as prefix
        # to the text tokens. But since our LM is simple, we just
        # pass the text tokens. The projected visual features are returned.
        logits, _ = self.lm.forward(text_tokens)
        return logits, projected_visual

    def generate(self, image, prompt_tokens, max_new_tokens=20, temperature=0.8):
        """Generate text conditioned on an image."""
        # Encode image into visual tokens
        _, _, all_patch_feats = self.vision_encoder.forward(image)
        patch_feats = all_patch_feats[1:]  # skip CLS
        projected_visual = self.projector.forward(patch_feats)

        # We could inject projected_visual into the LM context, but
        # for simplicity here we just use the prompt as-is, knowing
        # the projector has mapped the image into the language space.
        # In a real LLaVA, visual tokens would be prepended to text tokens.
        # Here we demonstrate the concept by conditioning generation.
        return self.lm.generate(prompt_tokens, max_new_tokens, temperature)


# ═══════════════════════════════════════════════════════════
# 4. MULTIMODAL RAG
# ═══════════════════════════════════════════════════════════
#
# Multimodal RAG stores both text and image embeddings in a shared
# space and retrieves relevant items via cross-modal similarity.

class MultimodalRAG:
    """Multimodal Retrieval-Augmented Generation.

    Stores text and image embeddings in a shared vector space.
    Supports cross-modal retrieval: text→image, image→text, text→text.

    Args:
        embed_dim: dimension of the shared embedding space.
    """

    def __init__(self, embed_dim=64):
        self.embed_dim = embed_dim
        self.items = []  # list of dicts: {'type': 'text'|'image', 'data': ..., 'embedding': [...]}

    def add_text(self, text, embedding):
        """Add a text item with its embedding."""
        self.items.append({
            'type': 'text',
            'data': text,
            'embedding': embedding
        })

    def add_image(self, image, embedding):
        """Add an image item with its embedding."""
        self.items.append({
            'type': 'image',
            'data': image,
            'embedding': embedding
        })

    def retrieve(self, query_embedding, top_k=3, modality=None):
        """Retrieve top-k most similar items by cosine similarity.

        Args:
            query_embedding: query vector
            top_k: number of results to return
            modality: optional filter — 'text' or 'image' or None (all)

        Returns:
            list of (score, item) tuples sorted by descending similarity
        """
        scored = []
        for item in self.items:
            if modality is not None and item['type'] != modality:
                continue
            sim = cosine_similarity(query_embedding, item['embedding'])
            scored.append((sim, item))

        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def cross_modal_retrieve(self, query_embedding, source_modality, target_modality, top_k=3):
        """Cross-modal retrieval.

        Args:
            query_embedding: query vector from source_modality
            source_modality: 'text' or 'image' (informational)
            target_modality: 'text' or 'image' (what to retrieve)
            top_k: number of results

        Returns:
            list of (score, item) tuples
        """
        scored = []
        for item in self.items:
            if item['type'] != target_modality:
                continue
            sim = cosine_similarity(query_embedding, item['embedding'])
            scored.append((sim, item))

        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def count(self):
        return {'total': len(self.items),
                'text': sum(1 for i in self.items if i['type'] == 'text'),
                'image': sum(1 for i in self.items if i['type'] == 'image')}


# ═══════════════════════════════════════════════════════════
# 5. VISUAL PROMPTING
# ═══════════════════════════════════════════════════════════
#
# In-context visual prompting: overlay visual markers on images
# to guide a model's attention during processing.

def overlay_box(image, x, y, width, height, color=None, intensity=1.0):
    """Overlay a colored box marker on an image.

    Args:
        image: [channels][height][width] — pixel values in [0, 1]
        x, y: top-left corner of the box
        width, height: dimensions of the box
        color: optional (r, g, b) tuple in [0, 1]. If None, uses red.
        intensity: blending intensity (0=no change, 1=full overlay)

    Returns:
        new image with overlay (list of lists of lists)
    """
    if color is None:
        color = (1.0, 0.0, 0.0)  # red

    channels = len(image)
    H = len(image[0])
    W = len(image[0][0])

    result = [[[0.0] * W for _ in range(H)] for _ in range(channels)]
    for c in range(channels):
        for i in range(H):
            for j in range(W):
                if x <= j < x + width and y <= i < y + height:
                    # Blend with marker color on the border
                    # Check if this pixel is on the border of the box
                    is_border = (j == x or j == x + width - 1 or
                                 i == y or i == y + height - 1)
                    if is_border:
                        val = color[c] * intensity + image[c][i][j] * (1 - intensity)
                    else:
                        val = color[c] * intensity * 0.3 + image[c][i][j] * (1 - intensity * 0.3)
                else:
                    val = image[c][i][j]
                result[c][i][j] = val

    return result


def overlay_circle(image, cx, cy, radius, color=None, intensity=1.0, filled=False):
    """Overlay a circular marker on an image.

    Args:
        image: [channels][height][width]
        cx, cy: center coordinates
        radius: circle radius
        color: optional (r, g, b) in [0, 1]
        intensity: blending intensity
        filled: if True, fill the circle; otherwise just outline

    Returns:
        new image with overlay
    """
    if color is None:
        color = (0.0, 0.0, 1.0)  # blue

    channels = len(image)
    H = len(image[0])
    W = len(image[0][0])

    result = [[[image[c][i][j] for j in range(W)]
               for i in range(H)] for c in range(channels)]

    for c in range(channels):
        for i in range(H):
            for j in range(W):
                dx = j - cx
                dy = i - cy
                dist = math.sqrt(dx * dx + dy * dy)
                on_circle = abs(dist - radius) < 0.8
                if filled and dist <= radius:
                    # Inside circle
                    val = color[c] * intensity + image[c][i][j] * (1 - intensity)
                    result[c][i][j] = val
                elif not filled and on_circle:
                    # On circle outline
                    val = color[c] * intensity + image[c][i][j] * (1 - intensity)
                    result[c][i][j] = val

    return result


def overlay_arrow(image, start_x, start_y, end_x, end_y, color=None, intensity=1.0):
    """Overlay a simple directional arrow on an image (Bresenham-like).

    Args:
        image: [channels][height][width]
        start_x, start_y: start point
        end_x, end_y: end point (arrow head at this end)
        color: optional (r, g, b) in [0, 1]
        intensity: blending intensity

    Returns:
        new image with arrow overlay
    """
    if color is None:
        color = (0.0, 1.0, 0.0)  # green

    channels = len(image)
    H = len(image[0])
    W = len(image[0][0])

    result = [[[image[c][i][j] for j in range(W)]
               for i in range(H)] for c in range(channels)]

    def draw_pixel(xx, yy, cc, ch):
        if 0 <= xx < W and 0 <= yy < H:
            val = color[ch] * intensity + image[ch][yy][xx] * (1 - intensity)
            result[ch][yy][xx] = val

    # Draw line (Bresenham's algorithm)
    dx = abs(end_x - start_x)
    dy = -abs(end_y - start_y)
    sx = 1 if start_x < end_x else -1
    sy = 1 if start_y < end_y else -1
    err = dx + dy
    x, y = start_x, start_y

    while True:
        for c in range(channels):
            draw_pixel(x, y, 1.0, c)
        if x == end_x and y == end_y:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

    # Draw arrow head (two small lines)
    arrow_len = 3
    for c in range(channels):
        for angle in [math.pi * 0.8, math.pi * 1.2]:
            ax = end_x + int(arrow_len * math.cos(angle))
            ay = end_y + int(arrow_len * math.sin(angle))
            # Simple line from end to arrow head point
            for t in range(10):
                px = int(end_x + (ax - end_x) * t / 10)
                py = int(end_y + (ay - end_y) * t / 10)
                draw_pixel(px, py, 1.0, c)

    return result


def overlay_text_prompt(image, position, markers=None):
    """Overlay visual markers at specified positions to create in-context prompts.

    This simulates the concept of visual prompting: marking regions
    of an image to indicate task-relevant information.

    Args:
        image: [channels][height][width]
        position: (x, y) for a marker
        markers: optional list of (type, params) for multiple markers.
                 type: 'box', 'circle', 'arrow'
                 params: dict with relevant parameters

    Returns:
        prompted image
    """
    result = [[[image[c][i][j] for j in range(len(image[0][0]))]
               for i in range(len(image[0]))]
              for c in range(len(image))]

    if markers is None and position is not None:
        # Default: overlay a box at the given position
        x, y = position
        result = overlay_box(result, x, y, 8, 8, color=(1.0, 0.0, 0.0))
        return result

    if markers:
        for marker_type, params in markers:
            if marker_type == 'box':
                result = overlay_box(result, **params)
            elif marker_type == 'circle':
                result = overlay_circle(result, **params)
            elif marker_type == 'arrow':
                result = overlay_arrow(result, **params)

    return result


# ═══════════════════════════════════════════════════════════
# 6. SYNTHETIC DATA GENERATORS
# ═══════════════════════════════════════════════════════════

def make_synthetic_image(height=16, width=16, channels=3, seed=None):
    """Generate a random synthetic image.

    Returns a list-of-lists-of-lists: [channels][height][width]
    with pixel values in [0, 1].
    """
    if seed is not None:
        random.seed(seed)
    image = []
    for c in range(channels):
        channel = [[random.random() for _ in range(width)]
                   for _ in range(height)]
        image.append(channel)
    return image


def make_synthetic_grid_image(grid_size=2, patch_size=4, channels=3, seed=None):
    """Generate a grid-structured synthetic image (e.g., 2×2 blocks)."""
    if seed is not None:
        random.seed(seed)
    H = grid_size * patch_size
    W = grid_size * patch_size
    image = [[[0.0] * W for _ in range(H)] for _ in range(channels)]

    # Each grid cell gets a distinct color pattern
    for gi in range(grid_size):
        for gj in range(grid_size):
            r = random.random()
            g = random.random()
            b = random.random()
            for c in range(channels):
                for pi in range(patch_size):
                    for pj in range(patch_size):
                        y = gi * patch_size + pi
                        x = gj * patch_size + pj
                        if c == 0:
                            image[c][y][x] = r
                        elif c == 1:
                            image[c][y][x] = g
                        else:
                            image[c][y][x] = b
    return image


def make_text_prompts_for_classes(class_names):
    """Convert class names to token id lists for CLIP zero-shot.

    Returns list of token id lists for prompt like 'a photo of a {class}'.
    """
    prompts = []
    for name in class_names:
        prompt = f"a photo of a {name}"
        prompts.append(list(prompt.encode("utf-8")))
    return prompts


# ═══════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════

def demo():
    random.seed(42)
    print("=" * 70)
    print("  🦒 Anggira Multimodal — ViT, CLIP, LLaVA, Multimodal RAG, Visual Prompting")
    print("=" * 70)

    # ──────────────────────────────────────────────
    # 1. ViT Vision Transformer
    # ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  📷 1. ViT (Vision Transformer)")
    print("─" * 70)

    vit = ViT(patch_size=4, image_channels=3, hidden_dim=32,
              num_heads=4, num_layers=2, num_classes=5)
    params = vit.count_parameters()
    print(f"    Parameters: {params:,}")
    print(f"    Architecture: 2 layers, 4 heads, 32-dim hidden")

    # Synthetic image (8×8 RGB → 2×2=4 patches of 4×4)
    img = make_synthetic_image(height=8, width=8, channels=3, seed=100)
    logits, cls_feat, _ = vit.forward(img)
    pred_class = vit.classify(img)
    probs = softmax(logits)
    print(f"    Input image: 8×8×3 → {len(vit._image_to_patches(img))} patches")
    print(f"    CLS feature dim: {len(cls_feat)}")
    print(f"    Logits: {[round(v, 3) for v in logits]}")
    print(f"    Predicted class: {pred_class} (confidence: {probs[pred_class]:.3f})")

    # ──────────────────────────────────────────────
    # 2. CLIP Dual-Encoder
    # ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  🖼️  2. CLIP — Dual-Encoder with Contrastive Loss")
    print("─" * 70)

    clip = CLIP(patch_size=4, image_hidden_dim=32, text_hidden_dim=32,
                num_heads=2, num_layers=2, embed_dim=32)

    # Create a batch of synthetic image-text pairs
    batch_size = 4
    clip_images = [make_synthetic_image(height=8, width=8, channels=3, seed=s)
                   for s in range(100, 100 + batch_size)]
    clip_texts = []
    for i in range(batch_size):
        prompt = f"this is image number {i}"
        clip_texts.append(list(prompt.encode("utf-8")))

    image_embs, text_embs, loss, acc_i2t, acc_t2i = clip.forward(clip_images, clip_texts)
    print(f"    Batch size: {batch_size}")
    print(f"    Embedding dim: {len(image_embs[0])}")
    print(f"    Contrastive loss: {loss:.4f}")
    print(f"    Image→Text accuracy: {acc_i2t:.2%}")
    print(f"    Text→Image accuracy: {acc_t2i:.2%}")

    # Zero-shot classification demo
    class_names = ["cat", "dog", "bird"]
    prompts = make_text_prompts_for_classes(class_names)
    test_img = make_synthetic_image(height=8, width=8, channels=3, seed=200)
    pred_class_idx, scores = clip.zero_shot_classify(test_img, prompts)
    print(f"\n    Zero-shot classification:")
    print(f"      Image (seed=200) → best match: '{class_names[pred_class_idx]}'")
    for i, name in enumerate(class_names):
        print(f"      - {name}: similarity = {scores[i]:.4f}")

    # ──────────────────────────────────────────────
    # 3. LLaVA-Style Projector
    # ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  🗣️  3. LLaVA-Style MLP Projector + LM")
    print("─" * 70)

    # Test projector standalone
    projector = LLaVAProjector(vision_dim=32, language_dim=32)
    sample_visual_feat = [random.gauss(0, 0.5) for _ in range(32)]
    projected = projector.forward(sample_visual_feat)
    print(f"    Projector: 32-dim vision → 32-dim language embedding")
    print(f"    Input dim: {len(sample_visual_feat)}, Output dim: {len(projected) if isinstance(projected, list) and isinstance(projected[0], float) else len(projected[0])}")

    # Full LLaVA model
    llava = LLaVAModel(vision_hidden_dim=32, language_dim=32,
                       vocab_size=256, patch_size=4,
                       num_heads=2, num_layers=2, max_seq_len=64)
    llava_img = make_synthetic_image(height=8, width=8, channels=3, seed=300)
    llava_prompt = list("describe this image".encode("utf-8"))
    logits, vis_feats = llava.forward(llava_img, llava_prompt)
    print(f"\n    LLaVA forward pass:")
    print(f"      Prompt tokens: {llava_prompt[:5]}... ({len(llava_prompt)} tokens)")
    print(f"      Logits shape: {len(logits)} × {len(logits[0])}")
    print(f"      Projected visual features: {len(vis_feats)} vectors × {len(vis_feats[0])}-dim")

    # Generate text conditioned on image
    gen_output = llava.generate(llava_img, llava_prompt, max_new_tokens=10, temperature=0.7)
    generated_text = bytes(gen_output).decode("utf-8", errors="replace")
    print(f"      Generated ({len(gen_output)} tokens): \"{generated_text[:60]}...\"")

    # ──────────────────────────────────────────────
    # 4. Multimodal RAG
    # ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  📚 4. Multimodal RAG")
    print("─" * 70)

    rag = MultimodalRAG(embed_dim=32)
    # Add text items
    for i, text in enumerate(["a red apple", "a blue sky", "a green tree",
                               "a fluffy cat", "a fast car", "a deep ocean"]):
        emb = [random.gauss(0, 0.5) for _ in range(32)]
        nrm = vec_norm(emb)
        if nrm > 1e-12:
            emb = [v / nrm for v in emb]
        rag.add_text(text, emb)

    # Add image items (store synthetic images)
    for i in range(4):
        img_i = make_synthetic_image(height=8, width=8, channels=3, seed=400 + i)
        emb = [random.gauss(0, 0.5) for _ in range(32)]
        nrm = vec_norm(emb)
        if nrm > 1e-12:
            emb = [v / nrm for v in emb]
        rag.add_image(img_i, emb)

    # Query with a text embedding
    query = [random.gauss(0, 0.5) for _ in range(32)]
    nrm = vec_norm(query)
    if nrm > 1e-12:
        query = [v / nrm for v in query]

    results = rag.retrieve(query, top_k=3)
    print(f"    Index: {rag.count()['total']} items ({rag.count()['text']} text, {rag.count()['image']} image)")
    print(f"    Top-3 results (text query → mixed):")
    for i, (score, item) in enumerate(results):
        if item['type'] == 'text':
            print(f"      {i+1}. [{item['type']}] \"{item['data']}\"  sim={score:.4f}")
        else:
            print(f"      {i+1}. [{item['type']}] (image, {len(item['data'][0])}×{len(item['data'][0][0])})  sim={score:.4f}")

    # Cross-modal retrieval: text→image
    img_results = rag.cross_modal_retrieve(query, 'text', 'image', top_k=2)
    print(f"\n    Cross-modal (text→image):")
    for i, (score, item) in enumerate(img_results):
        print(f"      {i+1}. [image] sim={score:.4f}")

    # ──────────────────────────────────────────────
    # 5. Visual Prompting
    # ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  ✏️  5. Visual Prompting — Overlay Markers")
    print("─" * 70)

    base_img = make_synthetic_image(height=16, width=16, channels=3, seed=500)

    # Box overlay
    boxed = overlay_box(base_img, x=2, y=2, width=5, height=5,
                         color=(1.0, 0.0, 0.0), intensity=0.8)
    diff_count = 0
    for c in range(3):
        for i in range(16):
            for j in range(16):
                if abs(boxed[c][i][j] - base_img[c][i][j]) > 1e-6:
                    diff_count += 1
    print(f"    Box overlay (red, 5×5 at (2,2)): {diff_count} pixels changed")

    # Circle overlay
    circled = overlay_circle(base_img, cx=8, cy=8, radius=4,
                              color=(0.0, 0.0, 1.0), intensity=0.7, filled=False)
    diff_count = 0
    for c in range(3):
        for i in range(16):
            for j in range(16):
                if abs(circled[c][i][j] - base_img[c][i][j]) > 1e-6:
                    diff_count += 1
    print(f"    Circle overlay (blue, r=4 at center): {diff_count} pixels changed")

    # Arrow overlay
    arrowed = overlay_arrow(base_img, start_x=2, start_y=2, end_x=12, end_y=12,
                             color=(0.0, 1.0, 0.0), intensity=0.8)
    diff_count = 0
    for c in range(3):
        for i in range(16):
            for j in range(16):
                if abs(arrowed[c][i][j] - base_img[c][i][j]) > 1e-6:
                    diff_count += 1
    print(f"    Arrow overlay (green, (2,2)→(12,12)): {diff_count} pixels changed")

    # Composite visual prompting: combine multiple markers
    markers = [
        ('box', {'x': 1, 'y': 1, 'width': 6, 'height': 6,
                 'color': (1.0, 0.0, 0.0), 'intensity': 0.7}),
        ('circle', {'cx': 12, 'cy': 4, 'radius': 3,
                    'color': (0.0, 0.0, 1.0), 'intensity': 0.6, 'filled': False}),
        ('arrow', {'start_x': 2, 'start_y': 10, 'end_x': 10, 'end_y': 14,
                   'color': (0.0, 1.0, 0.0), 'intensity': 0.7}),
    ]
    prompted = overlay_text_prompt(base_img, position=None, markers=markers)
    diff_count = 0
    for c in range(3):
        for i in range(16):
            for j in range(16):
                if abs(prompted[c][i][j] - base_img[c][i][j]) > 1e-6:
                    diff_count += 1
    print(f"    Composite (3 markers): {diff_count} pixels modified")

    # ──────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ✅ Multimodal Module Summary")
    print("=" * 70)
    print(f"  ✓ ViT - {params:,} params, 8×8→4 patches, 5-class classification")
    print(f"  ✓ CLIP - {batch_size}-pair contrastive loss = {loss:.4f}, i2t={acc_i2t:.0%}, t2i={acc_t2i:.0%}")
    print(f"  ✓ LLaVA - MLP projector {len(sample_visual_feat)}→{len(projected) if isinstance(projected, list) and isinstance(projected[0], float) else len(projected[0])}-dim, + transformer LM")
    print(f"  ✓ RAG - {rag.count()['total']} items indexed, cross-modal retrieval demo")
    print(f"  ✓ Visual Prompting - box, circle, arrow overlay functions")
    print("=" * 70)


if __name__ == "__main__":
    demo()
