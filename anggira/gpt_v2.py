"""Anggira V2 — Scaled GPT using AnggiraGPT internals.

V2 specs:
  - d_model: 144, num_heads: 6, num_layers: 5, ff_dim: 576
  - max_seq_len: 512
  - BPE tokenizer (trained, ~5000 tokens)
  - ~2.0M params

Reuses same LayerNorm, MultiHeadAttention, FeedForward, TransformerBlock
from gpt.py. Only difference is config and using BPE tokenizer.
"""

import sys, os, pickle, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anggira.gpt import AnggiraGPT


# V2 Config
VOCAB_SIZE = 5000  # will be overridden by actual BPE vocab
DIM = 144
NUM_HEADS = 6       # 144 / 6 = 24 per head
NUM_LAYERS = 5
MAX_SEQ_LEN = 512
FF_DIM = 576        # 4 * DIM
SEQ_LEN = 512

LR = 3e-4
LR_WARMUP = 100
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.1
BURST = 25
BATCH_SIZE = 2
LOG_EVERY = 5


def create_v2_model(vocab_size=VOCAB_SIZE):
    """Create an Anggira V2 model with the scaled config."""
    model = AnggiraGPT(
        vocab_size=vocab_size,
        dim=DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        ff_dim=FF_DIM,
        dropout_rate=0.1,
    )
    return model


def count_model_params():
    """Count params for the V2 architecture."""
    model = create_v2_model()
    n = model.count_params()
    print(f"V2 model: {n:,} params")
    # Breakdown
    V, D, L, H, FF = VOCAB_SIZE, DIM, NUM_LAYERS, NUM_HEADS, FF_DIM
    emb = V * D + MAX_SEQ_LEN * D
    attn = L * 4 * D * D  # W_q, W_k, W_v, W_o
    ffn = L * (D * FF + FF + FF * D + D)  # W1, b1, W2, b2
    ln = L * 4 * D  # gamma + beta for ln1, ln2 per layer
    ln_f = 2 * D
    print(f"  Embedding: {emb:,}")
    print(f"  Attention (all layers): {attn:,}")
    print(f"  FFN (all layers): {ffn:,}")
    print(f"  LayerNorms (all): {ln:,}")
    print(f"  Final LN: {ln_f:,}")
    print(f"  Total: {emb + attn + ffn + ln + ln_f:,}")
    return n


if __name__ == '__main__':
    count_model_params()
