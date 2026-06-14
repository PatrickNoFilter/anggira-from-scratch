"""Train AnggiraGPT on TinyStories with full backprop."""
import sys
import os
import pickle
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from anggira.gpt import AnggiraGPT


def load_data(filepath, encode_fn, seq_len=128, max_tokens=None):
    """Tokenize the dataset into a 1D array."""
    tokens = [2]  # <BOS>
    story_count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            encoded = encode_fn(line)
            tokens.extend(encoded)
            tokens.append(3)  # <EOS>
            story_count += 1
            if max_tokens and len(tokens) >= max_tokens:
                break

    arr = np.array(tokens[:max_tokens] if max_tokens else tokens, dtype=np.int32)
    print(f"  Tokenized {story_count} stories -> {len(arr)} tokens")
    return arr


def train_model(vocab_size=4000, dim=256, num_heads=8, num_layers=6,
                max_seq_len=128, ff_dim=1024,
                num_steps=5000, batch_size=1, seq_len=128,
                lr=3e-4, lr_warmup=200, log_every=100):
    """Create, train, and return a model."""
    print(f"\n{'='*55}")
    print(f"🤖 AnggiraGPT — Full Backprop Training")
    print(f"{'='*55}")
    print(f"\nModel config:")
    print(f"  Vocab: {vocab_size}, Dim: {dim}, Heads: {num_heads}")
    print(f"  Layers: {num_layers}, Seq: {max_seq_len}, FF: {ff_dim}")

    model = AnggiraGPT(
        vocab_size=vocab_size,
        dim=dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        ff_dim=ff_dim,
    )
    total_params = model.count_params()
    print(f"  Total params: {total_params:,} ({total_params*4/1024/1024:.1f} MB as float32)")

    # Load vocab
    vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'data', 'vocab.pkl')
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    encode_fn = lambda text: [
        vocab_data['word2idx'].get(w, 1)
        for w in __import__('re').findall(r"[a-zA-Z0-9'']+(?:[''][a-zA-Z]+)?", text.lower())
    ]

    # Load data
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'tinystories_5k.txt')
    print(f"\nLoading data...")
    tokens = load_data(data_path, encode_fn, seq_len, max_tokens=200000)

    print(f"\nStarting training ({num_steps} steps)...")
    t0 = time.time()
    losses = model.train_on_data(
        tokens,
        seq_len=seq_len,
        num_steps=num_steps,
        lr=lr,
        lr_warmup=lr_warmup,
        log_every=log_every,
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s ({elapsed/num_steps:.2f}s/step)")
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    if len(losses) >= 100:
        print(f"  Avg last 100: {np.mean(losses[-100:]):.4f}")

    # Save model
    weights = {}
    weights['token_embed'] = model.embedding.token_embed
    weights['pos_embed'] = model.embedding.pos_embed
    for i, block in enumerate(model.blocks):
        for sub, attr in [('attn', ['W_q', 'W_k', 'W_v', 'W_o']),
                          ('ffn', ['W1', 'b1', 'W2', 'b2']),
                          ('ln1', ['gamma', 'beta']),
                          ('ln2', ['gamma', 'beta'])]:
            for a in attr:
                weights[f'block{i}_{sub}_{a}'] = getattr(getattr(block, sub), a)
    weights['ln_f_gamma'] = model.ln_f.gamma
    weights['ln_f_beta'] = model.ln_f.beta

    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'data', 'gpt_weights.npz')
    np.savez_compressed(save_path, **weights)
    print(f"\nWeights saved to {save_path} ({len(weights)} arrays)")

    return model, losses


def generate_sample(model, decode_fn, prompt="Once upon a time",
                    max_new=50, temp=0.8, top_k=40):
    """Generate a sample from the model."""
    import re
    # Tokenize prompt
    from anggira.nlp import BPETokenizer
    words = re.findall(r"[a-zA-Z0-9'']+(?:[''][a-zA-Z]+)?", prompt.lower())
    prompt_ids = [2]  # <BOS>
    vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'data', 'vocab.pkl')
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    word2idx = vocab_data['word2idx']
    for w in words:
        prompt_ids.append(word2idx.get(w, word2idx['<UNK>']))

    # Generate
    out_ids = model.generate(prompt_ids, max_new=max_new, temp=temp, top_k=top_k)
    generated = decode_fn(out_ids[len(prompt_ids):])

    return prompt + generated


if __name__ == '__main__':
    model, losses = train_model(
        vocab_size=4000,
        dim=256,
        num_heads=8,
        num_layers=6,
        max_seq_len=128,
        ff_dim=1024,
        num_steps=3000,
        seq_len=128,
        lr=3e-4,
        lr_warmup=200,
        log_every=100,
    )
    # Generate a sample
    for prompt in ["once upon a time", "lily found a", "the little boy"]:
        text = generate_sample(model, lambda ids: ' '.join(
            __import__('pickle').load(open(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'vocab.pkl'), 'rb'))['idx2word'].get(i, '<UNK>')
            for i in ids
        ), prompt=prompt, max_new=30, temp=0.8, top_k=40)
        print(f"\nPrompt: {prompt!r}")
        print(f"Generated: {text}")
