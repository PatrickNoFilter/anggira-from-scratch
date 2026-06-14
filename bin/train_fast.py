"""Train AnggiraGPT — optimized for phone-friendly runtime."""
import sys, os, pickle, re, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(42)

from anggira.gpt import AnggiraGPT

# ── Config ──────────────────────────────────────────────────────
VOCAB_SIZE = 2944
DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
MAX_SEQ_LEN = 64
FF_DIM = 256
NUM_STEPS = 5000
SEQ_LEN = 64
LR = 3e-4
LR_WARMUP = 200
LOG_EVERY = 500
NUM_STORIES = 1000
# ───────────────────────────────────────────────────────────────

# Load vocab
vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'vocab.pkl')
with open(vocab_path, 'rb') as f:
    v = pickle.load(f)
w2i, i2w = v['word2idx'], v['idx2word']

def encode(text):
    words = re.findall(r"[a-zA-Z0-9'']+(?:[''][a-zA-Z]+)?", text.lower())
    return [w2i.get(w, 1) for w in words]

# Load data
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'tinystories_5k.txt')
tokens = [2]
with open(data_path) as f:
    for i, line in enumerate(f):
        if i >= NUM_STORIES:
            break
        line = line.strip()
        if line:
            tokens.extend(encode(line))
            tokens.append(3)

arr = np.array(tokens, dtype=np.int32)
print(f"📚 Loaded {NUM_STORIES} stories → {len(arr)} tokens")

# Build model
print(f"\n🤖 Building model:")
model = AnggiraGPT(
    vocab_size=VOCAB_SIZE,
    dim=DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    ff_dim=FF_DIM,
)
print(f"   Config: dim={DIM}, heads={NUM_HEADS}, layers={NUM_LAYERS}")
print(f"   Params: {model.count_params():,}")

# Train
print(f"\n🏋️  Training: {NUM_STEPS} steps...")
print(f"   LR={LR}, warmup={LR_WARMUP} steps, seq_len={SEQ_LEN}")
t0 = time.time()
losses = model.train_on_data(arr, seq_len=SEQ_LEN, num_steps=NUM_STEPS,
                              lr=LR, lr_warmup=LR_WARMUP, log_every=LOG_EVERY)
elapsed = time.time() - t0
print(f"\n✅ Done! {elapsed:.0f}s ({elapsed/NUM_STEPS:.2f}s/step)")
print(f"   Loss: {losses[0]:.4f} → {losses[-1]:.4f}")

# Save weights
save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
weights = {
    'token_embed': model.embedding.token_embed,
    'pos_embed': model.embedding.pos_embed,
}
for i, block in enumerate(model.blocks):
    for sub, attrs in [('attn', ['W_q','W_k','W_v','W_o']),
                       ('ffn', ['W1','b1','W2','b2']),
                       ('ln1', ['gamma','beta']),
                       ('ln2', ['gamma','beta'])]:
        for a in attrs:
            weights[f'b{i}_{sub}_{a}'] = getattr(getattr(block, sub), a)
weights['ln_f_gamma'] = model.ln_f.gamma
weights['ln_f_beta'] = model.ln_f.beta

save_path = os.path.join(save_dir, 'gpt_weights.npz')
np.savez_compressed(save_path, **weights)
print(f"   Weights saved: {save_path}")

# Generate samples
print(f"\n🎯 Generation samples:")
for prompt in ['once upon a time', 'lily found', 'the dog', 'the boy and']:
    pids = [2] + encode(prompt)
    out = model.generate(pids, max_new=30, temp=0.9, top_k=30)
    gen = ' '.join(i2w.get(i, '<UNK>') for i in out[len(pids):])
    # Clean up: remove <EOS> and anything after it
    if '<EOS>' in gen:
        gen = gen.split('<EOS>')[0] + '.'
    print(f"\n  \"{prompt}\" → \"{gen}\"")
