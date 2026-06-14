"""Quick burst with higher LR for faster learning."""
import sys, os, pickle, re, time
import numpy as np
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.random.seed(42)
from anggira.gpt import AnggiraGPT

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CHKPT_FILE = os.path.join(DATA_DIR, 'checkpoint.pkl')

VOCAB_SIZE = 2944; DIM = 64; NUM_HEADS = 4; NUM_LAYERS = 3
MAX_SEQ_LEN = 64; FF_DIM = 256; SEQ_LEN = 64
LR = 3e-4  # keep but let's try higher
BURST = 200
LOG_EVERY = 50

with open(os.path.join(DATA_DIR, 'vocab.pkl'), 'rb') as f:
    v = pickle.load(f)
w2i, i2w = v['word2idx'], v['idx2word']

def encode(text):
    words = re.findall(r"[a-zA-Z0-9'']+(?:[''][a-zA-Z]+)?", text.lower())
    return [w2i.get(w, 1) for w in words]

cache_file = os.path.join(DATA_DIR, 'tokens.npy')
arr = np.load(cache_file) if os.path.exists(cache_file) else np.array([2])

model = AnggiraGPT(vocab_size=VOCAB_SIZE, dim=DIM, num_heads=NUM_HEADS,
                   num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN, ff_dim=FF_DIM)

total_steps = 0
best_loss = float('inf')

if os.path.exists(CHKPT_FILE):
    cp = pickle.load(open(CHKPT_FILE, 'rb'))
    model.embedding.token_embed = cp['token_embed']
    model.embedding.pos_embed = cp['pos_embed']
    for i in range(NUM_LAYERS):
        b = model.blocks[i]
        for sub, attrs in [('attn',['W_q','W_k','W_v','W_o']),
                           ('ffn',['W1','b1','W2','b2']),
                           ('ln1',['gamma','beta']),
                           ('ln2',['gamma','beta'])]:
            for a in attrs:
                setattr(getattr(b, sub), a, cp[f'b{i}_{sub}_{a}'])
    model.ln_f.gamma = cp['ln_f_gamma']
    model.ln_f.beta = cp['ln_f_beta']
    total_steps = cp.get('step', 0)
    best_loss = cp.get('best_loss', float('inf'))
    print(f"💾 Resumed from step {total_steps} (best loss: {best_loss:.4f})")

# Try higher learning rates
for test_lr in [1e-3, 3e-3, 1e-2]:
    print(f"\n{'='*50}")
    print(f"🧪 Testing LR={test_lr}")
    # Save pre-test weights
    keep = {k: v.copy() for k,v in model.__dict__.items() if isinstance(v, np.ndarray)}
    for i, block in enumerate(model.blocks):
        for sub in ['attn','ffn','ln1','ln2']:
            for k,v in getattr(block, sub).__dict__.items():
                if isinstance(v, np.ndarray):
                    keep[f'{i}_{sub}_{k}'] = v.copy()
    for k,v in model.ln_f.__dict__.items():
        if isinstance(v, np.ndarray):
            keep[f'ln_f_{k}'] = v.copy()

    t0 = time.time()
    loss_sum = 0
    for step in range(100):
        i = np.random.randint(0, max(1, len(arr) - SEQ_LEN - 1))
        batch = arr[i:i + SEQ_LEN + 1]
        inp = batch[:-1].reshape(1, -1)
        tgt = batch[1:].reshape(1, -1)
        loss = model.train_step(inp, tgt, test_lr)
        loss_sum += loss
    avg = loss_sum / 100
    elapsed = time.time() - t0
    print(f"  100 steps: avg loss={avg:.4f}, {elapsed:.1f}s ({elapsed/100*1000:.0f}ms/step)")

    # Restore weights for next LR test
    for k,v in keep.items():
        if k.startswith('ln_f_') or k.startswith('embed_'):
            setattr(model, k.split('_',1)[1] if k.startswith('embed_') else k, v)
        elif '_' in k and any(x in k for x in ['attn','ffn','ln1','ln2']):
            parts = k.split('_', 2)
            bi, sub, attr = int(parts[0]), parts[1], parts[2]
            setattr(getattr(model.blocks[bi], sub), attr, v)

print(f"\n{'='*50}")
print(f"✅ LR test complete — best LR from above")
print(f"   Then: python3 -u bin/burst_train.py (continues normally)")
