"""Incremental burst trainer for Indonesian corpus.

Same architecture as burst_train.py but with:
  - VOCAB_SIZE=5004 matching Indonesian vocabulary
  - Data from data/indonesian_corpus/
  - Separate checkpoint independent of English model
  - Indonesian text generation sample

Usage:
  python3 -u bin/burst_train_id.py
  # run again = continue
"""

import sys, os, pickle, re, time
import numpy as np
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
np.random.seed(42)

from anggira.gpt import AnggiraGPT
from anggira.optimizer import AdamW

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'indonesian_corpus')
CHKPT_FILE = os.path.join(DATA_DIR, 'checkpoint.pkl')

VOCAB_SIZE = 5004
DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
MAX_SEQ_LEN = 160
FF_DIM = 256
SEQ_LEN = 160

LR = 3e-4
LR_WARMUP = 50
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.1

BURST = 50
BATCH_SIZE = 2
LOG_EVERY = 10

# ── Setup ───────────────────────────────────────────────────────
with open(os.path.join(DATA_DIR, 'vocab.pkl'), 'rb') as f:
    v = pickle.load(f)
w2i, i2w = v['word2idx'], v['idx2word']
print(f"🇮🇩  Indonesian vocab: {len(w2i)} tokens")

# Load tokenized corpus
arr = np.load(os.path.join(DATA_DIR, 'tokens.npy'))
print(f"📚  Tokenized corpus: {len(arr):,} tokens, {np.sum(arr == 2):,} documents")

# ── Model ───────────────────────────────────────────────────────
model = AnggiraGPT(vocab_size=VOCAB_SIZE, dim=DIM, num_heads=NUM_HEADS,
                   num_layers=NUM_LAYERS, max_seq_len=MAX_SEQ_LEN, ff_dim=FF_DIM)

total_steps = 0
best_loss = float('inf')

params_count = model.count_params()
print(f"🤖  Fresh Indonesian model: {params_count:,} params (vocab={VOCAB_SIZE})")

# Load checkpoint if exists
loaded_optimizer_state = None
if os.path.exists(CHKPT_FILE):
    cp = pickle.load(open(CHKPT_FILE, 'rb'))
    model.embedding.token_embed[:] = cp['token_embed']
    old_pos = cp['pos_embed']
    new_pos = model.embedding.pos_embed
    if old_pos.shape[0] < new_pos.shape[0]:
        new_pos[:old_pos.shape[0]] = old_pos
        new_pos[old_pos.shape[0]:] = old_pos[-1:]
        print(f"   ↗️  Extended pos_embed {old_pos.shape[0]}→{new_pos.shape[0]} tokens")
    else:
        model.embedding.pos_embed[:] = old_pos
    for i in range(NUM_LAYERS):
        b = model.blocks[i]
        for sub, attrs in [('attn',['W_q','W_k','W_v','W_o']),
                           ('ffn',['W1','b1','W2','b2']),
                           ('ln1',['gamma','beta']),
                           ('ln2',['gamma','beta'])]:
            for a in attrs:
                getattr(getattr(b, sub), a)[:] = cp[f'b{i}_{sub}_{a}']
    model.ln_f.gamma[:] = cp['ln_f_gamma']
    model.ln_f.beta[:] = cp['ln_f_beta']
    total_steps = cp.get('step', 0)
    best_loss = cp.get('best_loss', float('inf'))
    loaded_optimizer_state = cp.get('optimizer')
    print(f"💾  Resumed step {total_steps} (best loss: {best_loss:.4f})")
else:
    print(f"   Optimizer: AdamW lr={LR}, wd={WEIGHT_DECAY}, batch={BATCH_SIZE}")

optimizer = AdamW(model.collect_params(), lr=LR, betas=BETAS,
                  weight_decay=WEIGHT_DECAY)
if loaded_optimizer_state is not None:
    optimizer.set_state(loaded_optimizer_state)
    print(f"   AdamW state restored (step={optimizer.t})")

# ── Train burst ─────────────────────────────────────────────────
print(f"\n🏋️  Training burst of {BURST} steps (continues at step {total_steps})...")
t0 = time.time()
losses = []
for step in range(BURST):
    cur_step = total_steps + step

    # LR schedule: linear warmup then constant
    if cur_step < LR_WARMUP:
        optimizer.lr = LR * (cur_step + 1) / LR_WARMUP
    else:
        optimizer.lr = LR

    # Sample batch_size random segments
    inputs = []
    targets = []
    for _ in range(BATCH_SIZE):
        i = np.random.randint(0, max(1, len(arr) - SEQ_LEN - 1))
        batch = arr[i:i + SEQ_LEN + 1]
        inputs.append(batch[:-1])
        targets.append(batch[1:])
    inp = np.stack(inputs, axis=0)
    tgt = np.stack(targets, axis=0)

    loss = model.train_step(inp, tgt, optimizer)
    losses.append(loss)
    if loss < best_loss:
        best_loss = loss

    if step % LOG_EVERY == 0 or step == BURST - 1 or step == 0:
        print(f"  Step {cur_step:5d} | loss={loss:.4f} | lr={optimizer.lr:.6f} | best={best_loss:.4f}")

elapsed = time.time() - t0
print(f"✅  Burst done in {elapsed:.1f}s ({elapsed/BURST*1000:.0f}ms/step)")
print(f"   Loss: {losses[0]:.4f} → {losses[-1]:.4f}")

# ── Save checkpoint ──────────────────────────────────────────
total_steps += BURST
cp = {
    'step': total_steps,
    'best_loss': best_loss,
    'token_embed': model.embedding.token_embed,
    'pos_embed': model.embedding.pos_embed,
    'optimizer': optimizer.get_state(),
}
for i, block in enumerate(model.blocks):
    for sub, attrs in [('attn',['W_q','W_k','W_v','W_o']),
                       ('ffn',['W1','b1','W2','b2']),
                       ('ln1',['gamma','beta']),
                       ('ln2',['gamma','beta'])]:
        for a in attrs:
            cp[f'b{i}_{sub}_{a}'] = getattr(getattr(block, sub), a)
cp['ln_f_gamma'] = model.ln_f.gamma
cp['ln_f_beta'] = model.ln_f.beta
pickle.dump(cp, open(CHKPT_FILE, 'wb'))
print(f"💾  Checkpoint saved: step {total_steps} -> {CHKPT_FILE}")

# ── Generate Indonesian sample ─────────────────────────────────
# Prompt in Indonesian
id_prompt = "indonesia adalah"
pids = [2] + [w2i.get(w, 1) for w in id_prompt.split()]
out = model.generate(pids, max_new=30, temp=0.9, top_k=30)
gen = ' '.join(i2w.get(i, '<UNK>') for i in out[len(pids):])
if '<EOS>' in gen:
    gen = gen.split('<EOS>')[0] + '.'
print(f"\n🎯  Sample: \"{id_prompt}\" → \"{gen}\"")

print(f"\n📋  Run again to continue training (press up+enter)")
