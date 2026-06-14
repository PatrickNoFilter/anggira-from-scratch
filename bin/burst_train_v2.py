#!/usr/bin/env python3
"""Anggira V2 — Burst trainer for mixed Indonesian + code corpus.

Usage:
  python3 -u bin/burst_train_v2.py
  # Run again = continue (auto-resumes from checkpoint)

Uses:
  - BPE tokenizer (data/indonesian_corpus_v2/tokenizer.pkl)
  - Scaled model: d_model=144, 5L, 6H, T=512, ~2.0M params
  - Burst pattern: 25 steps ~48s on phone
"""

import sys, os, pickle, time
import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(42)

# ── V2 Config ──────────────────────────────────────────────────
VOCAB_SIZE = 5000
DIM = 144
NUM_HEADS = 6       # 144 / 6 = 24
NUM_LAYERS = 5
MAX_SEQ_LEN = 512
FF_DIM = 576        # 4 * DIM
SEQ_LEN = 512

LR = 3e-4
LR_WARMUP = 100
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.1

BURST = 12
BATCH_SIZE = 2
LOG_EVERY = 3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'indonesian_corpus_v2')
CHKPT_FILE = os.path.join(DATA_DIR, 'checkpoint.pkl')
TOK_PATH = os.path.join(DATA_DIR, 'tokenizer.pkl')
TOKENS_PATH = os.path.join(DATA_DIR, 'tokens.npy')

# ── Load BPE tokenizer ─────────────────────────────────────────
from anggira.bpe_tokenizer import BPETokenizer
tok = BPETokenizer()
tok.load(TOK_PATH)
VOCAB_SIZE = tok.vocab_size
print(f"🔤  BPE tokenizer: {VOCAB_SIZE} tokens loaded")

# ── Load tokenized corpus ──────────────────────────────────────
arr = np.load(TOKENS_PATH)
print(f"📚  Tokenized corpus: {len(arr):,} tokens")
# Count documents (BOS tokens = 2)
doc_count = np.sum(arr == 2)
print(f"   Documents: {doc_count:,}")

# ── Model ──────────────────────────────────────────────────────
from anggira.gpt import AnggiraGPT

model = AnggiraGPT(
    vocab_size=VOCAB_SIZE,
    dim=DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    ff_dim=FF_DIM,
    dropout_rate=0.1,
)

# Disable dropout during inference
model.training = True

total_steps = 0
best_loss = float('inf')

params_count = model.count_params()
print(f"🤖  Anggira V2: {params_count:,} params (vocab={VOCAB_SIZE}, dim={DIM}, L={NUM_LAYERS}, H={NUM_HEADS}, T={MAX_SEQ_LEN})")

# ── Load checkpoint if exists ──────────────────────────────────
loaded_optimizer_state = None
if os.path.exists(CHKPT_FILE):
    print(f"📦  Loading checkpoint: {CHKPT_FILE}")
    cp = pickle.load(open(CHKPT_FILE, 'rb'))
    cp_vocab = cp.get('vocab_size', 0)
    if cp_vocab != VOCAB_SIZE:
        print(f"   ⚠️  Checkpoint vocab={cp_vocab} != current vocab={VOCAB_SIZE}")
        print(f"   ⚠️  Loading weights that match...")

    model.embedding.token_embed[:] = cp['token_embed']
    old_pos = cp['pos_embed']
    new_pos = model.embedding.pos_embed
    if old_pos.shape[0] < new_pos.shape[0]:
        new_pos[:old_pos.shape[0]] = old_pos
        # Extend by repeating last position
        new_pos[old_pos.shape[0]:] = old_pos[-1:]
        print(f"   ↗️  Extended pos_embed {old_pos.shape[0]}→{new_pos.shape[0]}")
    else:
        new_pos[:] = old_pos

    for i in range(NUM_LAYERS):
        b = model.blocks[i]
        for sub, attrs in [('attn', ['W_q','W_k','W_v','W_o']),
                           ('ffn', ['W1','b1','W2','b2']),
                           ('ln1', ['gamma','beta']),
                           ('ln2', ['gamma','beta'])]:
            for a in attrs:
                key = f'b{i}_{sub}_{a}'
                if key in cp and cp[key].shape == getattr(getattr(b, sub), a).shape:
                    getattr(getattr(b, sub), a)[:] = cp[key]

    model.ln_f.gamma[:] = cp['ln_f_gamma']
    model.ln_f.beta[:] = cp['ln_f_beta']
    total_steps = cp.get('step', 0)
    best_loss = cp.get('best_loss', float('inf'))
    loaded_optimizer_state = cp.get('optimizer')
    print(f"💾  Resumed step {total_steps} (best loss: {best_loss:.4f})")
else:
    print(f"🆕  Fresh V2 model: {params_count:,} params")
    print(f"   Optimizer: AdamW lr={LR}, wd={WEIGHT_DECAY}, batch={BATCH_SIZE}")

# ── Optimizer ──────────────────────────────────────────────────
from anggira.optimizer import AdamW

optimizer = AdamW(model.collect_params(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
if loaded_optimizer_state is not None:
    optimizer.set_state(loaded_optimizer_state)
    print(f"   AdamW state restored (step={optimizer.t})")

# ── Training burst ─────────────────────────────────────────────
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

    # Vectorized batch sampling (avoids Python loop overhead)
    max_idx = max(1, len(arr) - SEQ_LEN - 1)
    indices = np.random.randint(0, max_idx, size=BATCH_SIZE)
    inp = np.stack([arr[i:i + SEQ_LEN] for i in indices], axis=0)
    tgt = np.stack([arr[i + 1:i + SEQ_LEN + 1] for i in indices], axis=0)

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
    'vocab_size': VOCAB_SIZE,
    'token_embed': model.embedding.token_embed,
    'pos_embed': model.embedding.pos_embed,
    'optimizer': optimizer.get_state(),
}
for i, block in enumerate(model.blocks):
    for sub, attrs in [('attn', ['W_q','W_k','W_v','W_o']),
                       ('ffn', ['W1','b1','W2','b2']),
                       ('ln1', ['gamma','beta']),
                       ('ln2', ['gamma','beta'])]:
        for a in attrs:
            cp[f'b{i}_{sub}_{a}'] = getattr(getattr(block, sub), a)
cp['ln_f_gamma'] = model.ln_f.gamma
cp['ln_f_beta'] = model.ln_f.beta
pickle.dump(cp, open(CHKPT_FILE, 'wb'))
print(f"💾  Checkpoint saved: step {total_steps} -> {CHKPT_FILE}")

# ── Generate sample ────────────────────────────────────────────
# Try generating from 3 different prompts
prompts = [
    "Indonesia adalah",
    "def hello",
    "machine learning",
    "Prabowo Subianto",
    "import numpy",
]

model.training = False
for prompt in prompts:
    pids = tok.encode(prompt, bos=True)
    if len(pids) > SEQ_LEN - 10:
        pids = pids[:SEQ_LEN - 10]
    
    out = model.generate(np.array(pids, dtype=np.int32), max_new=20, temp=0.9, top_k=30)
    
    gen = tok.decode(out)
    # Remove special tokens from output for display
    gen_clean = gen.replace('<BOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()
    print(f"\n🎯  \"{prompt}\" → \"{gen_clean}\"")

# ── Quick Grammar Score ────────────────────────────────────────
# Build bigram stats (sample for speed)
print(f"\n📊 Calculating grammar score...")
bigram_counts = {}
unigram_counts = {}
sample_size = min(50000, len(arr) - 2)
sample_idx = np.random.choice(len(arr) - 2, sample_size, replace=False)
for idx in sample_idx:
    t1, t2 = int(arr[idx]), int(arr[idx + 1])
    unigram_counts[t1] = unigram_counts.get(t1, 0) + 1
    bigram_counts[(t1, t2)] = bigram_counts.get((t1, t2), 0) + 1

# Evaluate on one prompt
test_prompt = "Indonesia adalah"
pids = tok.encode(test_prompt, bos=True)
if len(pids) > SEQ_LEN - 10:
    pids = pids[:SEQ_LEN - 10]
out = model.generate(np.array(pids, dtype=np.int32), max_new=30, temp=0.7, top_k=30)
gen = tok.decode(out).replace('<BOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()

# Perplexity
log_prob = 0.0
count = 0
smooth = 1.0
for i in range(1, len(out)):
    prev_id, curr_id = int(out[i - 1]), int(out[i])
    bg = bigram_counts.get((prev_id, curr_id), 0)
    ug = unigram_counts.get(prev_id, 0)
    prob = (bg + smooth) / (ug + smooth * VOCAB_SIZE)
    log_prob += np.log2(prob + 1e-10)
    count += 1
perplexity = 2 ** (-log_prob / max(count, 1))

# Word validity
special = {0, 1, 2, 3}
valid = sum(1 for t in out if int(t) not in special)
validity = valid / max(len(out), 1) * 100

# Structure
struct_score = 0
if gen and gen[0].isupper(): struct_score += 1
if gen and gen[-1] in '.!?': struct_score += 1
if '  ' not in gen: struct_score += 1
if ' ' in gen: struct_score += 1
alpha_ratio = sum(c.isalpha() for c in gen) / max(len(gen), 1)
if alpha_ratio > 0.5: struct_score += 1
words = gen.split()
if len(words) > 1 and len(set(words)) / len(words) > 0.5: struct_score += 1
structure = struct_score / 6 * 100

# Combined score
ppl_score = max(0, min(100, 100 - (perplexity - 1) * 100 / 999))
grammar_score = ppl_score * 0.4 + validity * 0.3 + structure * 0.3

print(f"📈  Grammar Score: {grammar_score:.1f}% | PPL: {perplexity:.0f} | Valid: {validity:.0f}% | Struct: {structure:.0f}%")
model.training = True

print(f"\n📋  Run again to continue training")
