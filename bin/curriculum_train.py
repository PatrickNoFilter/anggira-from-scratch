#!/usr/bin/env python3
"""Anggira V2 — Curriculum Burst Trainer

3-phase curriculum learning:
  Phase 1: Indonesian (2000 steps, lr=3e-4)
  Phase 2: Indonesian + English mix (2000 steps, lr=2e-4)  
  Phase 3: Indonesian + English + Code mix (2000 steps, lr=1e-4)

Each phase: burst of 12 steps, auto-resumes from checkpoint.
"""

import sys, os, json, pickle, time
import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'  # Optimal: threading overhead hurts at dim=144
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(42)

# ── V2 Config ──────────────────────────────────────────────────
DIM = 144
NUM_HEADS = 6
NUM_LAYERS = 5
MAX_SEQ_LEN = 512  # Model capacity - kept large for position embeddings
FF_DIM = 576
SEQ_LEN = 512  # Default, overridden per phase
BATCH_SIZE = 2
BURST = 12
LOG_EVERY = 3
LR_WARMUP = 50  # Short warmup for pretrained model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'indonesian_corpus_v2')
CURR_DIR = os.path.join(BASE_DIR, 'data', 'curriculum')
CURR_CHKPT = os.path.join(CURR_DIR, 'curriculum_checkpoint.pkl')
OLD_CHKPT = os.path.join(DATA_DIR, 'checkpoint.pkl')
CHKPT_FILE = CURR_CHKPT  # Active checkpoint path
TOK_PATH = os.path.join(DATA_DIR, 'tokenizer.pkl')
CONFIG_PATH = os.path.join(CURR_DIR, 'curriculum_config.json')

# ── Load curriculum config ──────────────────────────────────────
with open(CONFIG_PATH) as f:
    curriculum = json.load(f)

# ── Load BPE tokenizer ─────────────────────────────────────────
from anggira.bpe_tokenizer import BPETokenizer
tok = BPETokenizer()
tok.load(TOK_PATH)
VOCAB_SIZE = tok.vocab_size
print(f"🔤  BPE tokenizer: {VOCAB_SIZE} tokens loaded")

# ── Load tokenized corpora ──────────────────────────────────────
id_tokens = np.load(os.path.join(DATA_DIR, 'tokens.npy'))
print(f"📚  Indonesian tokens: {len(id_tokens):,}")

en_tokens_path = os.path.join(CURR_DIR, 'english_tokens.npy')
code_tokens_path = os.path.join(CURR_DIR, 'code_tokens.npy')

en_tokens = np.load(en_tokens_path) if os.path.exists(en_tokens_path) else None
code_tokens = np.load(code_tokens_path) if os.path.exists(code_tokens_path) else None

if en_tokens is not None:
    print(f"📚  English tokens: {len(en_tokens):,}")
if code_tokens is not None:
    print(f"📚  Code tokens: {len(code_tokens):,}")

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

model.training = True
total_steps = 0
best_loss = float('inf')
current_phase = 0

params_count = model.count_params()
print(f"🤖  Anggira V2: {params_count:,} params (vocab={VOCAB_SIZE}, dim={DIM}, L={NUM_LAYERS}, H={NUM_HEADS}, T={MAX_SEQ_LEN})")

# ── Load checkpoint if exists ──────────────────────────────────
loaded_optimizer_state = None
# Try curriculum checkpoint first, then old checkpoint
chkpt_to_load = None
if os.path.exists(CURR_CHKPT):
    chkpt_to_load = CURR_CHKPT
    print(f"📦  Loading curriculum checkpoint: {CURR_CHKPT}")
elif os.path.exists(OLD_CHKPT):
    chkpt_to_load = OLD_CHKPT
    print(f"📦  Loading original checkpoint: {OLD_CHKPT}")
    
if chkpt_to_load:
    cp = pickle.load(open(chkpt_to_load, 'rb'))
    
    model.embedding.token_embed[:] = cp['token_embed']
    old_pos = cp['pos_embed']
    new_pos = model.embedding.pos_embed
    if old_pos.shape[0] < new_pos.shape[0]:
        new_pos[:old_pos.shape[0]] = old_pos
        new_pos[old_pos.shape[0]:] = old_pos[-1:]
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
    current_phase = cp.get('phase', 0)
    loaded_optimizer_state = cp.get('optimizer')
    
    # If loading from old checkpoint (no 'phase' key), reset for curriculum
    if 'phase' not in cp:
        print(f"   ↩️  Old checkpoint detected — resetting step counter for curriculum")
        total_steps = 0
        current_phase = 0
        best_loss = float('inf')
        loaded_optimizer_state = None  # Don't carry over old optimizer momentum
        print(f"   💡  Weights preserved, optimizer & step reset for curriculum")
    
    print(f"💾  Resumed step {total_steps} (phase {current_phase}, best loss: {best_loss:.4f})")
else:
    print(f"🆕  Fresh curriculum training: {params_count:,} params")

# ── Determine current phase ─────────────────────────────────────
# Find which phase we're in based on total_steps
phases = curriculum['phases']
phase_idx = 0
steps_into_phase = total_steps
for i, phase in enumerate(phases):
    if total_steps < phase['target_steps']:
        phase_idx = i
        steps_into_phase = total_steps
        break
    else:
        steps_into_phase -= phase['target_steps']
        if i == len(phases) - 1:
            phase_idx = i
            steps_into_phase = total_steps

current_phase = phase_idx
phase = phases[current_phase]
lr = phase['lr']
phase_start_step = sum(p['target_steps'] for p in phases[:current_phase])
steps_in_phase = total_steps - phase_start_step
remaining = phase['target_steps'] - steps_in_phase

print(f"\n🎯  Phase {current_phase + 1}/{len(phases)}: {phase['name']}")
print(f"   Corpus: {phase['corpus']}")
print(f"   Learning rate: {lr}")
print(f"   Steps: {steps_in_phase}/{phase['target_steps']} ({remaining} remaining)")
print(f"   Mix ratio: {phase.get('mix_ratio', 1.0)}")

# ── Optimizer ──────────────────────────────────────────────────
from anggira.optimizer import AdamW

optimizer = AdamW(model.collect_params(), lr=lr, betas=(0.9, 0.999), weight_decay=0.1)
if loaded_optimizer_state is not None:
    optimizer.set_state(loaded_optimizer_state)
    print(f"   AdamW state restored (step={optimizer.t})")

# ── Corpus selection for current phase ──────────────────────────
def get_corpus_tokens(phase_config):
    """Get the token array for the current phase."""
    corpus = phase_config['corpus']
    if corpus == 'indonesian':
        return id_tokens
    elif corpus == 'indonesian_english_mix':
        # Mix Indonesian and English
        mix_ratio = phase_config.get('mix_ratio', 0.5)
        if en_tokens is not None:
            # Interleave: 50% Indonesian, 50% English
            return np.concatenate([id_tokens, en_tokens])
        return id_tokens
    elif corpus == 'indonesian_english_code_mix':
        # Mix all three
        mix_ratio = phase_config.get('mix_ratio', 0.33)
        if en_tokens is not None and code_tokens is not None:
            return np.concatenate([id_tokens, en_tokens, code_tokens])
        elif en_tokens is not None:
            return np.concatenate([id_tokens, en_tokens])
        return id_tokens
    return id_tokens

arr = get_corpus_tokens(phase)
print(f"📚  Training tokens: {len(arr):,}")

# ── Phase-aware context length ─────────────────────────────────
def get_phase_seq_len(phase_config):
    """Get the sequence length for the current phase from config."""
    return phase_config.get('max_seq_len', MAX_SEQ_LEN)

phase_seq_len = get_phase_seq_len(phase)
speedup_estimate = (MAX_SEQ_LEN / phase_seq_len) ** 2
print(f"📏  Phase context: T={phase_seq_len} (attention speedup: {speedup_estimate:.1f}x vs T={MAX_SEQ_LEN})")

# ── Training burst ─────────────────────────────────────────────
print(f"\n🏋️  Training burst of {BURST} steps (continues at step {total_steps})...")
t0 = time.time()
losses = []

for step in range(BURST):
    cur_step = total_steps + step

    # Check if we've completed current phase
    if cur_step >= sum(p['target_steps'] for p in phases[:current_phase + 1]):
        # Move to next phase
        if current_phase < len(phases) - 1:
            current_phase += 1
            phase = phases[current_phase]
            lr = phase['lr']
            arr = get_corpus_tokens(phase)
            phase_seq_len = get_phase_seq_len(phase)
            speedup_estimate = (MAX_SEQ_LEN / phase_seq_len) ** 2
            print(f"\n🎯  Transitioning to Phase {current_phase + 1}: {phase['name']}")
            print(f"   Corpus: {phase['corpus']}")
            print(f"   Learning rate: {lr}")
            print(f"   Mix ratio: {phase.get('mix_ratio', 1.0)}")
            print(f"   📏 Context: T={phase_seq_len} (attention speedup: {speedup_estimate:.1f}x)")

    # LR schedule: linear warmup only on fresh start, constant LR when pretrained
    if cur_step < LR_WARMUP and best_loss == float('inf'):
        optimizer.lr = lr * (cur_step + 1) / LR_WARMUP
    else:
        optimizer.lr = lr

    # Vectorized batch sampling — use phase-specific context length
    max_idx = max(1, len(arr) - phase_seq_len - 1)
    indices = np.random.randint(0, max_idx, size=BATCH_SIZE)
    inp = np.stack([arr[i:i + phase_seq_len] for i in indices], axis=0)
    tgt = np.stack([arr[i + 1:i + phase_seq_len + 1] for i in indices], axis=0)

    loss = model.train_step(inp, tgt, optimizer)
    losses.append(loss)
    if loss < best_loss:
        best_loss = loss

    if step % LOG_EVERY == 0 or step == BURST - 1 or step == 0:
        print(f"  Step {cur_step:5d} | loss={loss:.4f} | lr={optimizer.lr:.6f} | best={best_loss:.4f} | phase={current_phase+1}")

elapsed = time.time() - t0
print(f"✅  Burst done in {elapsed:.1f}s ({elapsed/BURST*1000:.0f}ms/step)")
print(f"   Loss: {losses[0]:.4f} → {losses[-1]:.4f}")

# ── Save checkpoint ──────────────────────────────────────────
total_steps += BURST
cp = {
    'step': total_steps,
    'best_loss': best_loss,
    'phase': current_phase,
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
print(f"💾  Curriculum checkpoint saved: step {total_steps}, phase {current_phase}")

# ── Generate sample ────────────────────────────────────────────
prompts = [
    "Indonesia adalah",
    "def hello",
    "machine learning",
    "import numpy",
    "The quick brown",
]

model.training = False
for prompt in prompts:
    pids = tok.encode(prompt, bos=True)
    if len(pids) > phase_seq_len - 10:
        pids = pids[:phase_seq_len - 10]
    
    out = model.generate(np.array(pids, dtype=np.int32), max_new=20, temp=0.9, top_k=30)
    
    gen = tok.decode(out)
    gen_clean = gen.replace('<BOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()
    print(f"\n🎯  \"{prompt}\" → \"{gen_clean}\"")

model.training = True

print(f"\n📋  Run again to continue curriculum training")
