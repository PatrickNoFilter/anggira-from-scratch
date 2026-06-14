#!/usr/bin/env python3
"""Anggira Grammar Evaluator — measures text quality metrics.

Usage:
  python3 bin/grammar_eval.py [--steps N]

Metrics:
  - Perplexity: How predictable text is (lower = better)
  - Word Validity: % of tokens that are real words
  - Structure Score: Proper punctuation, capitalization
  - Combined Grammar Score: Weighted average (0-100%)

Run after training to track progress.
"""

import sys, os, pickle, time
import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'indonesian_corpus_v2')
CHKPT_FILE = os.path.join(DATA_DIR, 'checkpoint.pkl')
TOK_PATH = os.path.join(DATA_DIR, 'tokenizer.pkl')
TOKENS_PATH = os.path.join(DATA_DIR, 'tokens.npy')

# ── Load Components ─────────────────────────────────────────────
from anggira.bpe_tokenizer import BPETokenizer
from anggira.gpt import AnggiraGPT

tok = BPETokenizer()
tok.load(TOK_PATH)
arr = np.load(TOKENS_PATH)

# ── Load Model ──────────────────────────────────────────────────
cp = pickle.load(open(CHKPT_FILE, 'rb'))
model = AnggiraGPT(
    vocab_size=tok.vocab_size, dim=144, num_heads=6,
    num_layers=5, max_seq_len=512, ff_dim=576, dropout_rate=0.0
)

# Load weights
model.embedding.token_embed[:] = cp['token_embed']
model.embedding.pos_embed[:] = cp['pos_embed']
for i in range(5):
    b = model.blocks[i]
    for sub, attrs in [('attn', ['W_q','W_k','W_v','W_o']),
                       ('ffn', ['W1','b1','W2','b2']),
                       ('ln1', ['gamma','beta']),
                       ('ln2', ['gamma','beta'])]:
        for a in attrs:
            key = f'b{i}_{sub}_{a}'
            if key in cp:
                getattr(getattr(b, sub), a)[:] = cp[key]
model.ln_f.gamma[:] = cp['ln_f_gamma']
model.ln_f.beta[:] = cp['ln_f_beta']
model.training = False

print(f"🤖 Model loaded: step {cp['step']}, loss {cp['best_loss']:.4f}")

# ── Build N-gram Statistics from Training Data ──────────────────
# Simple bigram model for perplexity calculation
print("📊 Building bigram stats from corpus...")
bigram_counts = {}
unigram_counts = {}

# Sample a subset for speed
sample_size = min(100000, len(arr) - 2)
sample_idx = np.random.choice(len(arr) - 2, sample_size, replace=False)

for idx in sample_idx:
    tok_id = int(arr[idx])
    next_id = int(arr[idx + 1])
    
    unigram_counts[tok_id] = unigram_counts.get(tok_id, 0) + 1
    bigram_key = (tok_id, next_id)
    bigram_counts[bigram_key] = bigram_counts.get(bigram_key, 0) + 1

total_bigrams = sum(bigram_counts.values())
vocab_size = tok.vocab_size

print(f"   Bigrams: {len(bigram_counts):,}, Unigrams: {len(unigram_counts):,}")


# ── Evaluation Functions ────────────────────────────────────────
def calculate_perplexity(text_tokens, smooth=1.0):
    """Calculate perplexity using bigram model with add-one smoothing."""
    log_prob = 0.0
    count = 0
    
    for i in range(1, len(text_tokens)):
        prev_id = int(text_tokens[i - 1])
        curr_id = int(text_tokens[i])
        
        bigram_count = bigram_counts.get((prev_id, curr_id), 0)
        unigram_count = unigram_counts.get(prev_id, 0)
        
        # Add-one smoothing
        prob = (bigram_count + smooth) / (unigram_count + smooth * vocab_size)
        log_prob += np.log2(prob + 1e-10)
        count += 1
    
    if count == 0:
        return float('inf')
    
    avg_log_prob = log_prob / count
    perplexity = 2 ** (-avg_log_prob)
    return perplexity


def calculate_word_validity(text_tokens):
    """Calculate % of tokens that are real words (not UNK/PAD/BOS/EOS)."""
    special_tokens = {0, 1, 2, 3}  # PAD, UNK, BOS, EOS
    
    valid_count = 0
    total_count = 0
    
    for tok_id in text_tokens:
        tok_id = int(tok_id)
        if tok_id not in special_tokens:
            total_count += 1
            # Check if it's a multi-byte token (likely a word piece)
            token_bytes = tok.vocab.get(tok_id, b'')
            if len(token_bytes) > 0 and token_bytes not in (b'<PAD>', b'<UNK>', b'<BOS>', b'<EOS>'):
                valid_count += 1
    
    return (valid_count / total_count * 100) if total_count > 0 else 0.0


def calculate_structure_score(text):
    """Score text structure: proper punctuation, capitalization, spacing."""
    score = 0.0
    checks = 0
    
    # Check 1: Starts with capital letter (for sentences)
    if text and text[0].isupper():
        score += 1
    checks += 1
    
    # Check 2: Has proper sentence ending
    if text and text[-1] in '.!?':
        score += 1
    checks += 1
    
    # Check 3: No double spaces
    if '  ' not in text:
        score += 1
    checks += 1
    
    # Check 4: Has spaces between words
    if ' ' in text:
        score += 1
    checks += 1
    
    # Check 5: Reasonable character mix (not all special chars)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio > 0.5:
        score += 1
    checks += 1
    
    # Check 6: No excessive repetition
    words = text.split()
    if len(words) > 1:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio > 0.5:
            score += 1
    checks += 1
    
    return (score / checks * 100) if checks > 0 else 0.0


def calculate_combined_score(perplexity, word_validity, structure_score):
    """Combined grammar score (0-100%). Lower perplexity = higher score."""
    # Normalize perplexity (assume 1000 = worst, 1 = best)
    ppl_score = max(0, 100 - (perplexity - 1) * 100 / 999)
    ppl_score = min(100, max(0, ppl_score))
    
    # Weighted average
    combined = (ppl_score * 0.4 + word_validity * 0.3 + structure_score * 0.3)
    return combined


# ── Test Prompts ────────────────────────────────────────────────
TEST_PROMPTS = [
    "Indonesia adalah",
    "def hello():",
    "machine learning adalah",
    "import numpy as np",
    "Python adalah bahasa",
    "belajar AI",
    "Prabowo Subianto adalah",
]


# ── Main Evaluation ─────────────────────────────────────────────
def run_evaluation(num_samples=5):
    """Run full evaluation and return metrics."""
    print("\n" + "=" * 60)
    print("📝 GRAMMAR EVALUATION")
    print("=" * 60)
    
    all_perplexities = []
    all_validities = []
    all_structures = []
    
    for prompt in TEST_PROMPTS[:num_samples]:
        pids = tok.encode(prompt, bos=True)
        if len(pids) > 500:
            pids = pids[:500]
        
        out = model.generate(
            np.array(pids, dtype=np.int32),
            max_new=40, temp=0.7, top_k=30
        )
        
        gen = tok.decode(out).replace('<BOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()
        
        # Calculate metrics
        ppl = calculate_perplexity(out)
        validity = calculate_word_validity(out)
        structure = calculate_structure_score(gen)
        
        all_perplexities.append(ppl)
        all_validities.append(validity)
        all_structures.append(structure)
        
        print(f"\n \"{prompt}\"")
        print(f"    Output: \"{gen[:60]}...\"")
        print(f"    Perplexity: {ppl:.1f} | Validity: {validity:.1f}% | Structure: {structure:.1f}%")
    
    # Average metrics
    avg_ppl = np.mean(all_perplexities)
    avg_validity = np.mean(all_validities)
    avg_structure = np.mean(all_structures)
    combined = calculate_combined_score(avg_ppl, avg_validity, avg_structure)
    
    print("\n" + "-" * 60)
    print("📊 SUMMARY")
    print("-" * 60)
    print(f"  Step:            {cp['step']}")
    print(f"  Training Loss:   {cp['best_loss']:.4f}")
    print(f"  ─────────────────────────────────")
    print(f"  Avg Perplexity:  {avg_ppl:.1f}")
    print(f"  Avg Validity:    {avg_validity:.1f}%")
    print(f"  Avg Structure:   {avg_structure:.1f}%")
    print(f"  ─────────────────────────────────")
    print(f"  📈 GRAMMAR SCORE: {combined:.1f}%")
    print("=" * 60)
    
    return {
        'step': cp['step'],
        'loss': cp['best_loss'],
        'perplexity': avg_ppl,
        'validity': avg_validity,
        'structure': avg_structure,
        'grammar_score': combined
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5, help='Number of prompts to test')
    args = parser.parse_args()
    
    metrics = run_evaluation(num_samples=args.steps)
    
    # Save metrics history
    metrics_file = os.path.join(DATA_DIR, 'grammar_metrics.json')
    history = []
    if os.path.exists(metrics_file):
        import json
        with open(metrics_file, 'r') as f:
            history = json.load(f)
    
    import json
    history.append({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        **metrics
    })
    
    with open(metrics_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n💾 Metrics saved to {metrics_file}")
