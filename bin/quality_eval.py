#!/usr/bin/env python3
"""Anggira Multi-Domain Quality Evaluator — Indonesian, English, Code.

Usage:
  python3 bin/quality_eval.py

Tests:
  - Indonesian: grammar, vocabulary, sentence structure
  - English: grammar, vocabulary, sentence structure  
  - Code: syntax, imports, function definitions, keywords
"""

import sys, os, pickle, json, time
import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'indonesian_corpus_v2')
CHKPT_FILE = os.path.join(DATA_DIR, 'checkpoint.pkl')
TOK_PATH = os.path.join(DATA_DIR, 'tokenizer.pkl')
TOKENS_PATH=os.path.join(DATA_DIR, 'tokens.npy')

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

# ── Build N-gram Stats ──────────────────────────────────────────
print("📊 Building language models...")
bigram_counts = {}
unigram_counts = {}
sample_size = min(100000, len(arr) - 2)
sample_idx = np.random.choice(len(arr) - 2, sample_size, replace=False)
for idx in sample_idx:
    t1, t2 = int(arr[idx]), int(arr[idx + 1])
    unigram_counts[t1] = unigram_counts.get(t1, 0) + 1
    bigram_counts[(t1, t2)] = bigram_counts.get((t1, t2), 0) + 1
vocab_size = tok.vocab_size


# ── Domain-Specific Test Prompts ────────────────────────────────
PROMPTS = {
    'indonesian': [
        "Indonesia adalah negara",
        "Bahasa Indonesia sangat",
        "Jakarta merupakan ibu kota",
        "Presiden Indonesia",
        "Untuk mempelajari",
        "Ekonomi Indonesia",
        "Budaya Indonesia",
        "Pendidikan di Indonesia",
    ],
    'english': [
        "The quick brown fox",
        "Machine learning is",
        "Python is a programming",
        "Artificial intelligence will",
        "The capital of Indonesia",
        "Natural language processing",
        "Deep learning models",
        "Data science involves",
    ],
    'code': [
        "def hello_world():",
        "import numpy as np",
        "class NeuralNetwork:",
        "for i in range(",
        "if __name__ ==",
        "return model.predict",
        "print('Hello, World!')",
        "with open('data.txt'",
    ],
}


# ── Evaluation Functions ────────────────────────────────────────
def calculate_perplexity(tokens):
    """Calculate perplexity using bigram model."""
    log_prob = 0.0
    count = 0
    smooth = 1.0
    for i in range(1, len(tokens)):
        prev_id, curr_id = int(tokens[i - 1]), int(tokens[i])
        bg = bigram_counts.get((prev_id, curr_id), 0)
        ug = unigram_counts.get(prev_id, 0)
        prob = (bg + smooth) / (ug + smooth * vocab_size)
        log_prob += np.log2(prob + 1e-10)
        count += 1
    return 2 ** (-log_prob / max(count, 1))


def evaluate_indonesian(text, tokens):
    """Evaluate Indonesian text quality."""
    score = 0.0
    checks = 0
    
    # 1. Word validity
    special = {0, 1, 2, 3}
    valid = sum(1 for t in tokens if int(t) not in special)
    validity = valid / max(len(tokens), 1) * 100
    score += validity / 100
    checks += 1
    
    # 2. Common Indonesian words
    indo_words = ['adalah', 'dan', 'di', 'yang', 'ini', 'itu', 'untuk', 'dengan',
                  'pada', 'dari', 'akan', 'tidak', 'juga', 'atau', 'oleh']
    text_lower = text.lower()
    found = sum(1 for w in indo_words if w in text_lower)
    word_score = min(1.0, found / 3)  # Expect at least 3 common words
    score += word_score
    checks += 1
    
    # 3. Sentence structure
    has_space = ' ' in text
    has_capital = text[0].isupper() if text else False
    no_double_space = '  ' not in text
    struct = (has_space + has_capital + no_double_space) / 3
    score += struct
    checks += 1
    
    # 4. Character ratio (Indonesian has specific patterns)
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    char_score = 1.0 if alpha_ratio > 0.6 else alpha_ratio
    score += char_score
    checks += 1
    
    return (score / checks) * 100


def evaluate_english(text, tokens):
    """Evaluate English text quality."""
    score = 0.0
    checks = 0
    
    # 1. Word validity
    special = {0, 1, 2, 3}
    valid = sum(1 for t in tokens if int(t) not in special)
    validity = valid / max(len(tokens), 1) * 100
    score += validity / 100
    checks += 1
    
    # 2. Common English words
    eng_words = ['the', 'is', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'for',
                 'with', 'on', 'at', 'by', 'from', 'as', 'this', 'that']
    text_lower = text.lower()
    found = sum(1 for w in eng_words if f' {w} ' in f' {text_lower} ')
    word_score = min(1.0, found / 3)
    score += word_score
    checks += 1
    
    # 3. Sentence structure
    has_space = ' ' in text
    has_capital = text[0].isupper() if text else False
    no_double_space = '  ' not in text
    struct = (has_space + has_capital + no_double_space) / 3
    score += struct
    checks += 1
    
    # 4. Proper spacing
    words = text.split()
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    word_len_score = 1.0 if 2 < avg_word_len < 10 else 0.5
    score += word_len_score
    checks += 1
    
    return (score / checks) * 100


def evaluate_code(text, tokens):
    """Evaluate code fluency."""
    score = 0.0
    checks = 0
    
    # 1. Code keywords present
    code_keywords = ['def', 'class', 'import', 'for', 'if', 'return', 'print',
                     'while', 'try', 'except', 'with', 'as', 'in', 'range',
                     'self', '__init__', '__name__', '__main__']
    found = sum(1 for kw in code_keywords if kw in text)
    kw_score = min(1.0, found / 3)
    score += kw_score
    checks += 1
    
    # 2. Syntax characters
    syntax_chars = ['(', ')', ':', '=', "'", '"', '[', ']', '{', '}']
    found_syntax = sum(1 for c in syntax_chars if c in text)
    syntax_score = min(1.0, found_syntax / 3)
    score += syntax_score
    checks += 1
    
    # 3. Indentation (spaces before keywords)
    lines = text.split('\n')
    has_indent = any(line.startswith('    ') or line.startswith('\t') for line in lines if line.strip())
    indent_score = 1.0 if has_indent or len(lines) == 1 else 0.5
    score += indent_score
    checks += 1
    
    # 4. Function/class structure
    has_def = 'def ' in text
    has_class = 'class ' in text
    has_import = 'import ' in text
    struct_score = (has_def + has_class + has_import) / 3
    score += struct_score
    checks += 1
    
    # 5. Proper parentheses matching
    open_p = text.count('(')
    close_p = text.count(')')
    paren_score = 1.0 if open_p == close_p else 0.5
    score += paren_score
    checks += 1
    
    return (score / checks) * 100


# ── Main Evaluation ─────────────────────────────────────────────
def run_evaluation():
    print("\n" + "=" * 70)
    print("📝 MULTI-DOMAIN QUALITY EVALUATION")
    print("=" * 70)
    
    results = {}
    
    for domain, prompts in PROMPTS.items():
        print(f"\n{'─' * 70}")
        print(f"🏷️  {domain.upper()} DOMAIN")
        print(f"{'─' * 70}")
        
        scores = []
        perplexities = []
        
        for prompt in prompts:
            pids = tok.encode(prompt, bos=True)
            if len(pids) > 500:
                pids = pids[:500]
            
            out = model.generate(
                np.array(pids, dtype=np.int32),
                max_new=40, temp=0.7, top_k=30
            )
            
            gen = tok.decode(out).replace('<BOS>', '').replace('<EOS>', '').replace('<PAD>', '').strip()
            
            # Calculate perplexity
            ppl = calculate_perplexity(out)
            perplexities.append(ppl)
            
            # Domain-specific evaluation
            if domain == 'indonesian':
                quality = evaluate_indonesian(gen, out)
            elif domain == 'english':
                quality = evaluate_english(gen, out)
            else:
                quality = evaluate_code(gen, out)
            
            scores.append(quality)
            
            print(f"  \"{prompt}\"")
            print(f"    → \"{gen[:50]}...\"")
            print(f"    Quality: {quality:.1f}% | PPL: {ppl:.0f}")
        
        avg_score = np.mean(scores)
        avg_ppl = np.mean(perplexities)
        results[domain] = {'score': avg_score, 'perplexity': avg_ppl}
        
        print(f"\n  📊 {domain.upper()} Average: {avg_score:.1f}% (PPL: {avg_ppl:.0f})")
    
    # ── Summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Step: {cp['step']} | Loss: {cp['best_loss']:.4f}")
    print(f"  {'─' * 50}")
    
    total_score = 0
    for domain, metrics in results.items():
        emoji = '🇮🇩' if domain == 'indonesian' else ('🇺🇸' if domain == 'english' else '💻')
        print(f"  {emoji} {domain.upper():12} → {metrics['score']:.1f}% (PPL: {metrics['perplexity']:.0f})")
        total_score += metrics['score']
    
    overall = total_score / len(results)
    print(f"  {'─' * 50}")
    print(f"  📈 OVERALL SCORE: {overall:.1f}%")
    print("=" * 70)
    
    return {
        'step': cp['step'],
        'loss': cp['best_loss'],
        'overall': overall,
        **{f"{k}_score": v['score'] for k, v in results.items()},
        **{f"{k}_ppl": v['perplexity'] for k, v in results.items()},
    }


if __name__ == '__main__':
    metrics = run_evaluation()
    
    # Save to history
    metrics_file = os.path.join(DATA_DIR, 'quality_metrics.json')
    history = []
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            history = json.load(f)
    
    history.append({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        **metrics
    })
    
    with open(metrics_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n💾 Metrics saved to {metrics_file}")
