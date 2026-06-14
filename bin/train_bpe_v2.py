"""Fast BPE: train on small sample, tokenize full corpus."""
import sys, os, random, pickle
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

DATA_DIR = os.path.join(BASE, 'data', 'indonesian_corpus_v2')
corpus_path = os.path.join(DATA_DIR, 'indonesian_corpus.txt')

from anggira.bpe_tokenizer import BPETokenizer

# Load corpus
with open(corpus_path, 'r', encoding='utf-8') as f:
    text = f.read()

lines = [l for l in text.split('\n') if len(l.strip()) > 0]
print(f"Corpus: {len(text):,} chars, {len(lines):,} lines")

# Train BPE on a TINY sample for speed (pure Python is slow)
SAMPLE_SIZE = min(300, len(lines))
random.seed(42)
sample_lines = random.sample(lines, SAMPLE_SIZE)
# Use shorter lines only to speed up
sample_lines.sort(key=len)
sample_lines = sample_lines[:200]
print(f"Training BPE on {len(sample_lines):,} lines (avg {sum(len(l) for l in sample_lines)//len(sample_lines)} chars)...")

tok = BPETokenizer()
VOCAB_TARGET = 1500  # small but beats char-level
tok.train(sample_lines, vocab_size=VOCAB_TARGET, verbose=True)

# Save tokenizer
tok_path = os.path.join(DATA_DIR, 'tokenizer.pkl')
tok.save(tok_path)
print(f"Tokenizer saved: {tok_path} (vocab={tok.vocab_size})")

# Tokenize FULL corpus
print("Tokenizing full corpus (this is the fast part)...")
all_ids = []
doc_count = 0
batch_size = 500
for i in range(0, len(lines), batch_size):
    batch = lines[i:i+batch_size]
    for line in batch:
        ids = tok.encode(line, bos=True, eos=True)
        all_ids.extend(ids)
        doc_count += 1
    if (i // batch_size) % 5 == 0:
        print(f"  ... {doc_count}/{len(lines)} docs, {len(all_ids):,} tokens")

tokens_arr = np.array(all_ids, dtype=np.int32)
token_path = os.path.join(DATA_DIR, 'tokens.npy')
np.save(token_path, tokens_arr)
print(f"\nTokens generated: {len(tokens_arr):,}")
print(f"Docs: {doc_count:,}")
print(f"Vocab used: {len(np.unique(tokens_arr)):,}/{tok.vocab_size}")
print(f"UNK: {np.sum(tokens_arr == 1):,} ({np.sum(tokens_arr == 1)/len(tokens_arr)*100:.1f}%)")

# Quick test
tests = [
    "import numpy as np",
    "def hello(): pass",
    "Indonesia adalah negara",
    "machine learning is fun",
]
for t in tests:
    ids = tok.encode(t, bos=True, eos=True)
    print(f"  '{t[:40]}' -> {len(ids)} ids")

# Save info
with open(os.path.join(DATA_DIR, 'vocab_info.txt'), 'w') as f:
    f.write(f"Corpus: {len(text):,} chars\n")
    f.write(f"Lines: {len(lines):,}\n")
    f.write(f"Tokens: {len(tokens_arr):,}\n")
    f.write(f"Vocab: {tok.vocab_size}\n")
    f.write(f"Merges: {len(tok.merges)}\n")
    f.write(f"UNK: {np.sum(tokens_arr == 1):,} ({np.sum(tokens_arr == 1)/len(tokens_arr)*100:.1f}%)\n")

print(f"\n✅ Done! Data ready for burst_train_v2.py")
