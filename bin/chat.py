"""Chat with trained AnggiraGPT."""
import sys, os, pickle, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anggira.gpt import AnggiraGPT

# ── Config (match training) ─────────────────────────────────────
VOCAB_SIZE = 2944
DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 3
MAX_SEQ_LEN = 64
FF_DIM = 256
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

def decode(ids):
    return ' '.join(i2w.get(i, '<UNK>') for i in ids)

# Build model
print("🤖 Loading AnggiraGPT...")
model = AnggiraGPT(
    vocab_size=VOCAB_SIZE,
    dim=DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    ff_dim=FF_DIM,
)

# Load weights
weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data', 'gpt_weights.npz')
if os.path.exists(weights_path):
    w = np.load(weights_path)
    model.embedding.token_embed = w['token_embed']
    model.embedding.pos_embed = w['pos_embed']
    for i in range(NUM_LAYERS):
        block = model.blocks[i]
        block.attn.W_q = w[f'b{i}_attn_W_q']
        block.attn.W_k = w[f'b{i}_attn_W_k']
        block.attn.W_v = w[f'b{i}_attn_W_v']
        block.attn.W_o = w[f'b{i}_attn_W_o']
        block.ffn.W1 = w[f'b{i}_ffn_W1']
        block.ffn.b1 = w[f'b{i}_ffn_b1']
        block.ffn.W2 = w[f'b{i}_ffn_W2']
        block.ffn.b2 = w[f'b{i}_ffn_b2']
        block.ln1.gamma = w[f'b{i}_ln1_gamma']
        block.ln1.beta = w[f'b{i}_ln1_beta']
        block.ln2.gamma = w[f'b{i}_ln2_gamma']
        block.ln2.beta = w[f'b{i}_ln2_beta']
    model.ln_f.gamma = w['ln_f_gamma']
    model.ln_f.beta = w['ln_f_beta']
    print(f"✅ Loaded trained weights ({len(w)} arrays)")
else:
    print(f"⚠️  No weights found at {weights_path}")
    print("   Using untrained model (run bin/train_fast.py first)")

print(f"   Params: {model.count_params():,}")
print()

# Interactive chat loop
print("=" * 50)
print("💬 Anggira Chat (type 'quit' to exit)")
print("   Tips: lowercase short prompts work best")
print("   Try: 'once upon a time', 'lily', 'the dog'")
print("=" * 50)

while True:
    try:
        prompt = input("\nYou: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break
    
    if not prompt:
        continue
    if prompt.lower() in ('quit', 'exit', 'q'):
        print("Bye!")
        break
    
    # Tokenize
    pids = [2] + encode(prompt)[:32]  # cap prompt length
    
    # Generate
    out = model.generate(pids, max_new=40, temp=0.85, top_k=30)
    response = decode(out[len(pids):])
    
    # Clean up
    if '<EOS>' in response:
        response = response.split('<EOS>')[0] + '.'
    
    print(f"🤖: {response}")
