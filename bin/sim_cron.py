"""Simulate 3 cron bursts back-to-back with 60s gaps."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from anggira.gpt import AnggiraGPT
from anggira.optimizer import AdamW

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
model = AnggiraGPT(vocab_size=2944, dim=64, num_heads=4, num_layers=3, max_seq_len=64, ff_dim=256)

import pickle
cp = pickle.load(open(os.path.join(DATA_DIR, 'checkpoint.pkl'), 'rb'))
model.embedding.token_embed[:] = cp['token_embed']
model.embedding.pos_embed[:] = cp['pos_embed']
for i in range(3):
    b = model.blocks[i]
    for sub, attrs in [('attn',['W_q','W_k','W_v','W_o']),('ffn',['W1','b1','W2','b2']),('ln1',['gamma','beta']),('ln2',['gamma','beta'])]:
        for a in attrs:
            getattr(getattr(b, sub), a)[:] = cp[f'b{i}_{sub}_{a}']
model.ln_f.gamma[:] = cp['ln_f_gamma']
model.ln_f.beta[:] = cp['ln_f_beta']

arr = np.load(os.path.join(DATA_DIR, 'tokens.npy'))

for burst_idx in range(3):
    optimizer = AdamW(model.collect_params(), lr=3e-4, weight_decay=0.1)
    if 'optimizer' in cp:
        optimizer.set_state(cp['optimizer'])
    
    t0 = time.time()
    for step in range(100):
        inputs, targets = [], []
        for _ in range(2):
            i = np.random.randint(0, len(arr) - 65)
            b = arr[i:i+65]
            inputs.append(b[:-1])
            targets.append(b[1:])
        inp = np.stack(inputs)
        tgt = np.stack(targets)
        _ = model.train_step(inp, tgt, optimizer)
    elapsed = time.time() - t0
    print(f'Burst {burst_idx+1}: {elapsed:.1f}s ({elapsed/100*1000:.0f}ms/step)')
    
    # Save checkpoint
    cp = {
        'step': 14900 + (burst_idx + 1) * 100,
        'best_loss': 3.1838,
        'token_embed': model.embedding.token_embed,
        'pos_embed': model.embedding.pos_embed,
        'optimizer': optimizer.get_state(),
    }
    for i, block in enumerate(model.blocks):
        for sub, attrs in [('attn',['W_q','W_k','W_v','W_o']),('ffn',['W1','b1','W2','b2']),('ln1',['gamma','beta']),('ln2',['gamma','beta'])]:
            for a in attrs:
                cp[f'b{i}_{sub}_{a}'] = getattr(getattr(block, sub), a)
    cp['ln_f_gamma'] = model.ln_f.gamma
    cp['ln_f_beta'] = model.ln_f.beta
    pickle.dump(cp, open(os.path.join(DATA_DIR, 'checkpoint.pkl'), 'wb'))
    if burst_idx < 2:
        print(f'  Cooling 30s...')
        time.sleep(30)
