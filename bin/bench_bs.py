"""Benchmark burst_train step times for different batch sizes."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from anggira.gpt import AnggiraGPT
from anggira.optimizer import AdamW

model = AnggiraGPT(vocab_size=2944, dim=64, num_heads=4, num_layers=3, max_seq_len=64, ff_dim=256)
arr = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tokens.npy'))

for bs in [1, 2, 4]:
    optimizer = AdamW(model.collect_params(), lr=3e-4, weight_decay=0.1)
    t0 = time.time()
    for step in range(20):
        inputs, targets = [], []
        for _ in range(bs):
            i = np.random.randint(0, len(arr) - 65)
            b = arr[i:i+65]
            inputs.append(b[:-1])
            targets.append(b[1:])
        inp = np.stack(inputs)
        tgt = np.stack(targets)
        _ = model.train_step(inp, tgt, optimizer)
    elapsed = time.time() - t0
    print(f'batch_size={bs}: {elapsed/20*1000:.0f}ms/step, {elapsed:.1f}s for 20 steps')
