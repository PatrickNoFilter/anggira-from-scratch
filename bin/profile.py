"""Profile where the time goes in the new training step."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from anggira.optimizer import AdamW
from anggira.gpt import AnggiraGPT, gelu, gelu_grad

model = AnggiraGPT(vocab_size=2944, dim=64, num_heads=4, num_layers=3, max_seq_len=64, ff_dim=256)
arr = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'tokens.npy'))

inp = arr[np.random.randint(0, len(arr)-65):-1][:64].reshape(1, 64)
tgt = arr[np.random.randint(0, len(arr)-65)+1:][:64].reshape(1, 64)

# Warmup
for _ in range(3):
    _ = model.forward(inp)

# Profile forward
t0 = time.time()
for _ in range(50):
    logits = model.forward(inp)
print(f"  Forward (50x): {(time.time()-t0)/50*1000:.0f}ms")

# Profile backward with weight tying
B, T, V = logits.shape
logits_flat = logits.reshape(-1, V)
tgt_flat = tgt.reshape(-1)
max_l = logits_flat.max(axis=-1, keepdims=True)
dlogits = np.exp(logits_flat - max_l)
dlogits = dlogits / dlogits.sum(axis=-1, keepdims=True)
dlogits[np.arange(len(tgt_flat)), tgt_flat] -= 1.0
dlogits = dlogits.reshape(B, T, V) / (B * T)

t0 = time.time()
for _ in range(50):
    model.backward(dlogits, inp)
print(f"  Backward (50x): {(time.time()-t0)/50*1000:.0f}ms")

# Profile optimizer step
optimizer = AdamW(model.collect_params(), lr=3e-4, weight_decay=0.1)
t0 = time.time()
for _ in range(50):
    optimizer.step()
print(f"  Optimizer.step (50x): {(time.time()-t0)/50*1000:.0f}ms")

# Profile GELU vs ReLU
x = np.random.randn(4, 64, 256)
t0 = time.time()
for _ in range(500):
    _ = np.maximum(0, x)
print(f"  ReLU (500x): {(time.time()-t0)/500*1000:.0f}ms")

t0 = time.time()
for _ in range(500):
    _ = gelu(x)
print(f"  GELU forward (500x): {(time.time()-t0)/500*1000:.0f}ms")

t0 = time.time()
for _ in range(500):
    _ = gelu_grad(x)
print(f"  GELU backward (500x): {(time.time()-t0)/500*1000:.0f}ms")

# Profile dropout
from anggira.gpt import Dropout
dp = Dropout(0.1)
t0 = time.time()
for _ in range(500):
    _ = dp.forward(x, True)
print(f"  Dropout forward (500x): {(time.time()-t0)/500*1000:.0f}ms")
