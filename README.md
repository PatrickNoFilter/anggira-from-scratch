# 🧠 Anggira AI — Built From Scratch

A complete language model implementation built entirely in **Python/NumPy** — no PyTorch, no TensorFlow, no HuggingFace. Every algorithm implemented from first principles.

## Architecture

- **Model**: GPT-style Transformer (decoder-only)
- **Tokenizer**: BPE (Byte Pair Encoding) with 1,178 vocab
- **Dimensions**: dim=144, 5 blocks, 6 heads, context=512 (max capacity)
- **Parameters**: ~1.49M
- **Curriculum**: 3-phase training with phase-specific context lengths

### Phase-Specific Context Lengths

| Phase | Context | Attention Speedup | Purpose |
|-------|---------|-------------------|---------|
| Indonesian | T=64 | 64× | Morphology + word order |
| English | T=128 | 16× | Cross-lingual transfer |
| Code | T=256 | 4× | Structure + indentation |

## Modules

| Module | Description |
|--------|-------------|
| `anggira/core.py` | Tensor ops — matmul, softmax, embedding lookup |
| `anggira/nn.py` | Neural network layers — Linear, Embedding, LayerNorm |
| `anggira/transformer.py` | Multi-head attention, Transformer blocks |
| `anggira/autodiff.py` | Automatic differentiation engine (educational reference) |
| `anggira/optimizer.py` | SGD, Adam, AdamW with weight decay |
| `anggira/bpe_tokenizer.py` | BPE tokenizer training + encode/decode |
| `anggira/gpt.py` | GPT model with manual backward passes |
| `anggira/losses.py` | Cross-entropy, perplexity |

## Training

```bash
# Curriculum training (recommended)
python bin/curriculum_train.py

# On ARM big cores for performance
taskset -c 6,7 python3 -u bin/curriculum_train.py
```

### Performance

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Steps/min | ~14 | **~113** | **8×** |
| Time/step | 4.23s | **0.53s** | **8×** |
| Phase 1 ETA | ~2.4 hours | **~18 minutes** | — |

### Optimization Details

1. **Phase-specific context lengths** — Attention is O(T²), so reducing T from 512 to 64 gives 64× speedup on attention computation
2. **OMP_NUM_THREADS=1** — Optimal for dim=144 (threading overhead hurts at small matrix sizes)
3. **Manual backward passes** — Hand-coded gradients per layer (faster than autodiff engine)

### Curriculum Phases

1. **Indonesian** — Base language comprehension (583K tokens, T=64)
2. **English** — Cross-lingual transfer (27K tokens, T=128)
3. **Code** — Programming understanding (16K tokens, T=256)

Each phase loads from the previous checkpoint, building knowledge incrementally.

## Quick Start

```bash
pip install numpy

# Train
python bin/curriculum_train.py

# Generate text
python bin/chat.py

# Benchmark threading
python bin/bench_omp.py
```

## Project Structure

```
anggira/
├── anggira/              # Core library (35 modules)
│   ├── core.py              # Tensor operations
│   ├── nn.py                # Neural network layers
│   ├── transformer.py       # Transformer architecture
│   ├── autodiff.py          # Automatic differentiation
│   ├── optimizer.py         # AdamW optimizer
│   ├── gpt.py               # GPT model
│   ├── bpe_tokenizer.py     # BPE tokenizer
│   └── ...
├── bin/                  # Scripts and entry points
│   ├── curriculum_train.py  # Main training (3-phase, optimized)
│   ├── bench_omp.py         # OpenBLAS threading benchmark
│   ├── burst_train.py       # Burst training
│   ├── chat.py              # Interactive chat
│   └── grammar_eval.py      # Quality evaluation
├── data/                 # Training data & checkpoints (gitignored)
│   ├── curriculum/          # Curriculum data & checkpoints
│   └── indonesian_corpus_v2/ # Indonesian corpus
└── tests/                # Unit tests
```

## Philosophy

> Before using a framework, understand what it does.

Every gradient, every matrix multiplication, every attention score is computed with raw NumPy. This project exists to learn, not to ship.

## License

MIT
