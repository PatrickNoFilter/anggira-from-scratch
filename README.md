# 🧠 Anggira AI — Built From Scratch

A complete language model implementation built entirely in **Python/NumPy** — no PyTorch, no TensorFlow, no HuggingFace. Every algorithm implemented from first principles.

## Architecture

- **Model**: GPT-style Transformer (decoder-only)
- **Tokenizer**: BPE (Byte Pair Encoding) with 1,178 vocab
- **Dimensions**: dim=144, 5 blocks, 8 heads, context=512
- **Parameters**: ~1.49M
- **Curriculum**: 3-phase training (Indonesian → English → Code)

## Modules

| Module | Description |
|--------|-------------|
| `anggira/core.py` | Tensor ops — matmul, softmax, embedding lookup |
| `anggira/nn.py` | Neural network layers — Linear, Embedding, LayerNorm |
| `anggira/transformer.py` | Multi-head attention, Transformer blocks |
| `anggira/autodiff.py` | Automatic differentiation engine |
| `anggira/optimizer.py` | SGD, Adam, AdamW with weight decay |
| `anggira/bpe_tokenizer.py` | BPE tokenizer training + encode/decode |
| `anggira/indonesian_tokenizer.py` | Indonesian-aware tokenizer |
| `anggira/gpt.py` / `gpt_v2.py` | GPT model implementations |
| `anggira/losses.py` | Cross-entropy, perplexity |
| `anggira/grammar_eval.py` | Grammar scoring and evaluation |

## Training

```bash
# Curriculum training (recommended)
python bin/curriculum_train.py

# On ARM big cores for performance
taskset -c 6,7 python3 -u bin/curriculum_train.py
```

Training runs at ~14 steps/min on Exynos 1280 (4.23s/step).

### Curriculum Phases

1. **Indonesian** — Base language comprehension (125K tokens)
2. **English** — Cross-lingual transfer (27K tokens)  
3. **Code** — Programming understanding (16K tokens)

Each phase loads from the previous checkpoint, building knowledge incrementally.

## Quick Start

```bash
pip install numpy

# Train
python bin/curriculum_train.py

# Generate text
python bin/chat.py
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
├── bin/                  # Scripts and entry points (19 scripts)
│   ├── curriculum_train.py  # Main training (3-phase)
│   ├── burst_train.py       # Burst training
│   ├── train.py             # Basic training
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
