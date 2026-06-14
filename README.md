# 🧠 Anggira AI

A from-scratch neural network framework with a bilingual (Indonesian + English) GPT model, trained on mobile hardware.

## Architecture

- **Framework**: Pure Python + NumPy (no PyTorch/TensorFlow)
- **Model**: GPT-style Transformer (1.49M params)
  - Embedding dim: 144, Layers: 5, Attention heads: 6
  - Max sequence length: 512
  - BPE tokenizer: 1,178 vocab (Indonesian, English, Code)
- **Training**: Burst training on ARM big.LITTLE (Samsung Exynos 1280)
  - Pinned to Cortex-A78 big cores for ~14 steps/min
  - Single-threaded NumPy for mobile safety

## Curriculum Learning

The model uses 3-phase curriculum training:

1. **Indonesian** — Foundation language (lr=3e-4)
2. **+ English** — Cross-lingual transfer (lr=2e-4, 50% mix)
3. **+ Code** — Programming patterns (lr=1e-4, 33% mix)

## Project Structure

```
anggira/
├── anggira/              # Core library
│   ├── core.py           # Linear algebra primitives (matmul, softmax, etc.)
│   ├── gpt.py            # GPT model (transformer, attention, embeddings)
│   ├── bpe_tokenizer.py  # Byte-Pair Encoding tokenizer
│   ├── optimizer.py      # AdamW optimizer
│   ├── ml.py             # Training utilities
│   ├── nlp.py            # NLP components
│   ├── nn.py             # Neural network layers
│   └── ...               # More modules (RL, vision, audio, etc.)
├── bin/                  # Training scripts
│   ├── curriculum_train.py   # Curriculum learning trainer
│   ├── burst_train_v2.py     # Burst training script
│   └── grammar_eval.py       # Grammar evaluation
├── data/                 # Training data (gitignored)
│   ├── curriculum/       # Curriculum corpora & checkpoints
│   └── indonesian_corpus_v2/ # Main corpus
└── README.md
```

## Quick Start

```bash
# Run curriculum training
python3 bin/curriculum_train.py

# Or with CPU pinning (ARM big.LITTLE)
taskset -c 6,7 python3 bin/curriculum_train.py
```

## Hardware

Tested on Samsung Exynos 1280 (ARM big.LITTLE):
- Cores 0-5: Cortex-A55 @ 2.0GHz (efficiency)
- Cores 6-7: Cortex-A78 @ 2.4GHz (performance) ← training runs here

## License

MIT
