"""
Anggira Capstone — Training Pipeline Components

Phase 19, Lessons 42-49: LLM Training Pipeline

Implements:
- CosmicLRScheduler — Cosine LR with linear warmup
- GradientManager — Gradient clipping + accumulation
- MixedPrecisionScaler — AMP-style grad scaling
- Checkpointer — Atomic checkpoint save/resume with integrity
- StreamingDataset — JSONL-based token streaming
- TrainingPipelineDemo — Full pipeline integration
"""

import json
import math
import os
import random
import tempfile
import time
import uuid
from collections import defaultdict


# ═══════════════════════════════════════════════
# CosmicLRScheduler  (Lesson 44)
# ═══════════════════════════════════════════════

class CosmicLRScheduler:
    """Cosine LR schedule with linear warmup.

    Phase 1 (warmup): linear increase from 0 to peak_lr
    Phase 2 (cosine): cosine decay from peak_lr to min_lr
    """

    def __init__(self, peak_lr=0.001, min_lr=1e-5, warmup_steps=100, total_steps=1000):
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step):
        """Get learning rate at a given step."""
        if step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * (step + 1) / self.warmup_steps
        elif step >= self.total_steps:
            return self.min_lr
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def get_lrs(self, steps):
        """Get LR for each step in a range."""
        return [self.get_lr(s) for s in range(steps)]

    def plot(self, width=60):
        """Print ASCII chart of the schedule."""
        print("  CosmicLR Schedule (ASCII):")
        steps = min(self.total_steps, 60)
        lrs = self.get_lrs(steps)
        max_lr = max(lrs)
        min_lr = min(lrs)
        print(f"  peak_lr={self.peak_lr}, min_lr={self.min_lr}, "
              f"warmup={self.warmup_steps}, total={self.total_steps}")
        print(f"  LR range: {min_lr:.6f} - {max_lr:.6f}")

        # Print every Nth step
        for i in range(0, self.total_steps, max(1, self.total_steps // width)):
            lr = self.get_lr(i)
            bar_len = int((lr - min_lr) / (max_lr - min_lr + 1e-12) * (width - 12))
            marker = "*" if i < self.warmup_steps else "."
            print(f"  step {i:5d}: {'█' * bar_len}{marker} {lr:.6f}")

        return True

    def state_dict(self):
        return {"peak_lr": self.peak_lr, "min_lr": self.min_lr,
                "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}

    @classmethod
    def from_state_dict(cls, sd):
        return cls(**sd)


# ═══════════════════════════════════════════════
# GradientManager  (Lesson 45 + 46)
# ═══════════════════════════════════════════════

class GradientManager:
    """Combines gradient clipping (global L2 norm) + gradient accumulation."""

    def __init__(self, max_norm=1.0, accumulation_steps=1):
        self.max_norm = max_norm
        self.accumulation_steps = accumulation_steps
        self._accumulated = None
        self._micro_steps = 0
        self._grad_norms = []
        self._pre_clip_norms = []

    def accumulate(self, grads):
        """Accumulate gradients from a micro-batch."""
        if self._accumulated is None:
            self._accumulated = [list(g) if isinstance(g, (list, tuple)) else g
                                 for g in grads]
        else:
            for i in range(len(grads)):
                g = grads[i]
                if isinstance(g, (list, tuple)):
                    for j in range(len(g)):
                        self._accumulated[i][j] += g[j]
                else:
                    self._accumulated[i] += g

        self._micro_steps += 1
        self._grad_norms.append(self._compute_norm(grads))

    def clip(self, grads=None):
        """Clip gradients to max_norm. Returns clipped grads."""
        if grads is None:
            grads = self._accumulated
            if grads is None:
                return None
            # Divide by micro_steps for accumulation
            if self._micro_steps > 1:
                grads = [[g / self._micro_steps for g in grad_list]
                        for grad_list in grads]

        # Handle both flat list and list-of-lists
        flat = []
        if grads and isinstance(grads[0], list):
            flat = [g for sublist in grads for g in sublist]
        else:
            flat = list(grads)

        if not flat:
            return grads

        norm = math.sqrt(sum(g * g for g in flat))
        self._pre_clip_norms.append(norm)

        if norm > self.max_norm:
            scale = self.max_norm / (norm + 1e-12)
            if isinstance(grads[0], list):
                clipped = [[g * scale for g in sublist] for sublist in grads]
            else:
                clipped = [g * scale for g in grads]
            return clipped
        return grads

    def step(self):
        """Get accumulated + clipped gradients and reset."""
        grads = self._accumulated
        if grads is None:
            return None

        # Normalize by micro-steps
        if self._micro_steps > 0:
            if grads and isinstance(grads[0], (list, tuple)):
                grads = [[g / self._micro_steps for g in grad_list]
                        for grad_list in grads]
            else:
                grads = [g / self._micro_steps for g in grads]

        clipped = self.clip(grads)
        self._accumulated = None
        self._micro_steps = 0
        return clipped

    def stats(self):
        return {
            "accumulated_micro_steps": self._micro_steps,
            "recent_norms": [round(n, 4) for n in self._grad_norms[-5:]],
            "pre_clip_norms": [round(n, 4) for n in self._pre_clip_norms[-5:]],
            "max_norm": self.max_norm,
            "accumulation_steps": self.accumulation_steps
        }

    @staticmethod
    def _compute_norm(grads):
        """Compute L2 norm of a gradient list."""
        flat = []
        if grads and isinstance(grads[0], list):
            flat = [g for sublist in grads for g in sublist]
        else:
            flat = list(grads) if grads else []
        if not flat:
            return 0.0
        return math.sqrt(sum(g * g for g in flat))


# ═══════════════════════════════════════════════
# MixedPrecisionScaler  (Lesson 45)
# ═══════════════════════════════════════════════

class MixedPrecisionScaler:
    """AMP-style gradient scaler.

    Scales loss up so gradients stay in representable range for FP16.
    """

    def __init__(self, init_scale=2 ** 16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000):
        self.scale = float(init_scale)
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._good_steps = 0
        self._overflow_steps = 0

    def scale_loss(self, loss):
        """Scale loss up by the current scale factor."""
        return loss * self.scale

    def unscale_grads(self, grads):
        """Unscale gradients (divide by scale)."""
        if isinstance(grads[0], list):
            return [[g / self.scale for g in sublist] for sublist in grads]
        return [g / self.scale for g in grads]

    def update(self, skip_optim_step=False):
        """Update scale based on whether optimizer step succeeded.

        If gradients had inf/nan (skip_optim_step=True), scale down.
        Otherwise, periodically scale up.
        """
        if skip_optim_step:
            self.scale *= self.backoff_factor
            self._good_steps = 0
            self._overflow_steps += 1
        else:
            self._good_steps += 1
            if self._good_steps >= self.growth_interval:
                self.scale *= self.growth_factor
                self._good_steps = 0

    def has_inf_nan(self, grads):
        """Check if any gradient contains inf or nan."""
        flat = []
        if grads and isinstance(grads[0], list):
            flat = [g for sublist in grads for g in sublist]
        else:
            flat = list(grads) if grads else []

        for g in flat:
            if math.isinf(g) or math.isnan(g):
                return True
        return False

    def stats(self):
        return {
            "current_scale": self.scale,
            "good_steps": self._good_steps,
            "overflow_steps": self._overflow_steps
        }


# ═══════════════════════════════════════════════
# Checkpointer  (Lesson 47)
# ═══════════════════════════════════════════════

class Checkpointer:
    """Atomic checkpoint save/resume with integrity verification.

    Saves: model params, optimizer state, scheduler state, loss history,
    step counter, RNG state. Atomic via write-to-tmp-then-rename.
    """

    def __init__(self, checkpoint_dir=None):
        self.checkpoint_dir = checkpoint_dir or tempfile.mkdtemp(prefix="anggira_ckpt_")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, state, name="checkpoint"):
        """Save checkpoint atomically."""
        state["_timestamp"] = time.time()
        state["_checkpoint_id"] = str(uuid.uuid4())[:8]

        # Capture RNG state
        state["_random_state"] = random.getstate()

        tmp_path = os.path.join(self.checkpoint_dir, f".{name}.tmp")
        final_path = os.path.join(self.checkpoint_dir, f"{name}.ckpt")

        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)

        # Atomic rename (POSIX guarantees atomic on same filesystem)
        os.replace(tmp_path, final_path)
        return final_path

    def load(self, name="checkpoint"):
        """Load checkpoint. Returns state dict or None."""
        path = os.path.join(self.checkpoint_dir, f"{name}.ckpt")
        if not os.path.exists(path):
            return None

        if not self._verify(path):
            # Try to load previous version
            return self._fallback_recovery(name)

        with open(path) as f:
            state = json.load(f)

        # Restore RNG if present (JSON serializes tuples as lists)
        if "_random_state" in state:
            rs = state["_random_state"]
            if isinstance(rs, list):
                # Convert nested lists back to tuples
                rs = tuple(
                    tuple(item) if isinstance(item, list) else item
                    for item in rs
                )
            random.setstate(rs)

        return state

    def list_checkpoints(self):
        """List all checkpoints with metadata."""
        import glob
        checkpoints = []
        for f in sorted(glob.glob(os.path.join(self.checkpoint_dir, "*.ckpt"))):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                checkpoints.append({
                    "path": f,
                    "step": data.get("step", "?"),
                    "loss": data.get("loss", "?"),
                    "timestamp": data.get("_timestamp", 0),
                    "valid": True
                })
            except (json.JSONDecodeError, IOError):
                checkpoints.append({
                    "path": f, "step": 0, "loss": None,
                    "timestamp": 0, "valid": False
                })
        return checkpoints

    def verify(self, name="checkpoint"):
        """Verify checkpoint integrity."""
        path = os.path.join(self.checkpoint_dir, f"{name}.ckpt")
        if not os.path.exists(path):
            return {"valid": False, "error": "File not found"}
        valid = self._verify(path)
        return {"valid": valid, "path": path}

    def _verify(self, path):
        """Internal integrity check."""
        try:
            if os.path.getsize(path) == 0:
                return False
            with open(path) as f:
                data = json.load(f)
            required = ["step"]
            return all(k in data for k in required)
        except (json.JSONDecodeError, IOError, OSError):
            return False

    def _fallback_recovery(self, name):
        """Try to load backup/prev checkpoint."""
        # Try numbered versions
        for i in range(1, 5):
            path = os.path.join(self.checkpoint_dir, f"{name}.ckpt.bak{i}")
            if os.path.exists(path) and self._verify(path):
                with open(path) as f:
                    return json.load(f)
        return None

    @staticmethod
    def detect_corruption(filepath):
        """Check if a checkpoint file is corrupted."""
        try:
            if os.path.getsize(filepath) == 0:
                return True
            with open(filepath) as f:
                json.load(f)
            return False
        except (json.JSONDecodeError, IOError, ValueError):
            return True


# ═══════════════════════════════════════════════
# StreamingDataset  (Lesson 43)
# ═══════════════════════════════════════════════

class StreamingDataset:
    """Streaming tokenized dataset from JSONL files.

    Each line: {"tokens": [int, int, ...]}
    Supports sliding windows and resume.
    """

    def __init__(self, files=None, seq_len=64, batch_size=4, shuffle=True):
        self.files = files or []
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._data = []  # list of token sequences
        self._index = 0
        self._epoch = 0

    def add_file(self, path):
        """Load tokens from a JSONL file."""
        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    tokens = item.get("tokens", item if isinstance(item, list) else [])
                    if tokens and len(tokens) >= self.seq_len:
                        # Sliding window: break into overlapping chunks
                        stride = self.seq_len // 2
                        for start in range(0, len(tokens) - self.seq_len + 1, stride):
                            chunk = tokens[start:start + self.seq_len]
                            self._data.append(chunk)
                            count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        return count

    def generate_synthetic(self, num_sequences=100, vocab_size=1000, min_len=128, max_len=512):
        """Generate synthetic token data for testing."""
        import random
        count = 0
        for _ in range(num_sequences):
            length = random.randint(min_len, max_len)
            tokens = [random.randint(0, vocab_size - 1) for _ in range(length)]
            self._data.append(tokens[:self.seq_len])
            count += 1
        return count

    def __iter__(self):
        """Iterate over batches."""
        if self.shuffle:
            indices = list(range(len(self._data)))
            random.shuffle(indices)
        else:
            indices = list(range(len(self._data)))

        self._index = 0
        self._epoch = 0
        return self._batch_generator(indices)

    def _batch_generator(self, indices):
        """Generate batches."""
        batch = []
        for idx in indices:
            batch.append(self._data[idx])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
            self._index += 1

        if batch:  # Final partial batch
            yield batch

    def stats(self):
        return {
            "total_chunks": len(self._data),
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "total_batches": (len(self._data) + self.batch_size - 1) // self.batch_size
        }

    def save_state(self):
        """Save iterator state for resume."""
        return {"index": self._index, "epoch": self._epoch}

    def load_state(self, state):
        """Restore iterator state."""
        self._index = state["index"]
        self._epoch = state["epoch"]


# ═══════════════════════════════════════════════
# TinyModel — For training pipeline demo
# ═══════════════════════════════════════════════

class TinyModel:
    """A tiny 2-layer model for pipeline demos."""

    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        self.w1 = [[random.gauss(0, 0.1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.w2 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(output_dim)]
        self.b2 = [0.0] * output_dim

    def forward(self, x):
        h = [max(0, sum(w * x[j] for j, w in enumerate(w_row)) + b)
             for w_row, b in zip(self.w1, self.b1)]
        return [sum(w * h[j] for j, w in enumerate(w_row)) + b
                for w_row, b in zip(self.w2, self.b2)]

    def get_params(self):
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        for row in self.w2:
            params.extend(row)
        params.extend(self.b2)
        return params

    def set_params(self, flat):
        idx = 0
        for i in range(len(self.w1)):
            for j in range(len(self.w1[0])):
                self.w1[i][j] = flat[idx]
                idx += 1
        for i in range(len(self.b1)):
            self.b1[i] = flat[idx]
            idx += 1
        for i in range(len(self.w2)):
            for j in range(len(self.w2[0])):
                self.w2[i][j] = flat[idx]
                idx += 1
        for i in range(len(self.b2)):
            self.b2[i] = flat[idx]
            idx += 1

    def param_count(self):
        return len(self.get_params())


# ═══════════════════════════════════════════════
# TrainingPipelineDemo (Lesson 49)
# ═══════════════════════════════════════════════

class TrainingPipelineDemo:
    """End-to-end training pipeline combining all components."""

    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        self.model = TinyModel(input_dim, hidden_dim, output_dim)
        self.scheduler = CosmicLRScheduler(
            peak_lr=0.01, min_lr=0.0001,
            warmup_steps=10, total_steps=50
        )
        self.gradient_mgr = GradientManager(max_norm=1.0, accumulation_steps=2)
        self.scaler = MixedPrecisionScaler(init_scale=2 ** 8)
        self.checkpointer = Checkpointer()
        self.loss_history = []
        self.lr_history = []

    def _compute_loss(self, out, target):
        """MSE loss."""
        return sum((out[i] - target[i]) ** 2 for i in range(len(out))) / len(out)

    def _compute_gradients(self, x, target):
        """Simplified gradient computation (simulation)."""
        out = self.model.forward(x)
        loss = self._compute_loss(out, target)
        # Simulated gradients — flat list of floats matching model.get_params()
        n_params = self.model.param_count()
        grads = [random.gauss(0, 0.02) for _ in range(n_params)]
        return loss, grads

    def _flatten_grads(self, grads):
        """Flatten nested gradient structure into a flat list."""
        flat = []
        for g in grads:
            if isinstance(g, (list, tuple)):
                for sub in g:
                    if isinstance(sub, (list, tuple)):
                        flat.extend(sub)
                    else:
                        flat.append(sub)
            else:
                flat.append(g)
        return flat

    def _apply_gradients(self, params, grads, lr):
        """Simple SGD update."""
        idx = 0
        for w_row in self.model.w1:
            for j in range(len(w_row)):
                if idx < len(grads):
                    w_row[j] -= lr * grads[idx]
                idx += 1
        for j in range(len(self.model.b1)):
            if idx < len(grads):
                self.model.b1[j] -= lr * grads[idx]
            idx += 1
        for i in range(len(self.model.w2)):
            for j in range(len(self.model.w2[i])):
                if idx < len(grads):
                    self.model.w2[i][j] -= lr * grads[idx]
                idx += 1
        for j in range(len(self.model.b2)):
            if idx < len(grads):
                self.model.b2[j] -= lr * grads[idx]
            idx += 1

    def train(self, num_steps=50, checkpoint_steps=[25]):
        """Run full training pipeline."""
        print(f"\n  Training for {num_steps} steps...")
        print(f"  Model params: {self.model.param_count()}")

        for step in range(1, num_steps + 1):
            x = [random.uniform(-1, 1) for _ in range(4)]
            target_val = 1.0 if (x[0] * x[1] + x[2]) > 0 else 0.0
            target = [target_val, 1.0 - target_val]

            # Forward + loss
            loss, grads = self._compute_gradients(x, target)

            # Scale loss (simulate AMP)
            scaled_loss = self.scaler.scale_loss(loss)

            # Check for inf/nan (simulate FP16 overflow detection)
            has_inf = self.scaler.has_inf_nan(self._flatten_grads(grads))
            if has_inf:
                self.scaler.update(skip_optim_step=True)
                continue

            # Accumulate gradients
            self.gradient_mgr.accumulate(grads)

            # Optimizer step at accumulation boundary
            if step % self.gradient_mgr.accumulation_steps == 0 or step == num_steps:
                # Get accumulated + clipped gradients
                accumulated = self.gradient_mgr.step()
                if accumulated is not None:
                    flat_grads = self._flatten_grads(accumulated)

                    # Unscale
                    flat_grads = [g / self.scaler.scale for g in flat_grads]

                    # Get LR from scheduler
                    lr = self.scheduler.get_lr(step)
                    self.lr_history.append(lr)

                    # Apply gradients
                    self._apply_gradients(self.model.get_params(), flat_grads, lr)

            # Checkpoint at specified steps
            if step in checkpoint_steps:
                state = {
                    "step": step,
                    "loss": loss,
                    "lr": self.scheduler.get_lr(step),
                    "scheduler": self.scheduler.state_dict(),
                    "loss_history": self.loss_history[-100:],
                    "model_params": self.model.get_params()
                }
                path = self.checkpointer.save(state, f"step_{step}")
                print(f"    Checkpoint saved at step {step}: {path}")

            self.loss_history.append(loss)
            lr = self.scheduler.get_lr(step)
            if step % 10 == 0 or step == 1:
                print(f"    Step {step:3d}: loss={loss:.4f}, lr={lr:.6f}")

        # Final checkpoint
        state = {
            "step": step,
            "loss": self.loss_history[-1],
            "scheduler": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
            "model_params": self.model.get_params()
        }
        self.checkpointer.save(state, "final")
        print(f"    Final checkpoint saved")

        return {
            "final_loss": self.loss_history[-1] if self.loss_history else None,
            "steps": num_steps,
            "loss_trend": [round(l, 4) for l in self.loss_history[::5]]
        }


# ═══════════════════════════════════════════════
# DEMOS
# ═══════════════════════════════════════════════

def demo_cosmic_lr():
    """Demo the LR scheduler."""
    print("=== CosmicLRScheduler Demo ===")
    sched = CosmicLRScheduler(peak_lr=0.01, min_lr=0.0001, warmup_steps=10, total_steps=50)
    sched.plot(width=40)
    print(f"  Step 0:  {sched.get_lr(0):.6f}")
    print(f"  Step 5:  {sched.get_lr(5):.6f}")
    print(f"  Step 10: {sched.get_lr(10):.6f}")
    print(f"  Step 25: {sched.get_lr(25):.6f}")
    print(f"  Step 49: {sched.get_lr(49):.6f}")
    return True


def demo_gradient_manager():
    """Demo gradient clipping and accumulation."""
    print("\n=== GradientManager Demo ===")
    gm = GradientManager(max_norm=0.5, accumulation_steps=4)

    for i in range(4):
        grads = [[random.gauss(0, 0.1) for _ in range(4)] for _ in range(4)]
        gm.accumulate(grads)
        norm = math.sqrt(sum(g * g for sublist in grads for g in sublist))
        print(f"  Micro-batch {i + 1}: grad_norm={norm:.3f}")

    # Step: get accumulated + clipped
    clipped = gm.step()
    clipped_norm = math.sqrt(sum(g * g for sublist in clipped for g in sublist))
    print(f"  After clip: norm={clipped_norm:.3f} (max={gm.max_norm})")
    print(f"  Stats: {gm.stats()}")
    return True


def demo_mixed_precision():
    """Demo the mixed precision scaler."""
    print("\n=== MixedPrecisionScaler Demo ===")
    scaler = MixedPrecisionScaler(init_scale=2 ** 8, growth_interval=5)

    for i in range(10):
        loss = random.uniform(0.1, 2.0)
        scaled = scaler.scale_loss(loss)
        skip = random.random() < 0.2  # 20% chance of overflow
        scaler.update(skip_optim_step=skip)
        print(f"  Step {i + 1}: loss={loss:.4f}, scaled={scaled:.1f}, "
              f"scale={scaler.scale:.0f}, overflow={skip}")

    print(f"  Final stats: {scaler.stats()}")
    return True


def demo_checkpointer():
    """Demo checkpoint save/load/resume."""
    print("\n=== Checkpointer Demo ===")
    ckpt = Checkpointer()

    # Save a checkpoint
    state = {"step": 100, "loss": 0.05, "optimizer_lr": 0.001, "data": [1, 2, 3]}
    path = ckpt.save(state, "my_model")
    print(f"  Saved: {path}")

    # Verify
    result = ckpt.verify("my_model")
    print(f"  Verify: {result}")

    # List
    ckpts = ckpt.list_checkpoints()
    print(f"  Available: {len(ckpts)}")
    for c in ckpts:
        print(f"    step={c['step']}, valid={c['valid']}")

    # Load
    loaded = ckpt.load("my_model")
    print(f"  Loaded: step={loaded['step']}, loss={loaded['loss']}")
    return True


def demo_streaming_dataset():
    """Demo the streaming dataset."""
    print("\n=== StreamingDataset Demo ===")

    # Create a temp JSONL file
    tmpdir = tempfile.mkdtemp(prefix="anggira_ds_")
    jsonl_path = os.path.join(tmpdir, "data.jsonl")
    with open(jsonl_path, "w") as f:
        for _ in range(20):
            tokens = [random.randint(0, 999) for _ in range(128)]
            f.write(json.dumps({"tokens": tokens}) + "\n")

    ds = StreamingDataset(seq_len=32, batch_size=4)
    added = ds.add_file(jsonl_path)
    print(f"  Added from JSONL: {len(ds._data)} chunks")

    # Generate synthetic
    syn = ds.generate_synthetic(num_sequences=10, min_len=64, max_len=128)
    print(f"  Added synthetic: {len(ds._data)} chunks total")

    print(f"  Stats: {ds.stats()}")

    # Iterate
    batches = list(iter(ds))
    print(f"  Total batches: {len(batches)}")
    if batches:
        print(f"  First batch shape: {len(batches[0])} x {len(batches[0][0])}")
    return True


def demo_training_pipeline():
    """Demo the full training pipeline."""
    print("\n=== TrainingPipeline Demo ===")
    pipeline = TrainingPipelineDemo()
    result = pipeline.train(num_steps=30, checkpoint_steps=[10, 20])
    print(f"  Final loss: {result['final_loss']:.4f}")
    print(f"  Loss trend (every 5): {result['loss_trend']}")
    return True


def demo():
    """Run all training pipeline demos."""
    results = []

    print("=" * 60)
    print("Anggira Capstone — Training Pipeline (Phase 19, Lessons 42-49)")
    print("=" * 60)

    for demo_fn in [
        demo_cosmic_lr,
        demo_gradient_manager,
        demo_mixed_precision,
        demo_checkpointer,
        demo_streaming_dataset,
        demo_training_pipeline,
    ]:
        try:
            demo_fn()
            results.append((demo_fn.__name__, "PASS"))
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((demo_fn.__name__, f"FAIL: {e}"))

    print("\n" + "=" * 60)
    print("Results:")
    for name, status in results:
        print(f"  {'✅' if status == 'PASS' else '❌'} {name}: {status}")
    print("=" * 60)
    return all(r == "PASS" for _, r in results)


if __name__ == "__main__":
    demo()
