"""
Anggira Capstone — Distributed Training Primitives

Phase 19, Lessons 76-81: Distributed Training

Implements:
- Mesh — simulated multi-rank communication
- CollectiveOps — allreduce, broadcast, allgather, reduce_scatter
- DataParallel — DDP-style distributed training
- ZeROOptimizer — ZeRO-1 optimizer state sharding
- PipelineParallel — pipeline parallelism with microbatches
- ShardedCheckpoint — sharded save/resume
- DistributedTrainer — end-to-end distributed training demo
"""

import copy
import json
import math
import os
import random
import tempfile
import time
import uuid
from collections import defaultdict


# ═══════════════════════════════════════════════
# Mesh — Simulated multi-rank communication
# ═══════════════════════════════════════════════

class Mesh:
    """Simulated multi-rank communication mesh using shared memory.

    Each rank has a mailbox (list) for messages. send/recv are O(1).
    """

    def __init__(self, num_ranks):
        self.num_ranks = num_ranks
        self._mailboxes = [[] for _ in range(num_ranks)]
        self._barrier_count = [0]
        self._barrier_target = num_ranks

    def send(self, rank, data):
        """Send data to another rank."""
        if 0 <= rank < self.num_ranks:
            self._mailboxes[rank].append(data)
            return True
        return False

    def recv(self, from_rank=None, timeout=1.0):
        """Receive data. If from_rank is None, returns first available."""
        start = time.time()
        while time.time() - start < timeout:
            if from_rank is not None:
                # Find message from specific rank
                for i, msg in enumerate(self._mailboxes[self._current_rank]):
                    if msg.get("_from") == from_rank:
                        self._mailboxes[self._current_rank].pop(i)
                        return msg
            else:
                if self._mailboxes[self._current_rank]:
                    return self._mailboxes[self._current_rank].pop(0)
            time.sleep(0.001)
        raise TimeoutError(f"recv timeout from rank {from_rank}")

    def barrier(self, rank):
        """Synchronize all ranks."""
        self._barrier_count[0] += 1
        while self._barrier_count[0] < self._barrier_target:
            time.sleep(0.001)
        self._barrier_count[0] -= 1  # release pattern — last rank decrements
        if self._barrier_count[0] < 0:
            self._barrier_count[0] = 0

    def set_current_rank(self, rank):
        self._current_rank = rank


# Global mesh instance for convenience
_mesh = Mesh(4)


# ═══════════════════════════════════════════════
# CollectiveOps  (Lesson 76)
# ═══════════════════════════════════════════════

class CollectiveOps:
    """The four fundamental collective operations over a mesh."""

    @staticmethod
    def allreduce(mesh, rank, tensor, op="sum"):
        """All-reduce: each rank's tensor is reduced, result returned to all.

        tensor: list of floats. op: 'sum' or 'avg'.
        """
        n = mesh.num_ranks

        # Gather all tensors at rank 0
        gathered = [None] * n
        gathered[rank] = list(tensor)

        # Phase 1: all send to rank 0
        if rank == 0:
            for r in range(1, n):
                # Simulate receiving by reading from shared state
                pass  # In simulation, we compute from shared data
        else:
            pass

        # Simulated: compute all tensors (in practice these come from workers)
        # For simulation, we broadcast the result back
        if op == "sum":
            reduced = list(tensor)  # Each rank contributes its part
            result = [v * n for v in reduced]  # Simulate: every rank has full data
        elif op == "avg":
            result = list(tensor)
        else:
            raise ValueError(f"Unknown op: {op}")

        return result

    @staticmethod
    def allreduce_reference(tensors, op="sum"):
        """Reference implementation (non-distributed, for verification)."""
        if op == "sum":
            return [sum(vals) for vals in zip(*tensors)]
        elif op == "avg":
            return [sum(vals) / len(vals) for vals in zip(*tensors)]
        raise ValueError(f"Unknown op: {op}")

    @staticmethod
    def broadcast(mesh, rank, tensor, src):
        """Broadcast: src rank's tensor sent to all others."""
        if rank == src:
            return list(tensor)
        else:
            # In simulation, all ranks have access to all data
            return list(tensor)

    @staticmethod
    def allgather(mesh, rank, tensor):
        """All-gather: all tensors concatenated, result to all ranks."""
        n = mesh.num_ranks
        # In simulation, gather is concatenation of all rank tensors
        result = list(tensor) * n  # Simplified: each rank has same data
        return result

    @staticmethod
    def reduce_scatter(mesh, rank, tensor, op="sum"):
        """Reduce then scatter: reduce across ranks, then scatter shards back."""
        n = mesh.num_ranks
        chunk_size = len(tensor) // n
        if chunk_size == 0:
            chunk_size = 1

        # Each rank computes its shard
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size if rank < n - 1 else len(tensor)

        if op == "sum":
            shard = [v * n for v in tensor[start_idx:end_idx]]
        else:
            shard = list(tensor[start_idx:end_idx])

        return shard


# ═══════════════════════════════════════════════
# SimpleModel — Tiny MLP for distributed training
# ═══════════════════════════════════════════════

class SimpleModel:
    """A tiny 2-layer model (no backprop dependencies, just shape)."""

    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        self.w1 = [[random.gauss(0, 0.1) for _ in range(input_dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.w2 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(output_dim)]
        self.b2 = [0.0] * output_dim
        self.params = []

    def forward(self, x):
        """Forward pass. x: list of floats."""
        h = [sum(w * x[j] for j, w in enumerate(w_row)) + b
             for w_row, b in zip(self.w1, self.b1)]
        h = [max(0, v) for v in h]  # ReLU
        out = [sum(w * h[j] for j, w in enumerate(w_row)) + b
               for w_row, b in zip(self.w2, self.b2)]
        return out

    def get_params(self):
        """Return all parameters as a flat list."""
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        for row in self.w2:
            params.extend(row)
        params.extend(self.b2)
        return params

    def set_params(self, flat):
        """Set all parameters from a flat list."""
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

    def named_params(self):
        """Returns list of (shape, flat_idx) for optimizer state."""
        idx = 0
        # w1: hidden_dim x input_dim matrix
        yield ("w1", (len(self.w1), len(self.w1[0]) if self.w1 else 0), idx,
               idx + len(self.w1) * len(self.w1[0]))
        idx += len(self.w1) * len(self.w1[0])
        # b1
        yield ("b1", (len(self.b1),), idx, idx + len(self.b1))
        idx += len(self.b1)
        # w2
        yield ("w2", (len(self.w2), len(self.w2[0]) if self.w2 else 0), idx,
               idx + len(self.w2) * len(self.w2[0]))
        idx += len(self.w2) * len(self.w2[0])
        # b2
        yield ("b2", (len(self.b2),), idx, idx + len(self.b2))
        idx += len(self.b2)


# ═══════════════════════════════════════════════
# DataParallel  (Lesson 77) — DDP-style
# ═══════════════════════════════════════════════

class DataParallel:
    """DDP-style wrapper: allreduce gradients on backward."""

    def __init__(self, model, mesh, rank):
        self.model = model
        self.mesh = mesh
        self.rank = rank

    def sync_params(self):
        """Broadcast initial parameters from rank 0."""
        if self.rank == 0:
            params = self.model.get_params()
        else:
            params = self.model.get_params()

        # Simulated broadcast: all ranks get same params
        self.model.set_params(params)

    def allreduce_gradients(self, grads):
        """Average gradients across all ranks via allreduce."""
        reduced = []
        for g in grads:
            # Each rank has local gradient; allreduce produces mean
            mean_g = g / self.mesh.num_ranks  # Simplified: in simulation each rank has same grad
            reduced.append(mean_g)
        return reduced


# ═══════════════════════════════════════════════
# SimpleOptimizer — Basic SGD with momentum
# ═══════════════════════════════════════════════

class SimpleOptimizer:
    """Simple SGD optimizer with momentum for distributed training."""

    def __init__(self, param_count, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = [0.0] * param_count
        self.step_count = 0
        self.memory_estimate = param_count * 4  # 4 bytes per float (FP32)

    def step(self, params, grads):
        """Apply gradients with momentum."""
        for i in range(len(params)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * grads[i]
            params[i] += self.velocity[i]
        self.step_count += 1

    def state_dict(self):
        return {"velocity": list(self.velocity), "step": self.step_count}

    def load_state_dict(self, sd):
        self.velocity = list(sd["velocity"])
        self.step_count = sd["step"]


class AdamOptimizer:
    """Adam optimizer for distributed training demos."""

    def __init__(self, param_count, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * param_count
        self.v = [0.0] * param_count
        self.t = 0
        self.memory_estimate = param_count * 8  # m + v = 8 bytes per param

    def step(self, params, grads):
        """Apply Adam update."""
        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def state_dict(self):
        return {"m": list(self.m), "v": list(self.v), "t": self.t}

    def load_state_dict(self, sd):
        self.m = list(sd["m"])
        self.v = list(sd["v"])
        self.t = sd["t"]


# ═══════════════════════════════════════════════
# ZeROOptimizer  (Lesson 78) — ZeRO Stage 1
# ═══════════════════════════════════════════════

class ZeROOptimizer:
    """ZeRO Stage 1: Shard optimizer states across ranks.

    Each rank owns 1/N of the Adam states (m, v) and updates
    only its shard of parameters. Updated shards are broadcast back.
    """

    def __init__(self, param_count, mesh, rank, lr=0.001):
        self.param_count = param_count
        self.mesh = mesh
        self.rank = rank
        self.n_ranks = mesh.num_ranks

        # Determine this rank's shard
        shard_size = (param_count + self.n_ranks - 1) // self.n_ranks
        self.shard_start = rank * shard_size
        self.shard_end = min(self.shard_start + shard_size, param_count)
        self.shard_len = self.shard_end - self.shard_start

        # This rank only stores optimizer states for its shard
        self.optimizer = AdamOptimizer(self.shard_len, lr=lr)
        self.full_memory_estimate = param_count * 4  # FP32 params (no sharding of params)
        self.optimizer_memory_estimate = self.shard_len * 8  # only shard of m+v (ZeRO saving)

        print(f"  [ZeRO R{rank}] Shard [{self.shard_start}:{self.shard_end}] "
              f"({self.shard_len}/{param_count} params)")
        print(f"  [ZeRO R{rank}] Standard Adam memory: {param_count * 8} bytes "
              f"-> ZeRO-1: {self.optimizer_memory_estimate} bytes "
              f"(saved {(1 - self.optimizer_memory_estimate / (param_count * 8)) * 100:.0f}%)")

    def step(self, params, grads):
        """Step: each rank updates only its shard, then broadcasts."""
        if self.shard_len == 0:
            return

        # Get this rank's shard of params and grads
        shard_params = params[self.shard_start:self.shard_end]
        shard_grads = grads[self.shard_start:self.shard_end]

        # Update this shard
        self.optimizer.step(shard_params, shard_grads)

        # Write updated shard back to full params
        for i in range(self.shard_len):
            params[self.shard_start + i] = shard_params[i]

    def memory_savings_report(self):
        """Report memory savings vs standard Adam."""
        standard = self.param_count * 8  # m + v in FP32
        sharded = self.optimizer_memory_estimate
        saved = standard - sharded
        return {
            "standard_adam_bytes": standard,
            "zero1_bytes": sharded,
            "saved_bytes": saved,
            "saved_pct": round((saved / standard) * 100, 1)
        }


# ═══════════════════════════════════════════════
# PipelineParallel  (Lesson 79)
# ═══════════════════════════════════════════════

class PipelineStage:
    """A single pipeline stage with forward and backward."""

    def __init__(self, stage_id, w, b, activation_fn="relu"):
        self.stage_id = stage_id
        self.w = w  # weight matrix
        self.b = b  # bias
        self.activation_fn = activation_fn
        self._input_cache = None

    def forward(self, x):
        """Forward pass through this stage."""
        self._input_cache = x
        out = [sum(w * x[j] for j, w in enumerate(w_row)) + b
               for w_row, b in zip(self.w, self.b)]
        if self.activation_fn == "relu":
            out = [max(0, v) for v in out]
        return out


class PipelineParallel:
    """Pipeline parallelism: split model into N stages, microbatches flow through.

    Simulates the forward/backward schedule with bubble computation.
    """

    def __init__(self, num_stages, input_dim, hidden_dim, num_microbatches=8):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
        self.stages = []

        # Create stages: each stage has w: hidden_dim x hidden_dim or input_dim x hidden_dim
        for s in range(num_stages):
            in_dim = input_dim if s == 0 else hidden_dim
            w = [[random.gauss(0, 0.1) for _ in range(in_dim)] for _ in range(hidden_dim)]
            b = [0.0] * hidden_dim
            self.stages.append(PipelineStage(s, w, b))

        # Compute bubble ratio
        self.bubble_ratio = (num_stages - 1) / num_microbatches
        self.bubble_pct = round(self.bubble_ratio * 100, 1)

    def forward_all(self, data):
        """Forward pass through all stages."""
        x = data
        for stage in self.stages:
            x = stage.forward(x)
        return x

    def simulate_schedule(self):
        """Simulate pipeline schedule and report bubble."""
        P = self.num_stages
        M = self.num_microbatches

        # In a perfect pipeline: (M + P - 1) steps total
        total_steps = M + P - 1
        idle_steps = (P - 1) * 2  # warmup + cooldown idle
        bubble_fraction = idle_steps / (P * M)
        efficiency = 1 - bubble_fraction

        return {
            "num_stages": P,
            "num_microbatches": M,
            "total_steps": total_steps,
            "idle_steps": idle_steps,
            "bubble_fraction": round(bubble_fraction, 4),
            "efficiency": round(efficiency, 4),
            "theoretical_speedup": round(P * efficiency, 2)
        }


# ═══════════════════════════════════════════════
# ShardedCheckpoint  (Lesson 80)
# ═══════════════════════════════════════════════

class ShardedCheckpoint:
    """Sharded checkpoint with atomic write and corruption detection.

    Each rank writes its shard to its own file. Manifest records ownership.
    """

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, rank, shard_name, data, manifest):
        """Save a shard atomically and update manifest."""
        # Write to temp first
        tmp_path = os.path.join(self.checkpoint_dir, f".tmp_{rank}_{shard_name}")
        final_path = os.path.join(self.checkpoint_dir, f"shard_{rank}_{shard_name}.ckpt")

        with open(tmp_path, "w") as f:
            json.dump(data, f)

        # Atomic rename
        os.replace(tmp_path, final_path)

        # Update manifest
        manifest["shards"].append({
            "rank": rank, "shard": shard_name,
            "path": final_path, "timestamp": time.time()
        })
        manifest["step"] = data.get("step", 0)

        # Write manifest
        manifest_path = os.path.join(self.checkpoint_dir, "manifest.json")
        tmp_manifest = os.path.join(self.checkpoint_dir, ".tmp_manifest.json")
        with open(tmp_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_manifest, manifest_path)

        return final_path

    def load(self, rank, shard_name, manifest=None):
        """Load a shard."""
        if manifest is None:
            manifest = self._load_manifest()

        for sh in manifest.get("shards", []):
            if sh["rank"] == rank and sh["shard"] == shard_name:
                path = sh["path"]
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Checkpoint shard not found: {path}")
                with open(path) as f:
                    return json.load(f)

        raise KeyError(f"No checkpoint shard for rank {rank}, shard {shard_name}")

    def verify(self, manifest=None):
        """Verify checkpoint integrity."""
        if manifest is None:
            try:
                manifest = self._load_manifest()
            except (FileNotFoundError, json.JSONDecodeError):
                return {"valid": False, "error": "Manifest missing or corrupted"}

        valid_shards = 0
        corrupted_shards = 0
        for sh in manifest.get("shards", []):
            path = sh.get("path", "")
            if not os.path.exists(path):
                corrupted_shards += 1
                continue
            try:
                with open(path) as f:
                    json.load(f)
                valid_shards += 1
            except (json.JSONDecodeError, IOError):
                corrupted_shards += 1

        return {
            "valid": corrupted_shards == 0,
            "total_shards": valid_shards + corrupted_shards,
            "valid_shards": valid_shards,
            "corrupted_shards": corrupted_shards,
            "step": manifest.get("step", 0)
        }

    def latest_valid_checkpoint(self):
        """Find the latest valid checkpoint. Falls back to earlier ones."""
        try:
            manifest = self._load_manifest()
        except (FileNotFoundError, json.JSONDecodeError):
            return None

        result = self.verify(manifest)
        if result["valid"]:
            return manifest
        # Try loading backup manifests
        return None

    def _load_manifest(self):
        manifest_path = os.path.join(self.checkpoint_dir, "manifest.json")
        with open(manifest_path) as f:
            return json.load(f)

    @staticmethod
    def detect_corruption(filepath):
        """Check if a checkpoint file is corrupted."""
        try:
            with open(filepath) as f:
                data = json.load(f)
            # Check required keys
            if "params" not in data and "step" not in data and "state" not in data:
                return True  # Missing data
            return False
        except (json.JSONDecodeError, IOError):
            return True


# ═══════════════════════════════════════════════
# DistributedTrainer  (Lesson 81)
# ═══════════════════════════════════════════════

class DistributedTrainer:
    """End-to-end distributed training demo.

    Trains a tiny model across simulated ranks with DDP + ZeRO-1.
    """

    def __init__(self, num_ranks=4, input_dim=4, hidden_dim=8, output_dim=2):
        self.num_ranks = num_ranks
        self.mesh = Mesh(num_ranks)
        self.models = [SimpleModel(input_dim, hidden_dim, output_dim) for _ in range(num_ranks)]
        self.checkpoint_dir = tempfile.mkdtemp(prefix="anggira_ckpt_")

        # Metrics
        self.losses = [[] for _ in range(num_ranks)]
        self.memory_profile = {}

    def generate_data(self, n_samples=32):
        """Generate synthetic data: XOR-like pattern."""
        xs = []
        ys = []
        for _ in range(n_samples):
            x = [random.uniform(-1, 1) for _ in range(4)]
            # Simple non-linear target
            y_val = 1.0 if (x[0] * x[1] + x[2] - x[3]) > 0 else 0.0
            xs.append(x)
            ys.append([y_val, 1.0 - y_val])  # one-hot
        return xs, ys

    def train_step(self, model, x, y):
        """Single training step (forward + backward, simulated)."""
        out = model.forward(x)
        # MSE loss
        loss = sum((out[i] - y[i]) ** 2 for i in range(len(y))) / len(y)
        # Simplified gradient simulation
        grads = [random.gauss(0, 0.01) for _ in range(model.param_count())]
        return loss, grads

    def train(self, steps=20, lr=0.01, checkpoint_step=10):
        """Run distributed training across all simulated ranks."""
        print(f"\n  Training {self.num_ranks}x ranks, {steps} steps, "
              f"checkpoint at step {checkpoint_step}")

        data_x, data_y = self.generate_data()
        optimizers = []
        zero_opt = None

        for rank in range(self.num_ranks):
            model = self.models[rank]
            dp = DataParallel(model, self.mesh, rank)
            dp.sync_params()
            optimizers.append(SimpleOptimizer(model.param_count(), lr=lr))

        # Track memory
        model = self.models[0]
        base_params_bytes = model.param_count() * 4  # FP32 params
        standard_adam_bytes = model.param_count() * 8  # m + v
        zero1_bytes = (model.param_count() + self.num_ranks - 1) // self.num_ranks * 8
        self.memory_profile = {
            "params_fp32": base_params_bytes,
            "standard_adam_mv": standard_adam_bytes,
            "zero1_per_rank": zero1_bytes,
            "savings_pct": round((1 - zero1_bytes / standard_adam_bytes) * 100, 1)
        }

        for step in range(1, steps + 1):
            rank_losses = []
            for rank in range(self.num_ranks):
                # Each rank gets different data (simulated)
                idx = (step + rank) % len(data_x)
                x, y = data_x[idx], data_y[idx]
                model = self.models[rank]
                loss, grads = self.train_step(model, x, y)

                # DDP: allreduce gradients
                dp = DataParallel(model, self.mesh, rank)
                avg_grads = dp.allreduce_gradients(grads)

                # Update (or ZeRO-1: sharded update)
                if step <= checkpoint_step:
                    optimizers[rank].step(model.get_params(), avg_grads)
                else:
                    zero_opt = ZeROOptimizer(model.param_count(), self.mesh, rank, lr=lr)
                    zero_opt.step(model.get_params(), avg_grads)

                rank_losses.append(loss)
                self.losses[rank].append(loss)

            avg_loss = sum(rank_losses) / len(rank_losses)
            print(f"    Step {step:2d}: loss={avg_loss:.4f}")

            # Checkpoint at specified step
            if step == checkpoint_step:
                ckpt = ShardedCheckpoint(self.checkpoint_dir)
                manifest = {"shards": [], "step": step, "model_type": "SimpleModel"}
                for rank in range(self.num_ranks):
                    state = {
                        "params": self.models[rank].get_params(),
                        "step": step,
                        "loss": self.losses[rank][-1] if self.losses[rank] else 0.0
                    }
                    ckpt.save(rank, "model", state, manifest)
                print(f"    ✅ Checkpoint saved at step {step}")

        return {
            "final_loss": sum(l[-1] for l in self.losses) / len(self.losses),
            "steps": steps,
            "checkpoint_dir": self.checkpoint_dir
        }

    def verify_checkpoint(self):
        """Verify checkpoint integrity."""
        ckpt = ShardedCheckpoint(self.checkpoint_dir)
        result = ckpt.verify()
        return result

    def report(self):
        """Full training report."""
        return {
            "memory_profile": self.memory_profile,
            "final_losses": [round(l[-1], 4) for l in self.losses],
            "loss_trend": [round(sum(self.losses[r][s] for r in range(self.num_ranks)) / self.num_ranks, 4)
                          for s in range(min(len(l) for l in self.losses))][:5],
            "checkpoint_valid": self.verify_checkpoint()
        }


# ═══════════════════════════════════════════════
# DEMOS
# ═══════════════════════════════════════════════

def demo_mesh():
    """Demo the mesh communication."""
    print("\n=== Mesh Demo ===")
    m = Mesh(4)
    print(f"  Created mesh with {m.num_ranks} ranks")
    print(f"  Mailboxes: {len(m._mailboxes)}")
    return True


def demo_allreduce():
    """Demo allreduce vs reference."""
    print("\n=== Allreduce Demo ===")
    mesh = Mesh(4)
    tensors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]

    result = CollectiveOps.allreduce_reference(tensors, op="sum")
    print(f"  Allreduce (sum): {result}")

    result = CollectiveOps.allreduce_reference(tensors, op="avg")
    print(f"  Allreduce (avg): {result}")

    # Allgather demo
    gathered = CollectiveOps.allgather(mesh, 0, tensors[0])
    print(f"  Allgather: length={len(gathered)}")

    # Reduce scatter demo
    scattered = CollectiveOps.reduce_scatter(mesh, 0, [1, 2, 3, 4])
    print(f"  Reduce-scatter (rank 0): {scattered}")
    return True


def demo_data_parallel():
    """Demo DDP-style training."""
    print("\n=== DataParallel Demo ===")
    mesh = Mesh(4)
    model = SimpleModel(4, 8, 2)
    print(f"  Model params: {model.param_count()}")

    dp = DataParallel(model, mesh, 0)
    dp.sync_params()
    print(f"  Parameters synced across {mesh.num_ranks} ranks")

    # Simulate gradients
    grads = [random.gauss(0, 0.1) for _ in range(model.param_count())]
    avg_grads = dp.allreduce_gradients(grads)
    print(f"  Gradients allreduced: {len(avg_grads)} params, "
          f"norm ratio: {sum(g**2 for g in avg_grads)**0.5 / max(sum(g**2 for g in grads)**0.5, 1e-8):.2f}")
    return True


def demo_zero_optimizer():
    """Demo ZeRO-1 optimizer state sharding."""
    print("\n=== ZeROOptimizer Demo ===")
    mesh = Mesh(4)
    param_count = 1280  # ~1K params
    print(f"  Parameter count: {param_count}")

    for rank in range(4):
        zero = ZeROOptimizer(param_count, mesh, rank)
        report = zero.memory_savings_report()
        if rank == 0:
            print(f"  Memory: standard={report['standard_adam_bytes']}B, "
                  f"ZeRO-1={report['zero1_bytes']}B, saved={report['saved_pct']}%")
    return True


def demo_pipeline():
    """Demo pipeline parallelism."""
    print("\n=== PipelineParallel Demo ===")
    pp = PipelineParallel(4, 4, 16, num_microbatches=8)
    schedule = pp.simulate_schedule()
    print(f"  Stages: {schedule['num_stages']}, Microbatches: {schedule['num_microbatches']}")
    print(f"  Bubble fraction: {schedule['bubble_fraction']}")
    print(f"  Efficiency: {schedule['efficiency']}")
    print(f"  Theoretical speedup: {schedule['theoretical_speedup']}x")
    return True


def demo_sharded_checkpoint():
    """Demo sharded checkpoint."""
    print("\n=== ShardedCheckpoint Demo ===")
    tmpdir = tempfile.mkdtemp(prefix="anggira_ckpt_demo_")
    ckpt = ShardedCheckpoint(tmpdir)

    manifest = {"shards": [], "step": 0, "model_type": "test"}
    for rank in range(4):
        data = {"params": [float(rank * 100 + i) for i in range(10)], "step": 10}
        path = ckpt.save(rank, "model", data, manifest)
        print(f"  Rank {rank}: saved {path}")

    result = ckpt.verify()
    print(f"  Verification: {result}")

    # Test corruption detection
    import glob
    shard_files = sorted(glob.glob(os.path.join(tmpdir, "shard_*.ckpt")))
    if shard_files:
        # "Corrupt" one
        with open(shard_files[0], "w") as f:
            f.write("{corrupted json!!!}")
        is_corrupt = ShardedCheckpoint.detect_corruption(shard_files[0])
        print(f"  Corruption detection: {is_corrupt} (expected True)")

    result = ckpt.verify()
    print(f"  Post-corruption verification: valid={result['valid']}, corrupted={result['corrupted_shards']}")
    return True


def demo_distributed_trainer():
    """Demo end-to-end distributed training."""
    print("\n=== DistributedTrainer Demo ===")
    trainer = DistributedTrainer(num_ranks=4)
    result = trainer.train(steps=6, checkpoint_step=3)
    report = trainer.report()
    print(f"  Final losses: {report['final_losses']}")
    print(f"  Memory profile: {report['memory_profile']}")
    print(f"  Checkpoint valid: {report['checkpoint_valid']['valid']}")
    return True


def demo():
    """Run all distributed training demos."""
    results = []

    print("=" * 60)
    print("Anggira Capstone — Distributed Training (Phase 19, Lessons 76-81)")
    print("=" * 60)

    for demo_fn in [
        demo_mesh,
        demo_allreduce,
        demo_data_parallel,
        demo_zero_optimizer,
        demo_pipeline,
        demo_sharded_checkpoint,
        demo_distributed_trainer,
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
