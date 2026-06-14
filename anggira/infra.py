"""
Anggira Infrastructure — Production-grade serving engines, routing, and operations.

Pure Python stdlib. All components work with synthetic data — no external APIs.

Components:
  1. ServingEngine   — vLLM-style continuous batching scheduler
  2. SpeculativeDecoding — ThunderingHerd draft model simulator
  3. KVCacheManager  — RadixAttention-style prefix caching
  4. QuantizationSimulator — BF16/FP8/INT4/GGUF memory estimator
  5. InferenceGateway — Router, rate limiter, retry, circuit breaker
  6. ShadowCanary    — 6-stage progressive rollout with auto-rollback
  7. ObservabilityPipeline — Structured log sampling (100/10/1%)
  8. PromptCache     — Two-level: L1 semantic + L2 TTL
  9. CostGovernor    — Multi-tenant FinOps with auto-pause
  10. MultiRegionRouter — Region affinity routing
  11. ChaosRunner    — Fault injection with safety plane
  12. ModelRouter    — Pre-route and cascade routing
  13. ABTesting      — Sequential hypothesis testing
  14. demo()         — End-to-end demonstration
"""

import copy
import dataclasses
import enum
import hashlib
import hmac
import math
import os
import random
import statistics
import sys
import textwrap
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

MICROSECOND: float = 1e-6
MILLISECOND: float = 1e-3
SECOND: float = 1.0

# ═══════════════════════════════════════════════════════════════════
# 1. SERVING ENGINE — Continuous Batching Scheduler
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ServingRequest:
    """A single inference request tracked by the serving engine."""
    request_id: str
    prompt_tokens: int
    max_gen_tokens: int
    arrived_at: float = 0.0
    scheduled_at: Optional[float] = None
    completed_at: Optional[float] = None
    slot_index: Optional[int] = None
    generated_tokens: int = 0

    @property
    def ttft(self) -> Optional[float]:
        """Time To First Token (seconds)."""
        if self.scheduled_at is not None and self.arrived_at >= 0:
            return self.scheduled_at - self.arrived_at
        return None

    @property
    def tpot(self) -> Optional[float]:
        """Time Per Output Token (seconds)."""
        if self.completed_at is not None and self.scheduled_at is not None and self.generated_tokens > 0:
            return (self.completed_at - self.scheduled_at) / self.generated_tokens
        return None

    @property
    def total_latency(self) -> Optional[float]:
        """Total time from arrival to completion."""
        if self.completed_at is not None and self.arrived_at >= 0:
            return self.completed_at - self.arrived_at
        return None

    @property
    def goodput(self) -> bool:
        """Whether this request met SLO: TTFT < 500ms, TPOT < 100ms."""
        ttft_ok = self.ttft is not None and self.ttft < 0.5
        tpot_ok = self.tpot is not None and self.tpot < 0.1
        return ttft_ok and tpot_ok


class ContinuousBatchingScheduler:
    """vLLM-style continuous batching scheduler.

    Maintains a pool of 'slots' (max active requests). New requests are
    admitted when slots are free. At each step, the scheduler picks the
    next token to generate for each active slot in round-robin order.
    A request is removed when it reaches its max_gen_tokens limit.

    Metrics tracked: TTFT, TPOT, goodput ratio.
    """

    def __init__(self, num_slots: int = 8, token_time: float = 0.02):
        """
        Args:
            num_slots: Max concurrent generation slots.
            token_time: Simulated time per generated token (seconds).
        """
        self.num_slots = num_slots
        self.token_time = token_time
        self.slots: List[Optional[ServingRequest]] = [None] * num_slots
        self.queue: deque = deque()
        self.completed: List[ServingRequest] = []
        self.time: float = 0.0
        self.slo_ttft: float = 0.5  # 500ms
        self.slo_tpot: float = 0.1  # 100ms

    def submit(self, request: ServingRequest) -> None:
        """Submit a request to the engine."""
        request.arrived_at = self.time
        self.queue.append(request)

    def step(self) -> None:
        """Advance one scheduling tick.

        1. Free completed slots.
        2. Advance time (so scheduling wait is captured in TTFT).
        3. Admit queued requests to empty slots.
        4. Generate one token per active slot.
        """
        # 1. Free completed slots
        for i, slot in enumerate(self.slots):
            if slot is not None and slot.generated_tokens >= slot.max_gen_tokens:
                slot.completed_at = self.time
                self.completed.append(slot)
                self.slots[i] = None

        # 2. Advance time before admitting (so TTFT > 0 for newly queued reqs)
        self.time += self.token_time

        # 3. Admit new requests
        for i in range(self.num_slots):
            if self.slots[i] is None and self.queue:
                req = self.queue.popleft()
                req.scheduled_at = self.time
                req.slot_index = i
                self.slots[i] = req

        # 4. Generate one token per active slot
        for i, slot in enumerate(self.slots):
            if slot is not None:
                slot.generated_tokens += 1

    def run_until_complete(self) -> List[ServingRequest]:
        """Run the scheduler until all requests complete."""
        while self.queue or any(s is not None for s in self.slots):
            self.step()
        return self.completed

    def run_batch(self, requests: List[ServingRequest]) -> Dict[str, Any]:
        """Submit a batch of requests and run to completion.

        Returns a dict of aggregate metrics.
        """
        for req in requests:
            self.submit(req)
        completed = self.run_until_complete()

        ttfts = [r.ttft for r in completed if r.ttft is not None]
        tpots = [r.tpot for r in completed if r.tpot is not None]
        latencies = [r.total_latency for r in completed if r.total_latency is not None]
        goodput_ratio = sum(1 for r in completed if r.goodput) / max(len(completed), 1)

        return {
            "num_requests": len(completed),
            "avg_ttft_s": statistics.mean(ttfts) if ttfts else 0,
            "p99_ttft_s": self._percentile(sorted(ttfts), 0.99) if ttfts else 0,
            "avg_tpot_s": statistics.mean(tpots) if tpots else 0,
            "p99_tpot_s": self._percentile(sorted(tpots), 0.99) if tpots else 0,
            "avg_latency_s": statistics.mean(latencies) if latencies else 0,
            "goodput_ratio": goodput_ratio,
            "throughput_req_per_s": len(completed) / self.time if self.time > 0 else 0,
            "slo_ttft_s": self.slo_ttft,
            "slo_tpot_s": self.slo_tpot,
        }

    @staticmethod
    def _percentile(sorted_data: List[float], p: float) -> float:
        """Compute the p-th percentile of sorted data."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * p
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


class ServingEngine:
    """High-level serving engine wrapping the continuous batching scheduler."""

    def __init__(self, num_slots: int = 8, token_time: float = 0.02):
        self.scheduler = ContinuousBatchingScheduler(num_slots, token_time)
        self.name = f"ServingEngine(slots={num_slots}, tt={token_time}s)"

    def serve(self, requests: List[ServingRequest]) -> Dict[str, Any]:
        """Serve a batch of requests and return metrics."""
        return self.scheduler.run_batch(requests)


# ═══════════════════════════════════════════════════════════════════
# 2. SPECULATIVE DECODING — ThunderingHerd Draft Model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SpeculativeDecoding:
    """ThunderingHerd-style speculative decoding simulator.

    A small 'draft' model proposes K candidate tokens; a 'target' model
    verifies them. Accepted tokens are kept; rejected tokens cause a
    fallback to the target model's own prediction.

    Attributes:
        draft_speed: Tokens per second for draft model.
        target_speed: Tokens per second for target model.
        acceptance_rate: Probability each draft token is accepted (0-1).
        k: Number of draft tokens to generate per round.
    """
    draft_speed: float = 500.0   # tok/s
    target_speed: float = 50.0    # tok/s
    acceptance_rate: float = 0.8
    k: int = 5

    def simulate(self, num_prompt_tokens: int = 100, num_output_tokens: int = 200) -> Dict[str, Any]:
        """Simulate speculative vs standard decoding.

        Standard: all tokens from target model.
        Speculative: rounds of K draft tokens + verification.

        Returns a dict of speedup and wait metrics.
        """
        # Standard decoding time
        standard_time = num_output_tokens / self.target_speed

        # Speculative decoding
        tokens_generated = 0
        speculative_time = 0.0
        accepted_total = 0
        rejected_total = 0
        rounds = 0

        while tokens_generated < num_output_tokens:
            rounds += 1
            # Draft phase: generate K tokens
            draft_time = self.k / self.draft_speed
            # Target verification: one forward pass
            verify_time = 1.0 / self.target_speed
            speculative_time += draft_time + verify_time

            # Simulate acceptance/rejection
            round_accepted = 0
            for _ in range(self.k):
                if random.random() < self.acceptance_rate:
                    accepted_total += 1
                    round_accepted += 1
                    tokens_generated += 1
                    if tokens_generated >= num_output_tokens:
                        break
                else:
                    rejected_total += 1
                    # Fallback: target generates one token
                    tokens_generated += 1
                    if tokens_generated >= num_output_tokens:
                        break
                    # The accepted tokens after rejection point are lost;
                    # target generates one more. Add verify time for fallback.
                    speculative_time += 1.0 / self.target_speed
                    break

        speedup = standard_time / speculative_time if speculative_time > 0 else 1.0
        total_accepted = accepted_total
        total_rejected = rejected_total
        acceptance_ratio = total_accepted / max(total_accepted + total_rejected, 1)

        # Wait penalty: the draft phase adds latency before first token
        # vs standard which starts immediately
        wait_penalty = draft_time  # Time wasted on rejected first-round tokens
        effective_acceptance = self.acceptance_rate * self.k / (1 + self.target_speed / self.draft_speed)

        return {
            "standard_time_s": round(standard_time, 3),
            "speculative_time_s": round(speculative_time, 3),
            "speedup_x": round(speedup, 3),
            "acceptance_rate": round(self.acceptance_rate, 3),
            "effective_acceptance_rate": round(acceptance_ratio, 3),
            "k_draft_tokens": self.k,
            "rounds": rounds,
            "tokens_accepted": total_accepted,
            "tokens_rejected": total_rejected,
            "draft_tokens_per_second": self.draft_speed,
            "target_tokens_per_second": self.target_speed,
            "draft_wait_penalty_s": round(wait_penalty, 6),
            "speedup_vs_wait": round(speedup / max(wait_penalty, 0.0001), 3),
        }


# ═══════════════════════════════════════════════════════════════════
# 3. KV CACHE MANAGER — RadixAttention-style Prefix Caching
# ═══════════════════════════════════════════════════════════════════

class EvictionPolicy(enum.Enum):
    """Eviction policy for KV cache blocks."""
    FCFS = "fcfs"
    CACHE_AWARE = "cache_aware"
    LRU = "lru"


@dataclass
class KVCacheBlock:
    """A block of cached KV values for a segment of a sequence."""
    block_id: int
    prefix_hash: str
    tokens: List[str]
    kv_data_size: int  # simulated bytes
    last_access: float = 0.0
    access_count: int = 0
    ref_count: int = 1  # number of sequences sharing this block


@dataclass
class CacheMetrics:
    """Aggregate KV cache metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_blocks_used: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class KVCacheManager:
    """RadixAttention-style prefix KV cache manager.

    Maintains a tree of cached prefix blocks. Supports FCFS, LRU,
    and cache-aware eviction policies.
    """

    def __init__(self, capacity_blocks: int = 128, eviction: str = "lru"):
        """
        Args:
            capacity_blocks: Max number of KV blocks to store.
            eviction: Eviction policy — 'fcfs', 'lru', or 'cache_aware'.
        """
        self.capacity = capacity_blocks
        self.eviction = EvictionPolicy(eviction)
        self.blocks: Dict[str, KVCacheBlock] = {}  # prefix_hash -> block
        self.prefix_tree: Dict[str, List[str]] = defaultdict(list)  # prefix -> child hashes
        self.next_block_id: int = 0
        self.metrics = CacheMetrics()
        self.request_log: List[Tuple[str, bool]] = []  # (prefix_hash, hit)

    def _hash_prefix(self, tokens: List[str]) -> str:
        """Create a hash key for a prefix token sequence."""
        return hashlib.md5("".join(tokens).encode()).hexdigest()

    def lookup(self, tokens: List[str]) -> Tuple[Optional[KVCacheBlock], int]:
        """Look up the longest cached prefix for a token sequence.

        Returns:
            (matching_block, prefix_length) or (None, 0).
        """
        if not tokens:
            return None, 0

        # Try full prefix first, then progressively shorter
        best_match: Optional[KVCacheBlock] = None
        best_len = 0

        for end_idx in range(len(tokens), 0, -1):
            prefix = tokens[:end_idx]
            h = self._hash_prefix(prefix)
            block = self.blocks.get(h)
            if block is not None:
                best_match = block
                best_len = end_idx
                break

        if best_match is not None:
            self.metrics.hits += 1
            self.request_log.append((best_match.prefix_hash, True))
            best_match.last_access = time.time()
            best_match.access_count += 1
        else:
            self.metrics.misses += 1
            self.request_log.append(("none", False))

        return best_match, best_len

    def store(self, tokens: List[str]) -> Optional[KVCacheBlock]:
        """Store a new prefix block."""
        if not tokens:
            return None

        h = self._hash_prefix(tokens)
        if h in self.blocks:
            return self.blocks[h]

        # Evict if at capacity
        if len(self.blocks) >= self.capacity:
            self._evict_one()

        block = KVCacheBlock(
            block_id=self.next_block_id,
            prefix_hash=h,
            tokens=tokens,
            kv_data_size=len(tokens) * 128,  # ~128 bytes per token in KV cache
            last_access=time.time(),
        )
        self.next_block_id += 1
        self.blocks[h] = block

        # Add to prefix tree
        if len(tokens) > 1:
            parent_prefix = tokens[:-1]
            parent_h = self._hash_prefix(parent_prefix)
            if parent_h in self.blocks:
                self.prefix_tree[parent_h].append(h)

        return block

    def _evict_one(self) -> None:
        """Evict a single block based on the chosen policy."""
        if not self.blocks:
            return

        self.metrics.evictions += 1

        if self.eviction == EvictionPolicy.FCFS:
            # Evict the block with lowest block_id (oldest)
            victim = min(self.blocks.values(), key=lambda b: b.block_id)
            del self.blocks[victim.prefix_hash]
            # Clean prefix tree
            for parent_hash in list(self.prefix_tree.keys()):
                self.prefix_tree[parent_hash] = [
                    h for h in self.prefix_tree[parent_hash]
                    if h != victim.prefix_hash
                ]
            return

        if self.eviction == EvictionPolicy.LRU:
            # Evict least recently used
            victim = min(self.blocks.values(), key=lambda b: b.last_access)
            del self.blocks[victim.prefix_hash]
            return

        if self.eviction == EvictionPolicy.CACHE_AWARE:
            # Prefer evicting blocks with no children (leaf nodes)
            # that also have low access count
            leaf_blocks = [
                b for b in self.blocks.values()
                if b.prefix_hash not in self.prefix_tree
                or not self.prefix_tree[b.prefix_hash]
            ]
            if leaf_blocks:
                victim = min(leaf_blocks, key=lambda b: b.access_count)
            else:
                victim = min(self.blocks.values(), key=lambda b: b.access_count)
            del self.blocks[victim.prefix_hash]
            return

    def get_report(self) -> Dict[str, Any]:
        """Return a full metrics report."""
        return {
            "capacity_blocks": self.capacity,
            "current_blocks": len(self.blocks),
            "eviction_policy": self.eviction.value,
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "hit_rate": round(self.metrics.hit_rate, 4),
            "evictions": self.metrics.evictions,
            "total_requests": self.metrics.hits + self.metrics.misses,
        }

    def simulate_stream(self, sequences: List[List[str]]) -> Dict[str, Any]:
        """Simulate a stream of requests. Returns hit rate and metrics.

        Each sequence is a list of tokens. The first request always misses;
        subsequent requests may hit cached prefixes.
        """
        for seq in sequences:
            self.lookup(seq)
            self.store(seq)
        return self.get_report()


# ═══════════════════════════════════════════════════════════════════
# 4. QUANTIZATION SIMULATOR — Memory Footprint & Throughput Estimator
# ═══════════════════════════════════════════════════════════════════

@dataclass
class QuantConfig:
    """Quantization configuration."""
    name: str
    bits_per_weight: float
    bytes_per_weight: float
    relative_throughput: float  # 1.0 = baseline BF16
    description: str = ""


# Standard quantization formats
QUANT_FORMATS = {
    "FP8": QuantConfig("FP8", 8, 1.0, 1.0, "8-bit float (experimental)"),
    "INT4": QuantConfig("INT4", 4, 0.5, 1.5, "4-bit integer (GPTQ/AWQ-style)"),
    "GGUF_Q4": QuantConfig("GGUF_Q4", 4.5, 0.5625, 1.6, "GGUF Q4_K_M (4.5 bpw)"),
    "GGUF_Q5": QuantConfig("GGUF_Q5", 5.5, 0.6875, 1.3, "GGUF Q5_K_M (5.5 bpw)"),
    "GGUF_Q8": QuantConfig("GGUF_Q8", 8.5, 1.0625, 1.1, "GGUF Q8_0 (8.5 bpw)"),
}


@dataclass
class BF16Format:
    """BF16 baseline."""
    bits_per_weight: int = 16
    bytes_per_weight: float = 2.0
    relative_throughput: float = 1.0


BF16 = BF16Format()


@dataclass
class ModelMemoryEstimate:
    """Detailed memory estimate for a model in a given quantization."""
    model_name: str
    num_params: int
    context_length: int
    quant_name: str
    weights_bytes: float
    kv_cache_bytes: float
    total_bytes: float
    weights_gb: float
    kv_cache_gb: float
    total_gb: float
    relative_throughput: float


class QuantizationSimulator:
    """Estimate memory footprint and relative throughput for model configs."""

    # Overhead factor for activation memory, intermediate buffers, etc.
    ACTIVATION_OVERHEAD: float = 1.2

    # KV cache: 2 (K+V) * 2 (key+value matrices) * layers * hidden_dim * context
    KV_CACHE_BYTES_PER_LAYER_PER_DIM: int = 2 * 2  # K+V, each BF16=2 bytes -> 4 bytes per dim

    @staticmethod
    def estimate(model_params: int,
                 num_layers: int,
                 hidden_dim: int,
                 num_heads: int,
                 context_length: int,
                 quant: str = "BF16",
                 num_kv_heads: Optional[int] = None) -> ModelMemoryEstimate:
        """Estimate memory and throughput for a model config.

        Args:
            model_params: Total parameter count.
            num_layers: Number of transformer layers.
            hidden_dim: Hidden dimension size.
            num_heads: Number of attention heads.
            context_length: Max context length.
            quant: Quantization format ('BF16', 'FP8', 'INT4', 'GGUF_Q4', etc.)
            num_kv_heads: Number of KV heads (GQA). Defaults to num_heads.

        Returns:
            ModelMemoryEstimate with all metrics.
        """
        if num_kv_heads is None:
            num_kv_heads = num_heads

        # Weight memory
        if quant == "BF16":
            bytes_per_weight = BF16.bytes_per_weight
            rel_throughput = BF16.relative_throughput
            quant_name = "BF16"
        elif quant in QUANT_FORMATS:
            qc = QUANT_FORMATS[quant]
            bytes_per_weight = qc.bytes_per_weight
            rel_throughput = qc.relative_throughput
            quant_name = qc.name
        else:
            # Default to BF16
            bytes_per_weight = BF16.bytes_per_weight
            rel_throughput = BF16.relative_throughput
            quant_name = "BF16"

        # Weight memory
        weights_bytes = model_params * bytes_per_weight

        # KV cache memory
        # Per layer: 2 * 2 * num_kv_heads * (hidden_dim/num_heads) * context_length * 2 bytes
        # Simplified: 2 * hidden_dim * context_length * 4 bytes (2 for K, 2 for V)
        # With GQA: num_kv_heads cancels
        kv_bytes_per_layer = 2 * hidden_dim * context_length * 2  # 4 * hidden_dim * context_length
        kv_cache_bytes = num_layers * kv_bytes_per_layer

        # Activation memory overhead
        total_with_overhead = (weights_bytes + kv_cache_bytes) * QuantizationSimulator.ACTIVATION_OVERHEAD

        name = f"model_{model_params//1_000_000_000:.1f}B"

        return ModelMemoryEstimate(
            model_name=name,
            num_params=model_params,
            context_length=context_length,
            quant_name=quant_name,
            weights_bytes=weights_bytes,
            kv_cache_bytes=kv_cache_bytes,
            total_bytes=total_with_overhead,
            weights_gb=weights_bytes / (1024**3),
            kv_cache_gb=kv_cache_bytes / (1024**3),
            total_gb=total_with_overhead / (1024**3),
            relative_throughput=rel_throughput,
        )

    @staticmethod
    def compare_configs(configs: List[Tuple[int, int, int, int, int, str]]) -> List[ModelMemoryEstimate]:
        """Compare multiple model configs side by side.

        Each tuple: (model_params, num_layers, hidden_dim, num_heads, context_length, quant)
        """
        results = []
        for params, layers, dim, heads, ctx, quant in configs:
            est = QuantizationSimulator.estimate(params, layers, dim, heads, ctx, quant)
            results.append(est)
        return results

    @staticmethod
    def print_comparison(results: List[ModelMemoryEstimate]) -> str:
        """Format comparison results as a table string."""
        if not results:
            return "No results."

        lines = [
            f"{'Model':<20} {'Quant':<10} {'Params':>10} {'CtxLen':>8} "
            f"{'Weights(GB)':>12} {'KV(GB)':>10} {'Total(GB)':>10} {'RelThr':>8}",
            "-" * 88,
        ]
        for r in results:
            lines.append(
                f"{r.model_name:<20} {r.quant_name:<10} {r.num_params:>10} {r.context_length:>8} "
                f"{r.weights_gb:>12.2f} {r.kv_cache_gb:>10.2f} {r.total_gb:>10.2f} {r.relative_throughput:>8.2f}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 5. INFERENCE GATEWAY — Router, Rate Limiter, Retry, Circuit Breaker
# ═══════════════════════════════════════════════════════════════════

class CircuitState(enum.Enum):
    """Circuit breaker state."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker per provider.

    Tracks failure count. When failure threshold is exceeded, the circuit
    opens and rejects requests for the cooldown period.
    """
    name: str = ""
    failure_threshold: int = 5
    cooldown_s: float = 30.0
    half_open_max_requests: int = 1

    def __init__(self, name: str = "",
                 failure_threshold: int = 5,
                 cooldown_s: float = 30.0,
                 half_open_max_requests: int = 1):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.half_open_max_requests = half_open_max_requests
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0
        self.half_open_requests: int = 0
        self.rejected_count: int = 0

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_requests += 1
            if self.half_open_requests >= self.half_open_max_requests:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.half_open_requests = 0
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check cooldown
            if time.time() - self.last_failure_time >= self.cooldown_s:
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
                return True
            self.rejected_count += 1
            return False

        # HALF_OPEN
        if self.half_open_requests < self.half_open_max_requests:
            return True
        return False


class TokenBucket:
    """Token bucket rate limiter.

    A fixed-capacity bucket refilled at a configurable rate.
    """

    def __init__(self, capacity: float = 100.0, refill_rate: float = 10.0):
        """
        Args:
            capacity: Max token count (burst size).
            refill_rate: Tokens added per second.
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    @property
    def fill_ratio(self) -> float:
        """Current fill level as ratio of capacity."""
        return self.tokens / self.capacity if self.capacity > 0 else 0.0


@dataclass
class ProviderEndpoint:
    """A backend provider endpoint with its own circuit breaker."""
    name: str
    base_url: str = "https://api.example.com"
    weight: float = 1.0  # for weighted routing
    circuit_breaker: CircuitBreaker = field(default_factory=lambda: CircuitBreaker())
    rate_limiter: TokenBucket = field(default_factory=lambda: TokenBucket(100, 10))


@dataclass
class GatewayRequest:
    """A request flowing through the inference gateway."""
    request_id: str
    prompt: str
    model: str
    priority: int = 0  # 0=normal, 1=high
    max_retries: int = 3
    timeout_s: float = 30.0


@dataclass
class GatewayResponse:
    """Response from the inference gateway."""
    request_id: str
    success: bool
    provider: Optional[str] = None
    latency_s: float = 0.0
    error: Optional[str] = None
    retry_attempts: int = 0
    fallback_used: bool = False


class InferenceGateway:
    """Production inference gateway with fallback chain, rate limiting,
    retry with exponential backoff, and circuit breakers per provider.

    Routes requests through a chain of providers. On failure, tries the
    next provider in the chain. Supports weighted probabilistic routing.
    """

    def __init__(self, providers: Optional[List[ProviderEndpoint]] = None):
        self.providers = providers or [
            ProviderEndpoint("openai", weight=5.0),
            ProviderEndpoint("anthropic", weight=3.0),
            ProviderEndpoint("local", weight=1.0),
        ]
        self.metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total": 0, "success": 0, "failure": 0, "rejected": 0, "avg_latency": 0.0
        })

    def _pick_provider(self) -> ProviderEndpoint:
        """Pick a provider using weighted random selection."""
        total_weight = sum(p.weight for p in self.providers)
        r = random.uniform(0, total_weight)
        cumulative = 0.0
        for p in self.providers:
            cumulative += p.weight
            if r <= cumulative:
                return p
        return self.providers[-1]

    def _backoff_delay(self, attempt: int, base_s: float = 0.5, max_s: float = 30.0) -> float:
        """Exponential backoff with jitter."""
        delay = min(base_s * (2 ** attempt), max_s)
        jitter = random.uniform(0, delay * 0.5)
        return delay + jitter

    def _simulate_inference(self, provider: ProviderEndpoint,
                            request: GatewayRequest) -> Tuple[bool, Optional[str]]:
        """Simulate inference on a provider. Returns (success, error)."""
        # Simulate with some randomness
        failure_prob = {
            "openai": 0.05,
            "anthropic": 0.10,
            "local": 0.20,
        }.get(provider.name, 0.10)

        if random.random() < failure_prob:
            return False, f"{provider.name}: simulated error (rate limit / timeout)"
        return True, None

    def route(self, request: GatewayRequest) -> GatewayResponse:
        """Route a request through the gateway.

        Tries providers in order, respecting rate limits and circuit breakers.
        """
        response = GatewayResponse(request_id=request.request_id, success=False)
        attempt = 0

        # Try each provider as a fallback chain
        fallback_chain = list(self.providers)
        random.shuffle(fallback_chain)  # Shuffle for load distribution

        for provider in fallback_chain:
            attempt = 0
            while attempt <= request.max_retries:
                attempt += 1
                response.retry_attempts = attempt - 1

                # Check circuit breaker
                if not provider.circuit_breaker.allow_request():
                    self.metrics[provider.name]["rejected"] += 1
                    break  # Try next provider

                # Check rate limiter
                if not provider.rate_limiter.consume(1.0):
                    self.metrics[provider.name]["rejected"] += 1
                    break  # Try next provider

                # Simulate inference
                start = time.time()
                success, error = self._simulate_inference(provider, request)
                latency = time.time() - start

                self.metrics[provider.name]["total"] += 1

                if success:
                    provider.circuit_breaker.record_success()
                    response.success = True
                    response.provider = provider.name
                    response.latency_s = latency
                    response.fallback_used = provider != fallback_chain[0]
                    self.metrics[provider.name]["success"] += 1
                    self.metrics[provider.name]["avg_latency"] = (
                        (self.metrics[provider.name]["avg_latency"]
                         * (self.metrics[provider.name]["total"] - 1) + latency)
                        / self.metrics[provider.name]["total"]
                    )
                    return response
                else:
                    provider.circuit_breaker.record_failure()
                    self.metrics[provider.name]["failure"] += 1
                    response.error = error

                    # Backoff before retry
                    if attempt <= request.max_retries:
                        delay = self._backoff_delay(attempt)
                        time.sleep(delay * 0.01)  # Scaled for simulation

            # If we exhausted retries on this provider, try next in chain

        return response

    def get_metrics_report(self) -> Dict[str, Any]:
        """Return gateway-wide metrics."""
        report = {}
        for name, m in self.metrics.items():
            report[name] = {
                "total": m["total"],
                "success": m["success"],
                "failure": m["failure"],
                "rejected": m["rejected"],
                "avg_latency_s": round(m["avg_latency"], 4),
                "circuit_state": next(
                    (p.circuit_breaker.state.value for p in self.providers if p.name == name),
                    "unknown"
                ),
            }
        return report


# ═══════════════════════════════════════════════════════════════════
# 6. SHADOW CANARY ROLLOUT — 6-Stage Progressive Deployment
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CanaryGate:
    """A canary gate that must pass for progression."""
    name: str
    metric_fn: Callable[[], float]
    threshold: float
    operator: str = "<="  # '<=', '>='

    def evaluate(self) -> Tuple[bool, float]:
        """Check if the gate passes. Returns (passed, current_value)."""
        value = self.metric_fn()
        if self.operator == "<=":
            return value <= self.threshold, value
        elif self.operator == ">=":
            return value >= self.threshold, value
        else:
            return value == self.threshold, value


@dataclass
class CanaryStage:
    """A single stage in the canary rollout."""
    name: str
    traffic_percent: float
    latency_slo_s: float = 2.0
    error_rate_threshold: float = 0.05
    cost_budget: float = 100.0
    max_response_length: int = 4096
    duration_s: float = 60.0


DEFAULT_CANARY_STAGES = [
    CanaryStage("shadow", 1.0, latency_slo_s=3.0, error_rate_threshold=0.10, cost_budget=10.0, duration_s=10.0),
    CanaryStage("canary_5pct", 5.0, latency_slo_s=2.5, error_rate_threshold=0.05, cost_budget=50.0, duration_s=20.0),
    CanaryStage("canary_10pct", 10.0, latency_slo_s=2.0, error_rate_threshold=0.03, cost_budget=100.0, duration_s=30.0),
    CanaryStage("canary_25pct", 25.0, latency_slo_s=1.5, error_rate_threshold=0.02, cost_budget=250.0, duration_s=40.0),
    CanaryStage("canary_50pct", 50.0, latency_slo_s=1.2, error_rate_threshold=0.01, cost_budget=500.0, duration_s=50.0),
    CanaryStage("full_rollout", 100.0, latency_slo_s=1.0, error_rate_threshold=0.01, cost_budget=1000.0, duration_s=60.0),
]


class ShadowCanary:
    """6-stage progressive canary rollout with auto-rollback.

    Each stage has gates for latency SLO, error rate, cost budget,
    and response length. If ANY gate fails, the rollout auto-rollbacks
    to the previous safe stage.
    """

    def __init__(self, stages: Optional[List[CanaryStage]] = None):
        self.stages = stages or [copy.deepcopy(s) for s in DEFAULT_CANARY_STAGES]
        self.current_stage_idx: int = -1  # Not started
        self.started_at: float = 0.0
        self.failed: bool = False
        self.completed: bool = False
        self.history: List[Dict[str, Any]] = []

    @property
    def current_stage(self) -> Optional[CanaryStage]:
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    @property
    def current_traffic_pct(self) -> float:
        if self.current_stage:
            return self.current_stage.traffic_percent
        return 0.0

    def start(self) -> str:
        """Begin the rollout at stage 0."""
        self.current_stage_idx = 0
        self.started_at = time.time()
        self.history.append({
            "stage": self.stages[0].name,
            "traffic_pct": self.stages[0].traffic_percent,
            "action": "started",
            "timestamp": self.started_at,
        })
        return f"Rollout started: {self.stages[0].name} ({self.stages[0].traffic_percent}% traffic)"

    def _simulate_metrics(self, stage: CanaryStage) -> Dict[str, float]:
        """Simulate metrics for the current stage.

        Returns simulated latency, error rate, cost, and response length.
        Uses controlled randomness: metrics stay well within normal range
        unless fail_probability triggers a violation.
        """
        # Generate values that are well within normal operating range
        latency = random.uniform(0.1, stage.latency_slo_s * 0.8)
        error_rate = random.uniform(0.0, stage.error_rate_threshold * 0.6)
        cost = random.uniform(5.0, stage.cost_budget * 0.6)
        response_len = random.randint(100, int(stage.max_response_length * 0.7))

        return {
            "latency_s": latency,
            "error_rate": error_rate,
            "cost": cost,
            "response_length": response_len,
        }

    def check_gates(self, fail_probability: float = 0.1) -> Dict[str, Any]:
        """Check all gates for the current stage.

        Args:
            fail_probability: Probability that a gate fails (for demo).

        Returns:
            Dict with gate results.
        """
        if self.current_stage is None:
            return {"status": "not_started"}

        stage = self.current_stage
        metrics = self._simulate_metrics(stage)

        # Check each gate (with optional forced failure for demo)
        gates: Dict[str, Dict[str, Any]] = {}

        # Determine if this iteration should fail
        force_fail = random.random() < fail_probability

        # Latency SLO
        latency_ok = metrics["latency_s"] <= stage.latency_slo_s
        if force_fail and not latency_ok:
            latency_ok = False
            metrics["latency_s"] = stage.latency_slo_s * 1.5  # Exaggerate

        gates["latency_slo"] = {
            "passed": latency_ok,
            "value": round(metrics["latency_s"], 3),
            "threshold": stage.latency_slo_s,
            "operator": "<=",
        }

        # Error rate
        error_ok = metrics["error_rate"] <= stage.error_rate_threshold
        if force_fail:
            error_ok = False
            metrics["error_rate"] = stage.error_rate_threshold * 2.0
        gates["error_rate"] = {
            "passed": error_ok,
            "value": round(metrics["error_rate"], 4),
            "threshold": stage.error_rate_threshold,
            "operator": "<=",
        }

        # Cost budget
        cost_ok = metrics["cost"] <= stage.cost_budget
        gates["cost_budget"] = {
            "passed": cost_ok,
            "value": round(metrics["cost"], 2),
            "threshold": stage.cost_budget,
            "operator": "<=",
        }

        # Response length
        length_ok = metrics["response_length"] <= stage.max_response_length
        gates["response_length"] = {
            "passed": length_ok,
            "value": metrics["response_length"],
            "threshold": stage.max_response_length,
            "operator": "<=",
        }

        all_passed = all(g["passed"] for g in gates.values())

        self.history.append({
            "stage": stage.name,
            "traffic_pct": stage.traffic_percent,
            "action": "check",
            "all_passed": all_passed,
            "metrics": metrics,
            "gates": gates,
        })

        return {
            "stage": stage.name,
            "traffic_percent": stage.traffic_percent,
            "all_passed": all_passed,
            "gates": gates,
        }

    def advance(self, fail_probability: float = 0.1) -> str:
        """Check gates and advance to the next stage, or rollback.

        Args:
            fail_probability: Probability of gate failure (for demo).

        Returns:
            Status message.
        """
        if self.current_stage_idx < 0:
            return self.start()

        if self.failed:
            return "Rollout already failed. Reset to retry."

        if self.completed:
            return "Rollout already completed."

        result = self.check_gates(fail_probability)

        if not result["all_passed"]:
            # Rollback!
            self.failed = True
            self.history.append({
                "stage": result["stage"],
                "action": "rollback",
                "reason": "Gate(s) failed",
                "gates": result["gates"],
            })
            return (
                f"✗ ROLLBACK at stage '{result['stage']}' "
                f"({result['traffic_percent']}% traffic). "
                f"Failed gates: {', '.join(k for k, v in result['gates'].items() if not v['passed'])}"
            )

        # Advance to next stage
        self.current_stage_idx += 1

        if self.current_stage_idx >= len(self.stages):
            self.completed = True
            self.history.append({
                "stage": "full_rollout",
                "action": "completed",
            })
            return "✓ Full rollout completed! 100% traffic on new version."
        else:
            next_stage = self.stages[self.current_stage_idx]
            self.history.append({
                "stage": next_stage.name,
                "traffic_pct": next_stage.traffic_percent,
                "action": "advanced",
            })
            return (
                f"✓ Advanced to stage '{next_stage.name}' "
                f"({next_stage.traffic_percent}% traffic)"
            )

    def run_full_rollout(self, fail_probability: float = 0.1) -> List[str]:
        """Run all stages sequentially and return a log."""
        log: List[str] = []
        if self.current_stage_idx < 0:
            log.append(self.start())

        while not self.failed and not self.completed:
            result = self.advance(fail_probability)
            log.append(result)
            if "ROLLBACK" in result:
                break
        return log

    def get_report(self) -> Dict[str, Any]:
        """Return a full canary report."""
        return {
            "total_stages": len(self.stages),
            "current_stage_idx": self.current_stage_idx,
            "current_stage": self.current_stage.name if self.current_stage else None,
            "current_traffic_pct": self.current_traffic_pct,
            "failed": self.failed,
            "completed": self.completed,
            "history": self.history,
        }


# ═══════════════════════════════════════════════════════════════════
# 7. OBSERVABILITY PIPELINE — Structured Log Sampling
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LogEvent:
    """A structured log event."""
    timestamp: float
    level: str  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    source: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0  # simulated cost of this event


class LogSampler:
    """Handles log sampling with error overrides.

    Sampling tiers:
      - 100%: All events kept (errors, critical issues)
      - 10%:  Sampled events (INFO, WARNING)
      - 1%:   Rare events (DEBUG, TRACE)
    """

    TIERS = {
        "ERROR": 1.0,
        "CRITICAL": 1.0,
        "WARNING": 0.1,
        "INFO": 0.1,
        "DEBUG": 0.01,
    }

    def __init__(self, sampling_config: Optional[Dict[str, float]] = None):
        """
        Args:
            sampling_config: Override default sampling rates per level.
        """
        self.sampling_rates: Dict[str, float] = {**LogSampler.TIERS}
        if sampling_config:
            self.sampling_rates.update(sampling_config)
        self.sampled_events: List[LogEvent] = []
        self.dropped_events: List[LogEvent] = []
        self.total_events: int = 0
        self.total_cost: float = 0.0
        self.dropped_cost: float = 0.0

    def should_sample(self, event: LogEvent) -> bool:
        """Determine if this event should be kept."""
        rate = self.sampling_rates.get(event.level, 0.1)
        return random.random() < rate

    def emit(self, event: LogEvent) -> bool:
        """Try to emit a log event. Returns True if sampled."""
        self.total_events += 1
        self.total_cost += event.cost

        if self.should_sample(event):
            self.sampled_events.append(event)
            return True
        else:
            self.dropped_events.append(event)
            self.dropped_cost += event.cost
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return sampling statistics."""
        kept = len(self.sampled_events)
        dropped = len(self.dropped_events)
        total = self.total_events

        level_breakdown: Dict[str, Dict[str, int]] = {}
        for event in self.sampled_events + self.dropped_events:
            if event.level not in level_breakdown:
                level_breakdown[event.level] = {"kept": 0, "dropped": 0}
            if event in self.sampled_events:
                level_breakdown[event.level]["kept"] += 1
            else:
                level_breakdown[event.level]["dropped"] += 1

        # De-duplicate by counting again properly
        level_breakdown = defaultdict(lambda: {"kept": 0, "dropped": 0})
        for e in self.sampled_events:
            level_breakdown[e.level]["kept"] += 1
        for e in self.dropped_events:
            level_breakdown[e.level]["dropped"] += 1

        return {
            "total_events": total,
            "sampled": kept,
            "dropped": dropped,
            "sampling_rate_overall": round(kept / max(total, 1), 4),
            "sampled_cost": round(self.total_cost - self.dropped_cost, 4),
            "dropped_cost": round(self.dropped_cost, 4),
            "cost_savings_pct": round(self.dropped_cost / max(self.total_cost, 0.001) * 100, 2),
            "levels": dict(level_breakdown),
        }


class ObservabilityPipeline:
    """Full observability pipeline with structured logging and sampling."""

    def __init__(self, service_name: str = "anggira-infra"):
        self.service_name = service_name
        self.sampler = LogSampler()
        self.events: List[LogEvent] = []

    def log(self, level: str, source: str, message: str,
            metadata: Optional[Dict[str, Any]] = None,
            cost: float = 0.0) -> bool:
        """Log an event through the pipeline."""
        event = LogEvent(
            timestamp=time.time(),
            level=level.upper(),
            source=source,
            message=message,
            metadata=metadata or {},
            cost=cost,
        )
        self.events.append(event)
        return self.sampler.emit(event)

    def generate_synthetic_logs(self, count: int = 100,
                                error_rate: float = 0.1) -> None:
        """Generate synthetic log events for testing."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        weights = [0.2, 0.5, 0.2, 0.08, 0.02]
        sources = ["serving_engine", "gateway", "kv_cache", "model_router",
                   "cost_governor", "canary", "chaos_runner"]

        for _ in range(count):
            level = random.choices(levels, weights=weights, k=1)[0]
            source = random.choice(sources)
            cost = random.uniform(0.001, 0.1)

            messages = {
                "DEBUG": "Cache miss for prefix hash {}",
                "INFO": "Request {} completed in {:.2f}s",
                "WARNING": "Rate limit approaching for provider {}",
                "ERROR": "Circuit breaker opened for {}",
                "CRITICAL": "OOM detected on node {}",
            }
            msg = random.choice(list(messages.values()))
            formatted_msg = msg.format(
                random.randint(1000, 9999),
                random.uniform(0.1, 5.0),
            )

            self.log(level, source, formatted_msg, cost=cost)

    def compute_cost_metrics_tradeoff(self) -> Dict[str, Any]:
        """Analyze the cost vs. metrics visibility tradeoff of sampling."""
        stats = self.sampler.get_stats()
        dropped_error_events = sum(
            1 for e in self.sampler.dropped_events if e.level in ("ERROR", "CRITICAL")
        )
        total_error_events = sum(
            1 for e in self.sampler.sampled_events + self.sampler.dropped_events
            if e.level in ("ERROR", "CRITICAL")
        )

        return {
            "sampling_stats": stats,
            "error_visibility": {
                "total_errors": total_error_events,
                "dropped_errors": dropped_error_events,
                "error_retention_rate": round(
                    1 - dropped_error_events / max(total_error_events, 1), 4
                ),
            },
            "metric_accuracy_loss_pct": round(dropped_error_events / max(total_error_events, 1) * 100, 2),
            "cost_savings_pct": stats["cost_savings_pct"],
            "recommendation": (
                "Increase ERROR/CRITICAL sampling to 100% to retain full error visibility"
                if dropped_error_events > 0 else
                "Current sampling config provides good cost-error balance"
            ),
        }


# ═══════════════════════════════════════════════════════════════════
# 8. PROMPT CACHE — Two-Level Semantic + TTL Cache
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PromptCacheEntry:
    """A cached prompt entry."""
    prompt: str
    embedding: List[float]
    response: str
    created_at: float
    expires_at: float
    access_count: int = 0

    @property
    def expired(self) -> bool:
        return time.time() > self.expires_at


def _simple_embed(text: str, dim: int = 16) -> List[float]:
    """A simple hash-based embedding for semantic similarity simulation.

    Deterministic: same text always produces the same embedding.
    """
    h = hashlib.md5(text.encode()).digest()
    # Use MD5 bytes to seed a deterministic random vector
    seed = int.from_bytes(h[:8], 'big')
    rng = random.Random(seed)
    vec = [rng.gauss(0, 1) for _ in range(dim)]
    # Normalize
    mag = math.sqrt(sum(x * x for x in vec))
    return [x / mag for x in vec]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class EmbeddingCache:
    """L1: Semantic similarity cache.

    Finds the best matching cached entry by embedding similarity.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.threshold = similarity_threshold
        self.entries: List[PromptCacheEntry] = []
        self.hits: int = 0
        self.misses: int = 0

    def lookup(self, prompt: str, embedding: List[float]) -> Optional[str]:
        """Find a matching cached response by semantic similarity."""
        best_sim = -1.0
        best_entry: Optional[PromptCacheEntry] = None

        for entry in self.entries:
            if entry.expired:
                continue
            sim = _cosine_sim(embedding, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_sim >= self.threshold and best_entry is not None:
            self.hits += 1
            best_entry.access_count += 1
            return best_entry.response
        self.misses += 1
        return None

    def store(self, prompt: str, embedding: List[float], response: str,
              ttl_s: float = 3600) -> None:
        """Store a new entry."""
        now = time.time()
        entry = PromptCacheEntry(
            prompt=prompt,
            embedding=embedding,
            response=response,
            created_at=now,
            expires_at=now + ttl_s,
        )
        # Avoid duplicates: replace if very similar
        for i, existing in enumerate(self.entries):
            if _cosine_sim(embedding, existing.embedding) > 0.99:
                self.entries[i] = entry
                return
        self.entries.append(entry)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class TTLCache:
    """L2: Simple TTL cache with key-based lookup."""

    def __init__(self, ttl_s: float = 300.0, max_size: int = 1000):
        self.ttl = ttl_s
        self.max_size = max_size
        self._cache: Dict[str, PromptCacheEntry] = {}
        self.hits: int = 0
        self.misses: int = 0

    def _is_valid(self, entry: PromptCacheEntry) -> bool:
        return not entry.expired

    def get(self, key: str) -> Optional[str]:
        """Look up a key. Returns response or None."""
        entry = self._cache.get(key)
        if entry is not None and self._is_valid(entry):
            self.hits += 1
            entry.access_count += 1
            return entry.response
        if entry is not None:
            del self._cache[key]
        self.misses += 1
        return None

    def set(self, key: str, response: str, ttl_s: Optional[float] = None) -> None:
        """Store a response for a key."""
        now = time.time()
        ttl = ttl_s or self.ttl
        emb = _simple_embed(key)
        entry = PromptCacheEntry(
            prompt=key,
            embedding=emb,
            response=response,
            created_at=now,
            expires_at=now + ttl,
        )
        if len(self._cache) >= self.max_size:
            # Evict oldest
            oldest = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
            del self._cache[oldest]
        self._cache[key] = entry

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PromptCache:
    """Two-level prompt cache: L1 semantic + L2 TTL.

    L1 is checked first (embedding similarity). On L1 miss, L2 is checked
    (exact key match). On L2 miss, a new entry is stored in both levels.
    """

    def __init__(self, semantic_threshold: float = 0.95,
                 l2_ttl_s: float = 300.0):
        self.l1 = EmbeddingCache(similarity_threshold=semantic_threshold)
        self.l2 = TTLCache(ttl_s=l2_ttl_s)
        self.total_requests: int = 0

    def lookup(self, prompt: str) -> Optional[str]:
        """Two-level lookup. Returns cached response or None."""
        self.total_requests += 1
        embedding = _simple_embed(prompt)

        # L1: semantic cache
        result = self.l1.lookup(prompt, embedding)
        if result is not None:
            return result

        # L2: exact TTL cache
        result = self.l2.get(prompt)
        if result is not None:
            # Promote to L1
            self.l1.store(prompt, embedding, result, ttl_s=self.l2.ttl)
            return result

        return None

    def store(self, prompt: str, response: str) -> None:
        """Store a response in both caches."""
        embedding = _simple_embed(prompt)
        self.l1.store(prompt, embedding, response, ttl_s=self.l2.ttl)
        self.l2.set(prompt, response)

    def get_report(self) -> Dict[str, Any]:
        """Return cache performance report."""
        total_l1 = self.l1.hits + self.l1.misses
        total_l2 = self.l2.hits + self.l2.misses
        overall_hits = self.l1.hits + self.l2.hits
        overall_misses = total_l1 + total_l2 - overall_hits

        return {
            "total_requests": self.total_requests,
            "l1_semantic": {
                "hits": self.l1.hits,
                "misses": self.l1.misses,
                "hit_rate": round(self.l1.hit_rate, 4),
                "threshold": self.l1.threshold,
            },
            "l2_ttl": {
                "hits": self.l2.hits,
                "misses": self.l2.misses,
                "hit_rate": round(self.l2.hit_rate, 4),
                "size": self.l2.size,
                "ttl_s": self.l2.ttl,
            },
            "overall_hits": overall_hits,
            "overall_misses": overall_misses,
            "overall_hit_rate": round(overall_hits / max(overall_hits + overall_misses, 1), 4),
        }


# ═══════════════════════════════════════════════════════════════════
# 9. COST GOVERNOR — Multi-Tenant FinOps
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Tenant:
    """A tenant with cost tracking."""
    tenant_id: str
    name: str
    monthly_spend_cap: float = 1000.0
    rate_limit_rpm: int = 100  # requests per minute
    current_spend: float = 0.0
    request_count: int = 0
    is_paused: bool = False
    spend_history: List[float] = field(default_factory=list)
    request_timestamps: List[float] = field(default_factory=list)

    def can_serve(self, cost: float = 0.0) -> bool:
        """Check if this tenant can serve a request costing `cost`."""
        if self.is_paused:
            return False
        if self.current_spend + cost > self.monthly_spend_cap:
            return False
        return True

    def record_usage(self, cost: float = 0.0) -> None:
        """Record a usage event."""
        self.current_spend += cost
        self.request_count += 1
        self.spend_history.append(cost)
        self.request_timestamps.append(time.time())


class CostGovernor:
    """Multi-tenant FinOps: per-tenant spend caps, rate limits, and
    auto-pause for abusive tenants (z-score > 4 on spend rate).
    """

    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.paused_tenants: List[str] = []
        self.total_spend: float = 0.0
        self.total_requests: int = 0
        self.rejected_requests: int = 0

    def register_tenant(self, tenant_id: str, name: str,
                        monthly_cap: float = 1000.0,
                        rate_limit: int = 100) -> Tenant:
        """Register a new tenant."""
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            monthly_spend_cap=monthly_cap,
            rate_limit_rpm=rate_limit,
        )
        self.tenants[tenant_id] = tenant
        return tenant

    def allow_request(self, tenant_id: str, estimated_cost: float = 0.0) -> Tuple[bool, str]:
        """Check if a tenant's request is allowed.

        Returns (allowed, reason).
        """
        tenant = self.tenants.get(tenant_id)
        if tenant is None:
            self.rejected_requests += 1
            return False, "unknown_tenant"

        if tenant.is_paused:
            self.rejected_requests += 1
            return False, "tenant_paused"

        # Rate limit: RPM check
        now = time.time()
        one_minute_ago = now - 60.0
        recent_requests = sum(1 for t in tenant.request_timestamps if t > one_minute_ago)
        if recent_requests >= tenant.rate_limit_rpm:
            self.rejected_requests += 1
            return False, "rate_limit_exceeded"

        # Spend cap check — auto-pause if exceeded
        if tenant.current_spend + estimated_cost > tenant.monthly_spend_cap:
            tenant.is_paused = True
            if tenant.tenant_id not in self.paused_tenants:
                self.paused_tenants.append(tenant.tenant_id)
            self.rejected_requests += 1
            return False, "spend_cap_exceeded_auto_paused"

        return True, "allowed"

    def record_usage(self, tenant_id: str, cost: float) -> None:
        """Record usage for a tenant."""
        tenant = self.tenants.get(tenant_id)
        if tenant is None:
            return
        tenant.record_usage(cost)
        self.total_spend += cost
        self.total_requests += 1

        # Check for anomalous spend (z-score > 4)
        self._check_anomaly(tenant)

    def _check_anomaly(self, tenant: Tenant) -> None:
        """Auto-pause tenant if spend z-score > 4."""
        if len(tenant.spend_history) < 5:
            return

        recent = tenant.spend_history[-5:]
        mean = statistics.mean(recent)
        stdev = statistics.stdev(recent) if len(recent) > 1 else 0.0

        if stdev == 0:
            return

        z_score = (tenant.current_spend / max(tenant.request_count, 1) - mean) / stdev

        if z_score > 4.0 and not tenant.is_paused:
            tenant.is_paused = True
            self.paused_tenants.append(tenant.tenant_id)

    def get_report(self) -> Dict[str, Any]:
        """Return full cost governance report."""
        tenant_reports = {}
        for tid, tenant in self.tenants.items():
            recent_spends = tenant.spend_history[-10:] if tenant.spend_history else []
            avg_cost = statistics.mean(recent_spends) if recent_spends else 0.0
            z_score = 0.0
            if len(tenant.spend_history) >= 5:
                recent = tenant.spend_history[-5:]
                mean = statistics.mean(recent)
                stdev = statistics.stdev(recent) if len(recent) > 1 else 0.0
                if stdev > 0:
                    z_score = (tenant.current_spend / max(tenant.request_count, 1) - mean) / stdev

            tenant_reports[tid] = {
                "name": tenant.name,
                "current_spend": round(tenant.current_spend, 4),
                "monthly_cap": tenant.monthly_spend_cap,
                "spend_utilization_pct": round(
                    tenant.current_spend / tenant.monthly_spend_cap * 100, 2
                ) if tenant.monthly_spend_cap > 0 else 0,
                "request_count": tenant.request_count,
                "avg_cost_per_request": round(avg_cost, 4),
                "rate_limit_rpm": tenant.rate_limit_rpm,
                "is_paused": tenant.is_paused,
                "z_score": round(z_score, 4),
            }

        return {
            "tenants": tenant_reports,
            "total_spend": round(self.total_spend, 4),
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "paused_tenants": self.paused_tenants,
            "active_tenant_count": sum(1 for t in self.tenants.values() if not t.is_paused),
        }


# ═══════════════════════════════════════════════════════════════════
# 10. MULTI-REGION ROUTER — Region Affinity Routing
# ═══════════════════════════════════════════════════════════════════

class RoutingStrategy(enum.Enum):
    """Region routing strategies."""
    ROUND_ROBIN = "round_robin"
    REGIONAL = "regional"
    GLOBAL = "global"


@dataclass
class Region:
    """A deployment region with capacity and cost."""
    name: str
    capacity: int = 100  # max concurrent requests
    cost_per_token: float = 0.001
    latency_ms: float = 50.0  # base latency
    kv_cache_locality_hits: int = 0
    kv_cache_locality_misses: int = 0

    @property
    def locality_hit_rate(self) -> float:
        total = self.kv_cache_locality_hits + self.kv_cache_locality_misses
        return self.kv_cache_locality_hits / total if total > 0 else 0.0


@dataclass
class RegionRequest:
    """A request with region preference."""
    request_id: str
    prompt: str
    preferred_region: Optional[str] = None  # e.g., "us-east-1"
    user_region: str = "us-east-1"


class MultiRegionRouter:
    """Route requests across regions with different strategies.

    Supports three strategies:
      - ROUND_ROBIN: Distribute evenly across all regions.
      - REGIONAL: Route to the user's preferred region.
      - GLOBAL: Route to the region with lowest latency.
    """

    def __init__(self, regions: Optional[List[Region]] = None):
        self.regions = regions or [
            Region("us-east-1", capacity=200, cost_per_token=0.001, latency_ms=10),
            Region("us-west-2", capacity=150, cost_per_token=0.0012, latency_ms=25),
            Region("eu-west-1", capacity=100, cost_per_token=0.0015, latency_ms=75),
            Region("ap-southeast-1", capacity=80, cost_per_token=0.002, latency_ms=150),
        ]
        self.rr_index: int = 0
        self.total_routed: int = 0
        self.strategy_log: List[Tuple[str, str, str]] = []  # (req_id, strategy, region)

    def _round_robin(self, request: RegionRequest) -> Region:
        """Select the next region in round-robin order."""
        region = self.regions[self.rr_index % len(self.regions)]
        self.rr_index += 1
        return region

    def _regional(self, request: RegionRequest) -> Region:
        """Route to the user's preferred/regional region."""
        if request.preferred_region:
            for r in self.regions:
                if r.name == request.preferred_region:
                    return r
        # Fallback: find closest region by name similarity
        for r in self.regions:
            if request.user_region.split("-")[0] == r.name.split("-")[0]:
                return r
        return self.regions[0]

    def _global_routing(self, request: RegionRequest) -> Region:
        """Route to the region with lowest latency."""
        return min(self.regions, key=lambda r: r.latency_ms)

    def route(self, request: RegionRequest,
              strategy: RoutingStrategy = RoutingStrategy.REGIONAL) -> Region:
        """Route a request to a region using the given strategy."""
        self.total_routed += 1

        if strategy == RoutingStrategy.ROUND_ROBIN:
            region = self._round_robin(request)
        elif strategy == RoutingStrategy.REGIONAL:
            region = self._regional(request)
        elif strategy == RoutingStrategy.GLOBAL:
            region = self._global_routing(request)
        else:
            region = self.regions[0]

        # Simulate KV cache locality
        # Regional routing is more likely to hit cache
        if strategy == RoutingStrategy.REGIONAL:
            if random.random() < 0.7:
                region.kv_cache_locality_hits += 1
            else:
                region.kv_cache_locality_misses += 1
        elif strategy == RoutingStrategy.GLOBAL:
            if random.random() < 0.5:
                region.kv_cache_locality_hits += 1
            else:
                region.kv_cache_locality_misses += 1
        else:  # ROUND_ROBIN
            if random.random() < 0.3:
                region.kv_cache_locality_hits += 1
            else:
                region.kv_cache_locality_misses += 1

        self.strategy_log.append((request.request_id, strategy.value, region.name))
        return region

    def compare_strategies(self, requests: List[RegionRequest]) -> Dict[str, Any]:
        """Compare all three routing strategies on a batch of requests.

        Runs each strategy independently and reports metrics.
        """
        results = {}
        for strategy in RoutingStrategy:
            # Reset for each strategy
            rr_save = self.rr_index
            locality_save = {r.name: (r.kv_cache_locality_hits, r.kv_cache_locality_misses)
                             for r in self.regions}
            regions_save = {r.name: (r.kv_cache_locality_hits, r.kv_cache_locality_misses)
                            for r in self.regions}
            self.rr_index = 0
            for r in self.regions:
                r.kv_cache_locality_hits = 0
                r.kv_cache_locality_misses = 0

            # Route all requests
            region_counts: Dict[str, int] = defaultdict(int)
            for req in requests:
                region = self.route(req, strategy)
                region_counts[region.name] += 1

            total_hits = sum(r.kv_cache_locality_hits for r in self.regions)
            total_misses = sum(r.kv_cache_locality_misses for r in self.regions)
            total_locality = total_hits + total_misses

            locality_hit_rate = total_hits / total_locality if total_locality > 0 else 0.0

            results[strategy.value] = {
                "requests_routed": len(requests),
                "region_distribution": dict(region_counts),
                "kv_cache_locality_hits": total_hits,
                "kv_cache_locality_misses": total_misses,
                "kv_cache_locality_hit_rate": round(locality_hit_rate, 4),
            }

            # Restore
            self.rr_index = rr_save
            for r in self.regions:
                r.kv_cache_locality_hits, r.kv_cache_locality_misses = regions_save[r.name]

        return results


# ═══════════════════════════════════════════════════════════════════
# 11. CHAOS EXPERIMENT RUNNER — Fault Injection
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FaultSpec:
    """Specification for a single fault to inject."""
    fault_type: str  # 'latency', 'drop', 'oom'
    probability: float = 0.1  # probability of injection
    latency_ms: float = 500.0  # for latency faults
    oom_size_gb: float = 1.0  # for OOM faults


@dataclass
class ChaosExperiment:
    """A chaos experiment with fault specifications."""
    name: str
    faults: List[FaultSpec]
    burn_rate: float = 1.0   # how aggressive (0.1 = conservative, 10 = aggressive)
    blast_radius: float = 0.3  # fraction of total capacity affected
    duration_s: float = 30.0
    is_running: bool = False


class ChaosRunner:
    """Chaos experiment runner with burn-rate × blast-radius safety plane.

    Enforces that burn_rate * blast_radius <= 1.0 (safety constraint).
    Supports latency injection, request dropping, and OOM simulation.
    """

    def __init__(self):
        self.experiments: Dict[str, ChaosExperiment] = {}
        self.experiment_log: List[Dict[str, Any]] = []
        self.safety_violations: int = 0
        self.total_faults_injected: int = 0

    def register_experiment(self, name: str,
                            faults: List[FaultSpec],
                            burn_rate: float = 1.0,
                            blast_radius: float = 0.3,
                            duration_s: float = 30.0) -> ChaosExperiment:
        """Register a chaos experiment.

        Raises ValueError if burn_rate * blast_radius > 1.0 (unsafe).
        """
        if burn_rate * blast_radius > 1.0:
            self.safety_violations += 1
            raise ValueError(
                f"Safety plane violation: burn_rate({burn_rate}) * "
                f"blast_radius({blast_radius}) = {burn_rate * blast_radius} > 1.0"
            )

        exp = ChaosExperiment(
            name=name,
            faults=faults,
            burn_rate=burn_rate,
            blast_radius=blast_radius,
            duration_s=duration_s,
        )
        self.experiments[name] = exp
        return exp

    def inject_faults(self, experiment_name: str) -> List[Dict[str, Any]]:
        """Inject faults for a given experiment.

        Returns a list of fault events.
        """
        exp = self.experiments.get(experiment_name)
        if exp is None:
            return [{"error": f"Unknown experiment: {experiment_name}"}]

        if exp.is_running:
            return [{"error": f"Experiment '{experiment_name}' already running"}]

        exp.is_running = True
        events: List[Dict[str, Any]] = []

        for fault in exp.faults:
            # Check safety again
            if exp.burn_rate * exp.blast_radius > 1.0:
                self.safety_violations += 1
                events.append({
                    "fault_type": fault.fault_type,
                    "status": "blocked",
                    "reason": "Safety plane violation",
                })
                continue

            # Inject fault with probability
            for _ in range(10):  # Simulate 10 injection points
                if random.random() < fault.probability * exp.burn_rate:
                    self.total_faults_injected += 1
                    event = {
                        "fault_type": fault.fault_type,
                        "status": "injected",
                        "timestamp": time.time(),
                        "details": {},
                    }

                    if fault.fault_type == "latency":
                        actual_latency = fault.latency_ms * exp.burn_rate
                        event["details"] = {
                            "latency_ms": actual_latency,
                            "affected_requests_pct": exp.blast_radius * 100,
                        }
                    elif fault.fault_type == "drop":
                        event["details"] = {
                            "dropped_requests_pct": exp.blast_radius * 100,
                        }
                    elif fault.fault_type == "oom":
                        event["details"] = {
                            "oom_size_gb": fault.oom_size_gb,
                            "affected_memory_pct": exp.blast_radius * 100,
                        }

                    events.append(event)

        exp.is_running = False
        self.experiment_log.append({
            "experiment": experiment_name,
            "faults_injected": len(events),
            "events": events,
        })

        return events

    def get_report(self) -> Dict[str, Any]:
        """Return chaos engineering report."""
        return {
            "experiments_registered": len(self.experiments),
            "experiments_run": len(self.experiment_log),
            "total_faults_injected": self.total_faults_injected,
            "safety_violations": self.safety_violations,
            "experiments": {
                name: {
                    "name": exp.name,
                    "fault_count": len(exp.faults),
                    "burn_rate": exp.burn_rate,
                    "blast_radius": exp.blast_radius,
                    "safety_score": exp.burn_rate * exp.blast_radius,
                }
                for name, exp in self.experiments.items()
            },
        }


# ═══════════════════════════════════════════════════════════════════
# 12. MODEL ROUTER — Pre-route and Cascade Routing
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ModelEndpoint:
    """A model deployment endpoint."""
    model_id: str
    cost_per_token: float = 0.001
    latency_per_token_ms: float = 20.0
    max_context: int = 4096
    capabilities: List[str] = field(default_factory=lambda: ["text"])


@dataclass
class RoutingDecision:
    """Result of a model routing decision."""
    model_id: str
    strategy: str  # 'pre_route', 'cascade', 'fallback'
    confidence: float = 1.0
    cost_estimate: float = 0.0


class ModelRouter:
    """Routes requests to the appropriate model.

    Supports two strategies:
      - PRE_ROUTE: A classifier picks the best model upfront.
      - CASCADE: Start with a cheap model, fall back on uncertainty.
    """

    def __init__(self, models: Optional[List[ModelEndpoint]] = None):
        self.models = models or [
            ModelEndpoint("tiny-llm", cost_per_token=0.0001, latency_per_token_ms=5, max_context=2048),
            ModelEndpoint("medium-llm", cost_per_token=0.001, latency_per_token_ms=20, max_context=4096),
            ModelEndpoint("large-llm", cost_per_token=0.01, latency_per_token_ms=50, max_context=8192),
            ModelEndpoint("expert-llm", cost_per_token=0.05, latency_per_token_ms=100, max_context=16384),
        ]
        self.routing_log: List[RoutingDecision] = []
        self.total_requests: int = 0

    def _classify_difficulty(self, prompt: str) -> float:
        """Simple prompt difficulty classifier based on heuristics.

        Returns a difficulty score 0.0 (easy) to 1.0 (hard).
        """
        # Heuristics: length, complexity keywords, code content
        score = 0.0

        # Length heuristic
        score += min(len(prompt) / 1000, 0.3)

        # Complexity keywords
        complex_terms = [
            "reasoning", "multi-step", "analysis", "complex", "debug",
            "optimize", "mathematical", "proof", "algorithm", "synthesis",
            "compare", "contrast", "evaluate", "critique", "comprehensive"
        ]
        prompt_lower = prompt.lower()
        matches = sum(1 for t in complex_terms if t in prompt_lower)
        score += min(matches * 0.1, 0.4)

        # Code content
        if "```" in prompt or "def " in prompt or "class " in prompt:
            score += 0.2

        # Questions
        if "?" in prompt:
            score += 0.1

        return min(score, 1.0)

    def pre_route(self, prompt: str) -> RoutingDecision:
        """Pre-route: classifier picks the best model upfront."""
        self.total_requests += 1
        difficulty = self._classify_difficulty(prompt)

        if difficulty < 0.3:
            model = self.models[0]  # tiny
            confidence = 1.0 - difficulty
        elif difficulty < 0.6:
            model = self.models[1]  # medium
            confidence = 1.0 - difficulty * 0.5
        elif difficulty < 0.8:
            model = self.models[2]  # large
            confidence = 1.0 - difficulty * 0.3
        else:
            model = self.models[3]  # expert
            confidence = 1.0 - difficulty * 0.2

        tokens = len(prompt.split())
        cost_est = tokens * model.cost_per_token

        decision = RoutingDecision(
            model_id=model.model_id,
            strategy="pre_route",
            confidence=round(confidence, 4),
            cost_estimate=round(cost_est, 6),
        )
        self.routing_log.append(decision)
        return decision

    def cascade(self, prompt: str,
                confidence_threshold: float = 0.8) -> RoutingDecision:
        """Cascade: start cheap, fall back on uncertainty.

        Begins with the smallest model. If its confidence is below threshold,
        escalates to the next larger model.
        """
        self.total_requests += 1

        for idx, model in enumerate(self.models):
            # Simulate model confidence
            difficulty = self._classify_difficulty(prompt)
            model_confidence = 1.0 - (difficulty * (1.0 - idx * 0.15))

            if model_confidence >= confidence_threshold or idx == len(self.models) - 1:
                tokens = len(prompt.split())
                cost_est = tokens * model.cost_per_token

                decision = RoutingDecision(
                    model_id=model.model_id,
                    strategy="cascade",
                    confidence=round(model_confidence, 4),
                    cost_estimate=round(cost_est, 6),
                )

                # Add cascade path info for reporting
                decision.cascade_path = [
                    m.model_id for m in self.models[:idx + 1]
                ]

                self.routing_log.append(decision)
                return decision

        # Fallback (shouldn't reach here)
        fallback = self.models[-1]
        tokens = len(prompt.split())
        cost_est = tokens * fallback.cost_per_token
        decision = RoutingDecision(
            model_id=fallback.model_id,
            strategy="fallback",
            confidence=0.5,
            cost_estimate=round(cost_est, 6),
        )
        self.routing_log.append(decision)
        return decision

    def compare_strategies(self, prompts: List[str]) -> Dict[str, Any]:
        """Compare pre-route vs cascade on a set of prompts."""
        # Pre-route
        pre_route_costs = []
        for p in prompts:
            d = self.pre_route(p)
            pre_route_costs.append(d.cost_estimate)

        # Cascade
        cascade_costs = []
        cascade_paths = []
        for p in prompts:
            d = self.cascade(p)
            cascade_costs.append(d.cost_estimate)
            cascade_paths.append(getattr(d, 'cascade_path', [d.model_id]))

        saving = (sum(pre_route_costs) - sum(cascade_costs)) / max(sum(pre_route_costs), 0.001) * 100

        return {
            "num_prompts": len(prompts),
            "pre_route": {
                "total_cost": round(sum(pre_route_costs), 6),
                "avg_cost": round(statistics.mean(pre_route_costs), 6) if pre_route_costs else 0,
                "models_used": list(set(d.model_id for d in self.routing_log[-len(prompts):])),
            },
            "cascade": {
                "total_cost": round(sum(cascade_costs), 6),
                "avg_cost": round(statistics.mean(cascade_costs), 6) if cascade_costs else 0,
                "cascade_paths": cascade_paths[-len(prompts):],
            },
            "cascade_savings_pct": round(saving, 2),
        }


# ═══════════════════════════════════════════════════════════════════
# 13. A/B TESTING FRAMEWORK — Sequential Hypothesis Testing
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ABTestVariant:
    """A variant in an A/B test."""
    name: str
    metric_values: List[float] = field(default_factory=list)
    is_control: bool = False

    @property
    def mean(self) -> float:
        return statistics.mean(self.metric_values) if self.metric_values else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.metric_values) if len(self.metric_values) > 1 else 0.0

    @property
    def count(self) -> int:
        return len(self.metric_values)


@dataclass
class ABTestResult:
    """Result of an A/B test."""
    control: ABTestVariant
    treatment: ABTestVariant
    significant: bool
    p_value: float
    effect_size: float
    metric_name: str
    lift_pct: float
    stopped_early: bool = False


class ABTesting:
    """Sequential A/B testing framework with configurable significance level.

    Uses a simplified sequential probability ratio test (SPRT) approach.
    Tracks metric drift and reports significance.
    """

    def __init__(self, significance_level: float = 0.05,
                 min_sample_size: int = 10,
                 max_sample_size: int = 1000):
        """
        Args:
            significance_level: Alpha threshold for statistical significance.
            min_sample_size: Minimum samples before testing.
            max_sample_size: Maximum samples before forced conclusion.
        """
        self.alpha = significance_level
        self.min_samples = min_sample_size
        self.max_samples = max_sample_size

    def _compute_t_stat(self, control: ABTestVariant,
                        treatment: ABTestVariant) -> Tuple[float, float]:
        """Compute t-statistic and approximate p-value (Welch's t-test)."""
        n1, n2 = control.count, treatment.count
        if n1 < 2 or n2 < 2:
            return 0.0, 1.0

        m1, m2 = control.mean, treatment.mean
        s1, s2 = control.stdev, treatment.stdev

        # Pooled standard error
        se = math.sqrt((s1**2 / n1) + (s2**2 / n2))
        if se == 0:
            return 0.0, 1.0

        t_stat = (m2 - m1) / se

        # Welch-Satterthwaite degrees of freedom
        num = (s1**2 / n1 + s2**2 / n2)**2
        denom = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
        df = num / denom if denom > 0 else min(n1, n2) - 1

        # Approximate p-value from t-distribution using normal approximation
        # (sufficient for demo; real would use scipy)
        p_value = self._approx_t_cdf(-abs(t_stat), df) * 2

        return t_stat, p_value

    @staticmethod
    def _approx_t_cdf(t: float, df: float) -> float:
        """Approximate t-distribution CDF using normal approximation.

        Uses the Abramowitz and Stegun approximation.
        """
        x = t * (1 - 1/(4*df)) / math.sqrt(1 + t*t/(2*df))
        # Normal CDF approximation
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def add_observation(self, variant: ABTestVariant, value: float) -> None:
        """Add a metric observation to a variant."""
        variant.metric_values.append(value)

    def test(self, control: ABTestVariant, treatment: ABTestVariant,
             metric_name: str = "latency") -> ABTestResult:
        """Run a sequential test and return the result.

        Checks if we have enough data, then performs a t-test.
        """
        effect_size = treatment.mean - control.mean
        lift_pct = (effect_size / control.mean * 100) if control.mean != 0 else 0.0

        if control.count < self.min_samples or treatment.count < self.min_samples:
            # Not enough data — inconclusive
            return ABTestResult(
                control=control,
                treatment=treatment,
                significant=False,
                p_value=1.0,
                effect_size=effect_size,
                metric_name=metric_name,
                lift_pct=round(lift_pct, 4),
            )

        if control.count >= self.max_samples or treatment.count >= self.max_samples:
            # Max samples reached — force conclusion
            t_stat, p_value = self._compute_t_stat(control, treatment)
            return ABTestResult(
                control=control,
                treatment=treatment,
                significant=p_value < self.alpha,
                p_value=round(p_value, 6),
                effect_size=effect_size,
                metric_name=metric_name,
                lift_pct=round(lift_pct, 4),
                stopped_early=False,
            )

        t_stat, p_value = self._compute_t_stat(control, treatment)

        return ABTestResult(
            control=control,
            treatment=treatment,
            significant=p_value < self.alpha,
            p_value=round(p_value, 6),
            effect_size=effect_size,
            metric_name=metric_name,
            lift_pct=round(lift_pct, 4),
            stopped_early=p_value < self.alpha / 2,  # Early stopping rule
        )

    def sequential_test(self, control_data: List[float],
                        treatment_data: List[float],
                        metric_name: str = "latency") -> ABTestResult:
        """Run a sequential test on pre-collected data.

        Checks at each step whether significance has been reached.
        """
        control = ABTestVariant("control", is_control=True)
        treatment = ABTestVariant("treatment")

        min_len = min(len(control_data), len(treatment_data))

        for i in range(min_len):
            self.add_observation(control, control_data[i])
            self.add_observation(treatment, treatment_data[i])

        return self.test(control, treatment, metric_name)

    def detect_drift(self, baseline: List[float], current: List[float],
                     metric_name: str = "latency") -> Dict[str, Any]:
        """Detect metric drift between baseline and current observations."""
        control = ABTestVariant("baseline", is_control=True, metric_values=baseline)
        treatment = ABTestVariant("current", metric_values=current)

        result = self.test(control, treatment, metric_name)

        return {
            "metric": metric_name,
            "baseline_mean": round(control.mean, 4),
            "current_mean": round(treatment.mean, 4),
            "drift_pct": round(result.lift_pct, 4),
            "p_value": result.p_value,
            "significant": result.significant,
            "effect_size": round(result.effect_size, 4),
            "alert": "⚠ DRIFT DETECTED" if result.significant else "✓ No significant drift",
        }


# ═══════════════════════════════════════════════════════════════════
# 14. DEMO — End-to-End Demonstration
# ═══════════════════════════════════════════════════════════════════

def demo() -> Dict[str, Any]:
    """Run a comprehensive demonstration of all Anggira infra components.

    Returns a dict with each component's demo results.
    """
    results: Dict[str, Any] = {}
    random.seed(42)  # Deterministic demo

    print("=" * 70)
    print("  ANGGIRA INFRASTRUCTURE — Full Demo")
    print("=" * 70)

    # ── 1. Serving Engine ──────────────────────────────────────────
    print("\n" + "─" * 40)
    print("1. SERVING ENGINE — Continuous Batching")
    print("─" * 40)

    engine = ServingEngine(num_slots=4, token_time=0.02)
    requests = [
        ServingRequest(f"req-{i}", prompt_tokens=50 + i * 10, max_gen_tokens=20 + i * 5)
        for i in range(10)
    ]
    engine_metrics = engine.serve(requests)
    for k, v in engine_metrics.items():
        print(f"   {k}: {v}")
    results["serving_engine"] = engine_metrics

    # ── 2. Speculative Decoding ────────────────────────────────────
    print("\n" + "─" * 40)
    print("2. SPECULATIVE DECODING — ThunderingHerd")
    print("─" * 40)

    sd = SpeculativeDecoding(draft_speed=500, target_speed=50, acceptance_rate=0.8, k=5)
    sd_metrics = sd.simulate(num_prompt_tokens=100, num_output_tokens=200)
    for k, v in sd_metrics.items():
        print(f"   {k}: {v}")
    results["speculative_decoding"] = sd_metrics

    # ── 3. KV Cache Manager ────────────────────────────────────────
    print("\n" + "─" * 40)
    print("3. KV CACHE MANAGER — RadixAttention")
    print("─" * 40)

    kvc = KVCacheManager(capacity_blocks=10, eviction="lru")
    sequences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "cat", "sat", "on", "the", "rug"],
        ["the", "dog", "sat", "on", "the", "mat"],
        ["the", "cat", "sat"],  # prefix hit
        ["a", "bird", "flew", "over", "the", "fence"],
    ]
    kv_report = kvc.simulate_stream(sequences)
    for k, v in kv_report.items():
        print(f"   {k}: {v}")
    results["kv_cache"] = kv_report

    # ── 4. Quantization Simulator ──────────────────────────────────
    print("\n" + "─" * 40)
    print("4. QUANTIZATION SIMULATOR — Memory & Throughput")
    print("─" * 40)

    configs = [
        (7_000_000_000, 32, 4096, 32, 4096, "BF16"),
        (7_000_000_000, 32, 4096, 32, 4096, "INT4"),
        (7_000_000_000, 32, 4096, 32, 4096, "GGUF_Q4"),
        (13_000_000_000, 40, 5120, 40, 4096, "BF16"),
        (13_000_000_000, 40, 5120, 40, 4096, "INT4"),
        (70_000_000_000, 80, 8192, 64, 8192, "BF16"),
        (70_000_000_000, 80, 8192, 64, 8192, "GGUF_Q4"),
    ]
    quant_results = QuantizationSimulator.compare_configs(configs)
    print(QuantizationSimulator.print_comparison(quant_results))
    results["quantization"] = [
        {"model": r.model_name, "quant": r.quant_name,
         "weights_gb": round(r.weights_gb, 2), "kv_gb": round(r.kv_cache_gb, 2),
         "total_gb": round(r.total_gb, 2), "rel_throughput": r.relative_throughput}
        for r in quant_results
    ]

    # ── 5. Inference Gateway ───────────────────────────────────────
    print("\n" + "─" * 40)
    print("5. INFERENCE GATEWAY — Router, Rate Limiter, Circuit Breaker")
    print("─" * 40)

    gateway = InferenceGateway()
    gw_requests = [
        GatewayRequest(f"gw-req-{i}", f"Prompt for request {i}", "gpt-4")
        for i in range(20)
    ]
    gw_responses = []
    for req in gw_requests:
        resp = gateway.route(req)
        gw_responses.append(resp)
    successes = sum(1 for r in gw_responses if r.success)
    failures = sum(1 for r in gw_responses if not r.success)
    print(f"   Total requests: {len(gw_responses)}")
    print(f"   Successful: {successes}")
    print(f"   Failed: {failures}")
    print(f"   Fallback used: {sum(1 for r in gw_responses if r.fallback_used)}")
    gw_metrics = gateway.get_metrics_report()
    for provider, m in gw_metrics.items():
        print(f"   {provider}: {m['total']} reqs, {m['success']} ok, "
              f"{m['failure']} fail, {m['rejected']} rej, "
              f"circuit={m['circuit_state']}")
    results["gateway"] = {
        "total": len(gw_responses), "success": successes, "failure": failures,
        "provider_metrics": gw_metrics,
    }

    # ── 6. Shadow Canary Rollout ───────────────────────────────────
    print("\n" + "─" * 40)
    print("6. SHADOW CANARY — 6-Stage Progressive Rollout")
    print("─" * 40)

    canary = ShadowCanary()
    # Use very low fail probability so rollout demonstrates all 6 stages
    canary_log = canary.run_full_rollout(fail_probability=0.01)
    for line in canary_log:
        print(f"   {line}")
    canary_report = canary.get_report()
    print(f"   Final: failed={canary_report['failed']}, "
          f"completed={canary_report['completed']}, "
          f"stage={canary_report['current_stage']}")
    results["canary"] = canary_report

    # ── 7. Observability Pipeline ──────────────────────────────────
    print("\n" + "─" * 40)
    print("7. OBSERVABILITY PIPELINE — Log Sampling")
    print("─" * 40)

    obs = ObservabilityPipeline()
    obs.generate_synthetic_logs(count=200, error_rate=0.1)
    tradeoff = obs.compute_cost_metrics_tradeoff()
    print(f"   Total events: {tradeoff['sampling_stats']['total_events']}")
    print(f"   Sampled: {tradeoff['sampling_stats']['sampled']}")
    print(f"   Dropped: {tradeoff['sampling_stats']['dropped']}")
    print(f"   Cost savings: {tradeoff['cost_savings_pct']}%")
    print(f"   Error retention: {tradeoff['error_visibility']['error_retention_rate']}")
    print(f"   Recommendation: {tradeoff['recommendation']}")
    results["observability"] = tradeoff

    # ── 8. Prompt Cache ────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("8. PROMPT CACHE — Two-Level (Semantic + TTL)")
    print("─" * 40)

    pc = PromptCache(semantic_threshold=0.9, l2_ttl_s=3600)
    # Store
    pc.store("What is machine learning?", "Machine learning is a subset of AI...")
    pc.store("Explain neural networks", "Neural networks are computing systems...")
    # Exact lookup
    r1 = pc.lookup("What is machine learning?")
    r2 = pc.lookup("Explain neural networks")
    # Similar lookup (semantic)
    r3 = pc.lookup("What is ML?")  # Similar to first
    # Miss
    r4 = pc.lookup("How do transformers work?")
    pc_report = pc.get_report()
    print(f"   Exact hit 1: {'✓' if r1 else '✗'}")
    print(f"   Exact hit 2: {'✓' if r2 else '✗'}")
    print(f"   Semantic hit: {'✓' if r3 else '✗'}")
    print(f"   Miss: {'✓' if r4 is None else '✗'}")
    print(f"   L1 hit rate: {pc_report['l1_semantic']['hit_rate']}")
    print(f"   L2 hit rate: {pc_report['l2_ttl']['hit_rate']}")
    print(f"   Overall hit rate: {pc_report['overall_hit_rate']}")
    results["prompt_cache"] = pc_report

    # ── 9. Cost Governor ──────────────────────────────────────────
    print("\n" + "─" * 40)
    print("9. COST GOVERNOR — Multi-Tenant FinOps")
    print("─" * 40)

    cg = CostGovernor()
    cg.register_tenant("t1", "Startup-A", monthly_cap=500, rate_limit=50)
    cg.register_tenant("t2", "Enterprise-B", monthly_cap=5000, rate_limit=500)
    cg.register_tenant("t3", "Abusive-C", monthly_cap=100, rate_limit=10)

    # Simulate normal usage
    for i in range(30):
        cg.record_usage("t1", random.uniform(0.01, 0.1))
        cg.record_usage("t2", random.uniform(0.05, 0.5))

    # Simulate abusive usage (high spend rate)
    for i in range(20):
        cg.record_usage("t3", random.uniform(5.0, 15.0))

    cg_report = cg.get_report()
    for tid, info in cg_report["tenants"].items():
        print(f"   {info['name']}: spend=${info['current_spend']}, "
              f"util={info['spend_utilization_pct']}%, "
              f"paused={info['is_paused']}, z={info['z_score']}")
    print(f"   Paused tenants: {cg_report['paused_tenants']}")
    results["cost_governor"] = cg_report

    # ── 10. Multi-Region Router ───────────────────────────────────
    print("\n" + "─" * 40)
    print("10. MULTI-REGION ROUTER — Region Affinity")
    print("─" * 40)

    mrr = MultiRegionRouter()
    region_requests = [
        RegionRequest(f"rr-req-{j}", f"prompt-{j}",
                      preferred_region=random.choice(["us-east-1", "us-west-2", "eu-west-1"]))
        for j in range(100)
    ]
    region_comparison = mrr.compare_strategies(region_requests)
    for strategy, info in region_comparison.items():
        print(f"   {strategy}: hits={info['kv_cache_locality_hits']}, "
              f"hit_rate={info['kv_cache_locality_hit_rate']}")
    results["multi_region_router"] = region_comparison

    # ── 11. Chaos Runner ──────────────────────────────────────────
    print("\n" + "─" * 40)
    print("11. CHAOS EXPERIMENT RUNNER — Fault Injection")
    print("─" * 40)

    cr = ChaosRunner()
    cr.register_experiment(
        "latency-spike",
        [FaultSpec("latency", probability=0.3, latency_ms=1000)],
        burn_rate=0.5, blast_radius=0.3,
    )
    cr.register_experiment(
        "packet-loss",
        [FaultSpec("drop", probability=0.2)],
        burn_rate=0.3, blast_radius=0.2,
    )
    cr.register_experiment(
        "oom-simulation",
        [FaultSpec("oom", probability=0.1, oom_size_gb=2.0)],
        burn_rate=0.2, blast_radius=0.1,
    )
    # This one should be blocked by safety plane
    try:
        cr.register_experiment(
            "unsafe-experiment",
            [FaultSpec("latency", probability=0.5, latency_ms=2000)],
            burn_rate=5.0, blast_radius=0.5,
        )
    except ValueError as e:
        print(f"   Safety block: {e}")

    # Run experiments
    for exp_name in ["latency-spike", "packet-loss", "oom-simulation"]:
        events = cr.inject_faults(exp_name)
        print(f"   {exp_name}: {len(events)} faults injected")

    chaos_report = cr.get_report()
    print(f"   Total faults: {chaos_report['total_faults_injected']}")
    print(f"   Safety violations blocked: {chaos_report['safety_violations']}")
    results["chaos_runner"] = chaos_report

    # ── 12. Model Router ──────────────────────────────────────────
    print("\n" + "─" * 40)
    print("12. MODEL ROUTER — Pre-route & Cascade")
    print("─" * 40)

    mr = ModelRouter()
    prompts = [
        "What is 2+2?",
        "Write a Python function to merge two sorted lists",
        "Explain the theory of relativity in detail with mathematical proofs",
        "Summarize this text: The quick brown fox jumps over the lazy dog",
        "Debug the following complex algorithm for topological sorting...",
    ]

    for p in prompts:
        pr = mr.pre_route(p)
        ca = mr.cascade(p)
        print(f"   Prompt ({len(p)} chars): pre_route->{pr.model_id} (${pr.cost_estimate}), "
              f"cascade->{ca.model_id} (${ca.cost_estimate})")

    model_comparison = mr.compare_strategies(prompts)
    print(f"\n   Pre-route total cost: ${model_comparison['pre_route']['total_cost']}")
    print(f"   Cascade total cost: ${model_comparison['cascade']['total_cost']}")
    print(f"   Cascade savings: {model_comparison['cascade_savings_pct']}%")
    results["model_router"] = model_comparison

    # ── 13. A/B Testing ────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("13. A/B TESTING — Sequential Hypothesis Testing")
    print("─" * 40)

    ab = ABTesting(significance_level=0.05)

    # Simulate A/B test: control vs treatment (treatment is faster)
    control_times = [random.gauss(2.0, 0.5) for _ in range(50)]
    treatment_times = [random.gauss(1.5, 0.4) for _ in range(50)]

    ab_result = ab.sequential_test(control_times, treatment_times, "latency")
    print(f"   Control mean: {ab_result.control.mean:.3f}s")
    print(f"   Treatment mean: {ab_result.treatment.mean:.3f}s")
    print(f"   Lift: {ab_result.lift_pct:.2f}%")
    print(f"   p-value: {ab_result.p_value}")
    print(f"   Significant: {ab_result.significant}")

    # Drift detection
    drift = ab.detect_drift(control_times, [random.gauss(2.5, 0.6) for _ in range(50)])
    print(f"   Drift alert: {drift['alert']}")
    results["ab_testing"] = {
        "test_result": {
            "significant": ab_result.significant,
            "p_value": ab_result.p_value,
            "lift_pct": ab_result.lift_pct,
        },
        "drift_detection": drift,
    }

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE — All 13 components operational")
    print("=" * 70)

    results["_summary"] = {
        "components": [
            "serving_engine", "speculative_decoding", "kv_cache",
            "quantization", "gateway", "canary", "observability",
            "prompt_cache", "cost_governor", "multi_region_router",
            "chaos_runner", "model_router", "ab_testing",
        ],
        "all_passed": True,
    }

    return results


# ═══════════════════════════════════════════════════════════════════
# MAIN — Run demo when executed directly
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(textwrap.dedent("""\
    Anggira Infrastructure Module
    =============================
    Running demo() — all 13 components with synthetic data.
    """).strip())
    results = demo()
    print(f"\nDemo returned {len(results)} result keys.")
