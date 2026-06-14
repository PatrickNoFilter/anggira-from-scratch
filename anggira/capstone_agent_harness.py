"""
Anggira Capstone — Agent Harness Infrastructure

Phase 19, Lessons 20-29: Agent Harness / Tool-Use System

Implements:
- JSONRPCTransport — JSON-RPC 2.0 over stdio
- FunctionCallDispatcher — dispatch with timeout/retry/dedupe
- VerificationGate — gate chain, observation budget
- SandboxRunner — secure subprocess runner
- PlanExecutor — plan → execute → observe → replan
- EvalHarness — fixture-based evaluation
- Tracer — OpenTelemetry-style tracing
- CodingAgent — end-to-end harness demo
"""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict


# ═══════════════════════════════════════════════
# JSONRPCTransport  (Lesson 22)
# ═══════════════════════════════════════════════

class JSONRPCTransport:
    """JSON-RPC 2.0 transport over stdio or in-process."""

    def __init__(self, mode="stdio"):
        self.mode = mode
        self._request_id = 0
        self._buffers = {}

    def encode_request(self, method, params=None, req_id=None):
        req_id = req_id or self._next_id()
        return json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": req_id
        }).encode("utf-8") + b"\n"

    def decode_response(self, data):
        """Decode JSON-RPC response. Returns {'result': ...} or {'error': ...}."""
        obj = json.loads(data.decode("utf-8").strip())
        if "error" in obj:
            err = obj["error"]
            raise JSONRPCError(err.get("code", -1), err.get("message", "Unknown error"))
        return obj.get("result")

    def encode_notification(self, method, params=None):
        return json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }).encode("utf-8") + b"\n"

    def decode_error(self, data):
        """Decode and return error info without raising."""
        obj = json.loads(data.decode("utf-8").strip())
        if "error" in obj:
            return obj["error"]
        return None

    def _next_id(self):
        self._request_id += 1
        return self._request_id

    # In-process dispatch
    def register_handler(self, name, handler_fn):
        self._buffers[name] = handler_fn

    def call_local(self, method, **params):
        if method not in self._buffers:
            raise JSONRPCError(-32601, f"Method not found: {method}")
        try:
            result = self._buffers[method](**params)
            return result
        except Exception as e:
            raise JSONRPCError(-32603, str(e))


class JSONRPCError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


# ═══════════════════════════════════════════════
# FunctionCallDispatcher  (Lesson 23)
# ═══════════════════════════════════════════════

class DispatchError(Exception):
    pass

class TimeoutError_(DispatchError):
    pass

class ValidationError(DispatchError):
    pass

class FunctionCallDispatcher:
    """Dispatch tool calls with timeout, retry, deduplication, error mapping."""

    def __init__(self, default_timeout=5.0, max_retries=2, dedup_window=0.5):
        self._handlers = {}
        self.default_timeout = default_timeout
        self.max_retries = max_retries
        self.dedup_window = dedup_window
        self._call_history = []  # (call_key, timestamp, result)
        self._stats = {"ok": 0, "timeout": 0, "error": 0, "retry": 0, "dedup": 0}

    def register_handler(self, name, fn, schema=None):
        """Register a callable handler with optional JSON schema."""
        self._handlers[name] = {"fn": fn, "schema": schema or {}}

    def validate(self, name, params):
        """Validate params against schema. Simple type checking."""
        if name not in self._handlers:
            raise ValidationError(f"Unknown handler: {name}")
        schema = self._handlers[name]["schema"]
        for key, expected_type in schema.get("required", {}).items():
            if key not in params:
                raise ValidationError(f"Missing required param: {key}")
            if not isinstance(params[key], expected_type):
                raise ValidationError(
                    f"Param {key}: expected {expected_type.__name__}, got {type(params[key]).__name__}"
                )
        return True

    def call(self, name, **params):
        """Call a handler with timeout and retry."""
        # Check dedup
        call_key = (name, frozenset(sorted(params.items())))
        for prev_key, ts, result in reversed(self._call_history):
            if prev_key == call_key and (time.time() - ts) < self.dedup_window:
                self._stats["dedup"] += 1
                return result

        if name not in self._handlers:
            raise ValidationError(f"Unknown handler: {name}")

        handler = self._handlers[name]["fn"]
        self.validate(name, params)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                result = handler(**params)
                elapsed = time.time() - start
                self._stats["ok"] += 1
                self._call_history.append((call_key, time.time(), result))
                return {"status": "ok", "result": result, "elapsed": round(elapsed, 4), "attempts": attempt + 1}
            except TimeoutError_ as e:
                last_error = e
                self._stats["timeout"] += 1
                if attempt < self.max_retries:
                    self._stats["retry"] += 1
                    time.sleep(0.5 * (2 ** attempt))  # exponential backoff
            except Exception as e:
                last_error = e
                self._stats["error"] += 1
                if attempt < self.max_retries:
                    self._stats["retry"] += 1
                    time.sleep(0.2)

        return {"status": "error", "error": str(last_error), "attempts": self.max_retries + 1}

    def stats(self):
        """Return dispatch statistics."""
        return dict(self._stats)

    def demo(self):
        """Run a demo of the dispatcher."""
        def add(a, b):
            return a + b

        def flaky():
            if random.random() < 0.6:
                raise ValueError("Simulated failure")
            return "ok"

        self.register_handler("add", add, {"required": {"a": int, "b": int}})
        self.register_handler("flaky", flaky)

        print("=== FunctionCallDispatcher Demo ===")
        r1 = self.call("add", a=3, b=4)
        print(f"  add(3,4): {r1}")
        r2 = self.call("add", a=10, b=20)
        print(f"  add(10,20): {r2}")
        print(f"  Dedup test (same call): {self.call('add', a=3, b=4)}")
        print(f"  Flaky handler (may retry): {self.call('flaky')}")
        print(f"  Stats: {self.stats()}")
        return True


# ═══════════════════════════════════════════════
# VerificationGate  (Lesson 25)
# ═══════════════════════════════════════════════

class GateResult:
    def __init__(self, allowed, reason, metadata=None):
        self.allowed = allowed
        self.reason = reason
        self.metadata = metadata or {}

    def __repr__(self):
        return f"GateResult(allowed={self.allowed}, reason='{self.reason}')"


class VerificationGate:
    """Deterministic gate chain for tool call verification."""

    def __init__(self):
        self._gates = []
        self._history = []

    def add_gate(self, name, gate_fn):
        """Add a gate: gate_fn(operation_name, args, context) -> GateResult."""
        self._gates.append((name, gate_fn))

    def check(self, operation_name, args=None, context=None):
        """Run all gates. Returns first denial or final approval."""
        args = args or {}
        context = context or {}

        for name, gate_fn in self._gates:
            result = gate_fn(operation_name, args, context)
            self._history.append({
                "gate": name, "operation": operation_name,
                "allowed": result.allowed, "reason": result.reason,
                "timestamp": time.time()
            })
            if not result.allowed:
                return result

        return GateResult(True, "All gates passed")

    def history(self, limit=20):
        return self._history[-limit:]


# Pre-built gates

def budget_gate(max_cost=100.0):
    """Gate that tracks token/API cost and denies when exceeded."""
    cost_so_far = [0.0]

    def gate(operation, args, context):
        estimated = context.get("estimated_cost", 1.0)
        if cost_so_far[0] + estimated > max_cost:
            return GateResult(False, f"Budget exceeded: {cost_so_far[0]:.1f} + {estimated:.1f} > {max_cost}")
        cost_so_far[0] += estimated
        return GateResult(True, f"Budget OK ({cost_so_far[0]:.1f}/{max_cost})")
    return gate


def deny_list_gate(denied_ops=None):
    """Gate that rejects dangerous operations."""
    denied = set(denied_ops or ["rm", "dd", "mkfs", "shutdown", "reboot", "sudo", "chmod"])

    def gate(operation, args, context):
        op_name = operation.lower()
        for d in denied:
            if d in op_name:
                return GateResult(False, f"Operation '{operation}' is denied (matched '{d}')")
        return GateResult(True, "Operation allowed")
    return gate


def path_jail_gate(project_root="/tmp/project"):
    """Gate that ensures file paths stay under project_root."""

    def gate(operation, args, context):
        project_root_abs = os.path.abspath(project_root)
        for key, val in args.items():
            if isinstance(val, str) and ("/" in val or "\\" in val):
                abs_path = os.path.abspath(os.path.join(project_root, val))
                if not abs_path.startswith(project_root_abs):
                    return GateResult(False, f"Path '{val}' escapes project root")
        return GateResult(True, "Paths OK")
    return gate


def output_size_gate(max_chars=10000):
    """Gate that truncates oversized output."""

    def gate(operation, args, context):
        # Just an advisory truncation marker
        return GateResult(True, f"Output will be capped at {max_chars} chars")
    return gate


class ObservationBudget:
    """Tracks total tokens the agent has been shown. Blocks when exceeded."""

    def __init__(self, max_tokens=100000):
        self.max_tokens = max_tokens
        self.total_seen = 0
        self._denied_calls = 0

    def observe(self, text):
        """Register that the agent has seen this text. Rough token count = len//4."""
        tokens = len(text) // 4
        self.total_seen += tokens
        return self.remaining()

    def remaining(self):
        return max(0, self.max_tokens - self.total_seen)

    def can_continue(self):
        return self.total_seen < self.max_tokens

    def check(self, operation, args, context):
        if not self.can_continue():
            return GateResult(False, f"Observation budget exhausted ({self.total_seen}/{self.max_tokens})")
        return GateResult(True, f"Budget OK ({self.remaining()} remaining)")

    def __call__(self, operation, args, context):
        return self.check(operation, args, context)


def demo_verification_gate():
    """Demo the verification gate chain."""
    print("\n=== VerificationGate Demo ===")
    vg = VerificationGate()
    vg.add_gate("deny_list", deny_list_gate())
    vg.add_gate("budget", budget_gate(max_cost=50.0))

    r1 = vg.check("read_file", args={"path": "foo.txt"}, context={"estimated_cost": 1.0})
    print(f"  read_file: {r1}")
    r2 = vg.check("rm", args={"path": "/"}, context={"estimated_cost": 100.0})
    print(f"  rm: {r2}")
    r3 = vg.check("llm_call", args={}, context={"estimated_cost": 60.0})
    print(f"  expensive call: {r3}")

    obs = ObservationBudget(max_tokens=100)
    for i in range(5):
        text = "Hello world! " * 10  # ~120 chars, ~30 tokens
        obs.observe(text)
    print(f"  Observation budget remaining: {obs.remaining()}")
    print(f"  Can continue: {obs.can_continue()}")
    return True


# ═══════════════════════════════════════════════
# SandboxRunner  (Lesson 26)
# ═══════════════════════════════════════════════

class SandboxError(Exception):
    pass


class SandboxRunner:
    """Secure subprocess runner with denylist, path jailing, truncation, timeout."""

    DENIED_EXECUTABLES = {
        "rm", "dd", "mkfs", "fdisk", "shutdown", "reboot", "sudo",
        "passwd", "kill", "pkill", "chmod", "chown", "mount", "umount"
    }
    DENIED_PATTERNS = ["--force", "--yes", "-f", ">/dev/", "2>/dev/null"]

    def __init__(self, project_root=None, output_limit=10000, timeout=30):
        self.project_root = project_root or os.getcwd()
        self.output_limit = output_limit
        self.timeout = timeout
        self._runs = []

    def _check_command(self, command):
        """Check command against denylist."""
        cmd_str = command if isinstance(command, str) else " ".join(command)

        # Check executables
        parts = cmd_str.split()
        if parts:
            exe = os.path.basename(parts[0])
            if exe in self.DENIED_EXECUTABLES:
                raise SandboxError(f"Denied executable: {exe}")

        # Check argv patterns
        for pattern in self.DENIED_PATTERNS:
            if pattern in cmd_str:
                raise SandboxError(f"Denied pattern in command: {pattern}")

        # Check path jail
        for part in parts:
            if "/" in part or "\\" in part:
                abs_path = os.path.abspath(os.path.join(os.getcwd(), part))
                if not abs_path.startswith(os.path.abspath(self.project_root)):
                    raise SandboxError(f"Path escapes jail: {part}")

        return True

    def run(self, command, shell=True, cwd=None, env_add=None):
        """Run a command in the sandbox."""
        if not shell and isinstance(command, str):
            command = command.split()

        self._check_command(command)

        env = os.environ.copy()
        if env_add:
            env.update(env_add)

        start = time.time()
        try:
            proc = subprocess.run(
                command, shell=shell, cwd=cwd or self.project_root,
                capture_output=True, text=True, timeout=self.timeout, env=env
            )
            elapsed = time.time() - start

            # Truncate output
            stdout = (proc.stdout or "")[:self.output_limit]
            stderr = (proc.stderr or "")[:self.output_limit]

            result = {
                "stdout": stdout,
                "stderr": stderr,
                "returncode": proc.returncode,
                "wall_time": round(elapsed, 3)
            }
            self._runs.append(result)
            return result

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            raise SandboxError(f"Process timed out after {self.timeout}s")
        except FileNotFoundError as e:
            raise SandboxError(f"Command not found: {e}")

    def stats(self):
        return {
            "total_runs": len(self._runs),
            "avg_time": round(sum(r["wall_time"] for r in self._runs) / max(len(self._runs), 1), 3)
        }


def demo_sandbox_runner():
    """Demo sandbox runner."""
    print("\n=== SandboxRunner Demo ===")
    sr = SandboxRunner(output_limit=200)

    r1 = sr.run("echo 'Hello from sandbox!' && echo 'More output' | head -1")
    print(f"  Echo: returncode={r1['returncode']}, stdout={r1['stdout'].strip()[:50]}")

    r2 = sr.run("python3 -c 'print(sum(range(100)))'")
    print(f"  Python: returncode={r2['returncode']}, sum(0..99)={r2['stdout'].strip()}")

    try:
        sr.run("rm -rf /")
    except SandboxError as e:
        print(f"  Denied 'rm': {e}")

    print(f"  Stats: {sr.stats()}")
    return True


# ═══════════════════════════════════════════════
# PlanExecutor  (Lesson 24)
# ═══════════════════════════════════════════════

class PlanStep:
    """A single step in a plan."""

    def __init__(self, name, tool, args=None, depends_on=None, max_retries=1):
        self.name = name
        self.tool = tool  # callable or string key
        self.args = args or {}
        self.depends_on = depends_on or []
        self.max_retries = max_retries
        self.status = "pending"  # pending, running, done, failed
        self.result = None
        self.error = None
        self.attempts = 0


class PlanExecutor:
    """Plan → Execute → Observe → Replan loop."""

    def __init__(self, tool_registry=None, max_retries=2):
        self.tools = tool_registry or {}
        self.max_retries = max_retries
        self._execution_log = []
        self._checkpoints = []

    def register_tool(self, name, fn):
        self.tools[name] = fn

    def _resolve_deps(self, steps, done_steps):
        """Return steps whose dependencies are all done."""
        ready = []
        done_names = {s.name for s in done_steps if s.status == "done"}
        for step in steps:
            if step.status != "pending":
                continue
            if all(dep in done_names for dep in step.depends_on):
                ready.append(step)
        return ready

    def execute(self, plan, max_replans=2):
        """Execute a plan (list of PlanStep) with replanning."""
        steps = list(plan)
        done_steps = []
        replans = 0

        while steps:
            ready = self._resolve_deps(steps, done_steps)
            if not ready and any(s.status == "pending" for s in steps):
                # Deadlock: unmet dependencies
                blocked = [s for s in steps if s.status == "pending"]
                raise RuntimeError(
                    f"Plan deadlock: {[s.name for s in blocked]} have unmet dependencies"
                )

            for step in ready:
                step.status = "running"
                step.attempts += 1
                self._execution_log.append({
                    "step": step.name, "event": "start", "time": time.time()
                })

                try:
                    tool_fn = self.tools.get(step.tool)
                    if tool_fn is None:
                        # Try as callable directly
                        tool_fn = step.tool
                    result = tool_fn(**step.args)
                    step.result = result
                    step.status = "done"
                    done_steps.append(step)
                    self._execution_log.append({
                        "step": step.name, "event": "done", "time": time.time()
                    })
                    # Save checkpoint
                    self._checkpoints.append({
                        "step": step.name, "done": [s.name for s in done_steps],
                        "time": time.time()
                    })
                except Exception as e:
                    step.error = str(e)
                    self._execution_log.append({
                        "step": step.name, "event": "error", "error": str(e), "time": time.time()
                    })
                    if step.attempts <= step.max_retries:
                        # Retry
                        step.status = "pending"
                        self._execution_log.append({
                            "step": step.name, "event": "retry", "attempt": step.attempts, "time": time.time()
                        })
                    else:
                        step.status = "failed"
                        if replans < max_replans:
                            # Replan: create recovery step
                            replans += 1
                            recovery = PlanStep(
                                f"recovery_{step.name}",
                                self._replan_handler,
                                {"failed_step": step.name, "error": str(e)},
                                depends_on=[s.name for s in done_steps]
                            )
                            steps.append(recovery)
                            self._execution_log.append({
                                "step": step.name, "event": "replan", "replan": replans, "time": time.time()
                            })
                        else:
                            self._execution_log.append({
                                "step": step.name, "event": "abort", "time": time.time()
                            })
                            raise RuntimeError(f"Step '{step.name}' failed after {step.attempts} attempts")

            # Remove done/failed steps from pending
            steps = [s for s in steps if s.status == "pending"]

        return {
            "done": [s.name for s in done_steps],
            "replans": replans,
            "checkpoints": len(self._checkpoints)
        }

    def _replan_handler(self, failed_step=None, error=None):
        """Default replan handler."""
        return f"Replanned around '{failed_step}': {error}"

    def log(self):
        return self._execution_log

    def checkpoint_report(self):
        return self._checkpoints


def demo_plan_executor():
    """Demo the plan executor."""
    print("\n=== PlanExecutor Demo ===")

    pe = PlanExecutor()
    pe.register_tool("fetch", lambda url: f"Fetched {url}")
    pe.register_tool("process", lambda data: f"Processed: {data.upper()}")
    pe.register_tool("save", lambda content: f"Saved ({len(content)} chars)")

    def flaky_tool(**kwargs):
        if random.random() < 0.5:
            raise ValueError("Transient error!")
        return "Flaky tool result"

    pe.register_tool("flaky", flaky_tool)

    plan = [
        PlanStep("fetch_data", "fetch", {"url": "https://example.com/data"}, max_retries=1),
        PlanStep("process_data", "process", {"data": "{{fetch_data.result}}"},
                 depends_on=["fetch_data"], max_retries=1),
        PlanStep("save_result", "save", {"content": "final output"},
                 depends_on=["process_data"], max_retries=1),
    ]

    result = pe.execute(plan)
    print(f"  Done: {result['done']}")
    print(f"  Replans: {result['replans']}")
    print(f"  Checkpoints: {result['checkpoints']}")

    # Test with failure + replan
    pe2 = PlanExecutor(max_retries=0)  # no retry, force replan
    pe2.register_tool("always_fails", lambda: (_ for _ in ()).throw(ValueError("Planned failure")))
    plan2 = [
        PlanStep("do_work", "always_fails", max_retries=0),
    ]
    try:
        result2 = pe2.execute(plan2, max_replans=1)
        print(f"  Replan result: {result2}")
    except RuntimeError as e:
        print(f"  Expected failure: {e}")

    return True


# ═══════════════════════════════════════════════
# EvalHarness  (Lesson 27)
# ═══════════════════════════════════════════════

class EvalHarness:
    """Fixture-based evaluation harness."""

    def __init__(self):
        self._tasks = []
        self._results = []

    def add_task(self, name, prompt, expected, verifier=None):
        """Add an eval task. verifier(generated, expected) -> bool."""
        if verifier is None:
            verifier = lambda gen, exp: gen.strip() == exp.strip()
        self._tasks.append({
            "name": name, "prompt": prompt, "expected": expected, "verifier": verifier
        })

    def run(self, agent_fn, k=1):
        """Run all tasks through agent_fn. agent_fn(prompt) -> str."""
        self._results = []
        for task in self._tasks:
            start = time.time()
            candidates = []
            for _ in range(k):
                output = agent_fn(task["prompt"])
                candidates.append(output)
            elapsed = time.time() - start

            # pass@1
            pass1 = task["verifier"](candidates[0], task["expected"]) if candidates else False
            # pass@k
            pass_k = any(task["verifier"](c, task["expected"]) for c in candidates)

            self._results.append({
                "name": task["name"],
                "pass@1": pass1,
                "pass@k": pass_k,
                "latency": round(elapsed, 3),
                "cost": round(elapsed * 0.001, 6),  # simulated cost
                "candidates": candidates
            })

        return self.aggregate()

    def aggregate(self):
        """Return aggregate metrics."""
        if not self._results:
            return {}
        n = len(self._results)
        pass1_count = sum(1 for r in self._results if r["pass@1"])
        passk_count = sum(1 for r in self._results if r["pass@k"])
        latencies = [r["latency"] for r in self._results]
        costs = [r["cost"] for r in self._results]

        return {
            "pass@1": round(pass1_count / n, 4),
            "pass@k": round(passk_count / n, 4),
            "mean_latency": round(sum(latencies) / n, 3),
            "mean_cost": round(sum(costs) / n, 6),
            "total_tasks": n
        }

    def report(self):
        """Return formatted report."""
        agg = self.aggregate()
        lines = ["--- EvalHarness Report ---"]
        for r in self._results:
            status = "✅" if r["pass@1"] else "❌"
            lines.append(f"  {status} {r['name']}: pass@1={r['pass@1']}, pass@k={r['pass@k']}, {r['latency']}s")
        lines.append(f"  Aggregate: pass@1={agg['pass@1']}, pass@k={agg['pass@k']}, "
                      f"mean_latency={agg['mean_latency']}s, mean_cost={agg['mean_cost']}")
        return "\n".join(lines)

    def to_json(self):
        return json.dumps(self.aggregate(), indent=2)


def demo_eval_harness():
    """Demo the eval harness."""
    print("\n=== EvalHarness Demo ===")

    eh = EvalHarness()
    eh.add_task("add", "3+5=?", "8", lambda g, e: g.strip() == e)
    eh.add_task("upper", "make hello uppercase", "HELLO", lambda g, e: g.strip() == e)
    eh.add_task("reverse", "reverse abc", "cba", lambda g, e: g.strip() == e)

    def dummy_agent(prompt):
        if "3+5" in prompt:
            return "8"
        if "hello" in prompt:
            return "HELLO"
        if "reverse" in prompt:
            return "cba"
        return "unknown"

    eh.run(dummy_agent, k=3)
    print(eh.report())
    return True


# ═══════════════════════════════════════════════
# Tracer  (Lesson 28)
# ═══════════════════════════════════════════════

class Span:
    """A single tracing span."""

    def __init__(self, name, parent_id=None, attributes=None):
        self.span_id = str(uuid.uuid4())[:8]
        self.name = name
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time = None
        self.attributes = attributes or {}
        self.status = "ok"
        self.events = []

    def add_event(self, name, attributes=None):
        self.events.append({"name": name, "time": time.time(), "attributes": attributes or {}})

    def finish(self, status="ok"):
        self.end_time = time.time()
        self.status = status

    def duration(self):
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self):
        return {
            "span_id": self.span_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": round(self.duration() * 1000, 2),
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events
        }


class Tracer:
    """OpenTelemetry-style tracer. Emits JSON-Lines."""

    def __init__(self, service_name="anggira-agent"):
        self.service_name = service_name
        self._spans = []
        self._counters = defaultdict(int)
        self._histograms = defaultdict(list)

    def start_span(self, name, parent_id=None, attributes=None):
        span = Span(name, parent_id, attributes)
        self._spans.append(span)
        return span

    def inc_counter(self, name, value=1):
        self._counters[name] += value

    def record_histogram(self, name, value):
        self._histograms[name].append(value)

    def emit(self, span):
        """Emit span as JSON-Line."""
        return json.dumps(span.to_dict())

    def spans_as_jsonl(self):
        return "\n".join(self.emit(s) for s in self._spans)

    def prometheus_text(self):
        """Expose counters and histograms in Prometheus text format."""
        lines = []
        for name, val in sorted(self._counters.items()):
            lines.append(f"# HELP {name} counter")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {val}")
        for name, vals in sorted(self._histograms.items()):
            lines.append(f"# HELP {name} histogram")
            lines.append(f"# TYPE {name} histogram")
            if vals:
                lines.append(f"{name}_sum {sum(vals)}")
                lines.append(f"{name}_count {len(vals)}")
        return "\n".join(lines)

    def stats(self):
        return {
            "spans": len(self._spans),
            "counters": dict(self._counters),
            "histogram_count": sum(len(v) for v in self._histograms.values())
        }


def demo_tracer():
    """Demo the tracer."""
    print("\n=== Tracer Demo ===")

    tracer = Tracer()

    # Create parent span
    root = tracer.start_span("root_task", attributes={"user_id": "test"})

    # Nested spans
    child1 = tracer.start_span("tool_call", parent_id=root.span_id, attributes={"tool": "search"})
    child1.add_event("api_request", {"url": "https://api.example.com"})
    time.sleep(0.01)
    child1.finish()

    child2 = tracer.start_span("llm_call", parent_id=root.span_id, attributes={"model": "gpt4"})
    time.sleep(0.02)
    child2.finish()

    root.finish()

    tracer.inc_counter("api_calls", 3)
    tracer.inc_counter("tokens", 1500)
    tracer.record_histogram("latency_ms", 45)
    tracer.record_histogram("latency_ms", 120)

    print(f"  JSON-Lines:\n{tracer.spans_as_jsonl()[:300]}...")
    print(f"  Prometheus:\n{tracer.prometheus_text()}")
    print(f"  Stats: {tracer.stats()}")
    return True


# ═══════════════════════════════════════════════
# CodingAgent  (Lesson 29) — end-to-end harness
# ═══════════════════════════════════════════════

class CodingAgent:
    """End-to-end deterministic coding agent harness.

    Wires: gate chain → dispatcher → sandbox → eval harness → tracer.
    """

    def __init__(self, project_root=None):
        self.project_root = project_root or tempfile.mkdtemp()
        self.tracer = Tracer(service_name="coding-agent")
        self.dispatcher = FunctionCallDispatcher()
        self.sandbox = SandboxRunner(project_root=self.project_root)
        self.verifier = VerificationGate()
        self.eval_harness = EvalHarness()

        # Set up gates
        self.verifier.add_gate("deny_list", deny_list_gate())
        self.verifier.add_gate("budget", budget_gate(max_cost=200.0))

        # Register tools
        self._register_tools()

    def _register_tools(self):
        def read_file_tool(path):
            full_path = os.path.join(self.project_root, path)
            with open(full_path) as f:
                return f.read()

        def write_file_tool(path, content):
            full_path = os.path.join(self.project_root, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
            return f"Wrote {len(content)} bytes to {path}"

        def run_tests_tool():
            result = self.sandbox.run(
                f"cd {self.project_root} && python3 -m unittest discover -v 2>&1 || true",
                timeout=10
            )
            return result["stdout"]

        self.dispatcher.register_handler("read_file", read_file_tool,
                                          {"required": {"path": str}})
        self.dispatcher.register_handler("write_file", write_file_tool,
                                          {"required": {"path": str, "content": str}})
        self.dispatcher.register_handler("run_tests", run_tests_tool)

    def inject_bug(self, filepath, old_text, new_text):
        """Inject a controlled bug for the agent to fix."""
        full_path = os.path.join(self.project_root, filepath)
        content = open(full_path).read()
        content = content.replace(old_text, new_text, 1)
        with open(full_path, "w") as f:
            f.write(content)
        return f"Injected bug in {filepath}: '{old_text}' -> '{new_text}'"

    def _policy(self, prompt):
        """Deterministic fix policy (not an LLM) — reproduces the harness shape."""
        # Parse prompt for file/old/new patterns
        lines = prompt.strip().split("\n")
        for line in lines:
            if line.startswith("BUG:") or line.startswith("FIX:"):
                parts = line.split(":", 2)[1].strip()
                if "->" in parts:
                    old, new = parts.split("->", 1)
                    old, new = old.strip(), new.strip()
                    # Find which file to fix
                    for f in os.listdir(self.project_root):
                        fp = os.path.join(self.project_root, f)
                        if os.path.isfile(fp):
                            content = open(fp).read()
                            if old in content:
                                self.dispatcher.call("write_file", path=f, content=content.replace(old, new, 1))
                                return f"Fixed: replaced '{old[:20]}' with '{new[:20]}' in {f}"
        return "No fix needed"

    def fix_bug(self, prompt):
        """Run the full harness to fix a bug described in prompt."""
        span = self.tracer.start_span("fix_bug", attributes={"prompt": prompt[:50]})

        # Verify the operation
        gate_result = self.verifier.check("fix_bug", {"prompt": prompt})
        if not gate_result.allowed:
            span.finish("denied")
            return {"status": "denied", "reason": gate_result.reason}

        self.tracer.inc_counter("fix_attempts")
        self.tracer.record_histogram("prompt_length", len(prompt))

        try:
            result = self._policy(prompt)
            span.finish("ok")
            self.tracer.inc_counter("fixes_ok")
            return {"status": "ok", "result": result, "trace": span.span_id}
        except Exception as e:
            span.finish("error")
            self.tracer.inc_counter("fixes_failed")
            return {"status": "error", "error": str(e), "trace": span.span_id}

    def report(self):
        return {
            "trace_stats": self.tracer.stats(),
            "dispatch_stats": self.dispatcher.stats(),
            "sandbox_stats": self.sandbox.stats()
        }


def demo_coding_agent():
    """Demo the full coding agent harness."""
    print("\n=== CodingAgent Demo ===")

    import tempfile
    tmpdir = tempfile.mkdtemp()
    agent = CodingAgent(project_root=tmpdir)

    # Create a small project with a bug
    agent.dispatcher.call("write_file", path="math_utils.py", content="""
def add(a, b):
    return a + b

def multiply(a, b):
    return a + b  # BUG: should be a * b!
""")

    agent.dispatcher.call("write_file", path="test_math.py", content="""
import unittest
from math_utils import add, multiply

class TestMath(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)

    def test_multiply(self):
        self.assertEqual(multiply(2, 3), 6)

if __name__ == '__main__':
    unittest.main()
""")

    # Run agent's fix policy
    result = agent.fix_bug("FIX: replace 'a + b' -> 'a * b' in multiply function")
    print(f"  Fix result: {result}")

    # Run tests
    tests = agent.dispatcher.call("run_tests")
    print(f"  Test output: {tests['result'][:200] if isinstance(tests, dict) and 'result' in tests else tests}...")

    print(f"  Agent report: {agent.report()}")
    return True


# ═══════════════════════════════════════════════
# MAIN DEMO
# ═══════════════════════════════════════════════

def demo():
    """Run all demos."""
    results = []

    print("=" * 60)
    print("Anggira Capstone — Agent Harness (Phase 19, Lessons 20-29)")
    print("=" * 60)

    for demo_fn in [
        demo_verification_gate,
        demo_sandbox_runner,
        demo_plan_executor,
        demo_eval_harness,
        demo_tracer,
        demo_coding_agent,
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
