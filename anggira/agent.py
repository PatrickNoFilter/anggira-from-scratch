"""
Anggira Agent — Agent Engineering from Scratch

Phase 13: Tools & Protocols
Phase 14: Agent Engineering

Implements:
- ReAct loop (Think → Act → Observe)
- Tool registry and function calling
- Reflexion (self-reflection from past failures)
- Episodic memory (stores outcomes across trials)
"""

import json
import math
import random
import time


# ═══════════════════════════════════════════════
# TOOL SYSTEM
# ═══════════════════════════════════════════════

class Tool:
    """A callable tool with name, description, and schema."""

    def __init__(self, name, description, fn, parameters=None):
        self.name = name
        self.description = description
        self.fn = fn
        # Simple schema inference
        self.parameters = parameters or {}

    def __repr__(self):
        return f"Tool({self.name}: {self.description})"

    def validate(self, **kwargs):
        if "a" in self.parameters and "a" not in kwargs:
            raise ValueError(f"Missing required parameter 'a'")
        return kwargs

    def __call__(self, **kwargs):
        return self.fn(**kwargs)


def tool(name, description, parameters=None):
    """Decorator to create a Tool."""
    def decorator(fn):
        return Tool(name, description, fn, parameters or {})
    return decorator


# ═══════════════════════════════════════════════
# TOOL REGISTRY
# ═══════════════════════════════════════════════

class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools = {}

    def register(self, tool_obj):
        self._tools[tool_obj.name] = tool_obj

    def get(self, name):
        return self._tools.get(name)

    def list(self):
        return list(self._tools.values())

    def call(self, name, **kwargs):
        t = self.get(name)
        if t is None:
            return {"error": f"Unknown tool: '{name}'"}
        try:
            result = t(**kwargs)
            return {"result": result}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def to_prompt(self):
        lines = ["Available tools:"]
        for t in self._tools.values():
            lines.append(f"  • {t.name}: {t.description}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════
# MEMORY
# ═══════════════════════════════════════════════

class EpisodicMemory:
    """Simple key-value store for agent memory."""

    def __init__(self):
        self._store = {}

    def set(self, key, value):
        self._store[key] = value
        return f"stored {key}"

    def get(self, key):
        return self._store.get(key, None)

    def reflect(self):
        """Return a reflection string based on store contents."""
        if not self._store:
            return "No prior knowledge available."
        return f"Memory contains: {json.dumps(self._store)}"


# ═══════════════════════════════════════════════
# REACT LOOP
# ═══════════════════════════════════════════════

class ReActAgent:
    """
    Reasoning + Acting agent with ReAct loop.

    At each step the agent:
      1. Decides: call a tool or produce final answer
      2. If tool call: executes tool, observes result
      3. If final: returns answer
    """

    def __init__(self, tools, memory=None, max_steps=10):
        self.tools = tools
        self.memory = memory or EpisodicMemory()
        self.max_steps = max_steps
        self.history = []

    def _simulate_llm(self, question):
        """
        Simulates an LLM deciding between tool calls or final answer.
        In production this would call an actual LLM.
        Here we use a scripted decision engine for demo.
        """
        q = question.lower()

        # Check memory first
        if "what do i know" in q or "what is stored" in q:
            return "final", self.memory.reflect()

        # Pattern matching for tool calls
        if "add" in q or "sum" in q or "plus" in q or "+" in q:
            import re
            nums = re.findall(r'\d+\.?\d*', q)
            if len(nums) >= 2:
                a, b = float(nums[0]), float(nums[1])
                return "tool", "add", {"a": a, "b": b}
            return "tool", "calculator", {"expr": q}

        if "multiply" in q or "times" in q:
            nums = re.findall(r'\d+\.?\d*', q)
            if len(nums) >= 2:
                return "tool", "multiply", {"a": float(nums[0]), "b": float(nums[1])}

        if "time" in q or "clock" in q:
            return "tool", "get_time", {}

        if "weather" in q or "temperature" in q:
            cities = ["london", "paris", "tokyo", "new york", "bengaluru",
                      "mumbai", "delhi", "san francisco"]
            for c in cities:
                if c in q:
                    return "tool", "get_weather", {"city": c.title(), "units": "celsius"}
            return "tool", "get_weather", {"city": "Unknown", "units": "celsius"}

        if "store" in q or "remember" in q or "save" in q:
            parts = q.split("as") if " as " in q else q.split(":")
            if len(parts) >= 2:
                key = parts[-1].strip().split()[0] if " as " in q else parts[0].split()[-1]
                val = parts[0].split("store")[-1].strip() if "store" in q else parts[1].strip()
                val = val.replace("as " + key, "").strip()
                if val:
                    return "tool", "kv_set", {"key": key, "value": val}
                return "tool", "kv_set", {"key": "value", "value": q}

        if "get " in q or "retrieve" in q or "what is" in q:
            for word in q.split():
                word = word.strip(".,?!")
                stored = self.memory.get(word)
                if stored:
                    return "tool", "kv_get", {"key": word}
            return "final", self.memory.reflect()

        if "classify" in q:
            for label in ["open", "closed", "pending"]:
                if label in q:
                    return "tool", "classify", {"status": label}
            return "final", "Cannot classify: no valid status found"

        if "reflect" in q or "what happened" in q:
            return "final", self.memory.reflect()

        # Default: final answer from knowledge
        return "final", f"I processed: '{question}'"

    def run(self, question):
        """Run the ReAct loop on a question."""
        self.history = []
        print(f"\n  [user] {question}")

        for step in range(self.max_steps):
            decision, *payload = self._simulate_llm(question)

            if decision == "final":
                answer = payload[0]
                print(f"  [{step:02d}  final] {answer}")
                self.history.append(("final", answer))
                return answer

            elif decision == "tool":
                tool_name = payload[0]
                tool_args = payload[1] if len(payload) > 1 else {}
                print(f"  [{step:02d}  tool] {tool_name}({json.dumps(tool_args)})", end="")

                result = self.tools.call(tool_name, **tool_args)
                print(f" → {result.get('result', result.get('error', '?'))}")
                self.history.append(("tool", tool_name, tool_args, result))

                # Update memory with tool results
                if "result" in result:
                    r = result["result"]
                    if isinstance(r, dict):
                        for k, v in r.items():
                            if isinstance(v, (str, int, float)):
                                self.memory.set(k, str(v))
                    elif isinstance(r, (str, int, float)):
                        self.memory.set(tool_name, str(r))

                question = str(result)  # Feed result back as observation

        return "Max steps reached."


# ═══════════════════════════════════════════════
# REFLEXION AGENT
# ═══════════════════════════════════════════════

class ReflexionAgent:
    """
    Agent with reflexion: after each trial, it reflects on what went wrong
    and uses that reflection to guide the next trial.
    """

    def __init__(self, tools, max_trials=10):
        self.tools = tools
        self.max_trials = max_trials
        self.reflections = []
        self.trial_history = []

    def _evaluate(self, question):
        """Check if the current trial succeeded."""
        q = question.lower()
        # For "pick three ints in [1..9] summing to N"
        if "sum" in q and "int" in q:
            import re
            nums = re.findall(r'\d+', self.trial_history[-1]) if self.trial_history else []
            nums = [int(n) for n in nums if 1 <= int(n) <= 9]
            # Extract target sum
            targets = re.findall(r'summing to (\d+)', q)
            target = int(targets[0]) if targets else 20
            return sum(nums) == target if nums else False

        # For constraint satisfaction
        return False

    def _get_reflection(self):
        """Generate a reflection from past failures."""
        if not self.reflections:
            return "No prior attempts."

        points = []
        for r in self.reflections[-3:]:  # Last 3 reflections
            points.append(f"  - {r}")
        return "Lessons from past attempts:\n" + "\n".join(points)

    def run(self, question):
        """Run with reflexion."""
        print(f"\n  [task] {question}")

        for trial in range(self.max_trials):
            reflection = self._get_reflection()
            print(f"\n  --- Trial {trial + 1} ---")

            # Simulate an attempt (would call LLM in production)
            if trial == 0:
                attempt = "[1, 2, 3] sum=6 delta=-14"
            else:
                # Use reflection to improve
                if any("larger" in r for r in self.reflections):
                    if trial == 1:
                        attempt = "[5, 6, 7] sum=18 delta=-2"
                    else:
                        attempt = "[6, 7, 7] sum=20 delta=0"
                else:
                    attempt = "[1, 2, 3] sum=6 delta=-14"

            print(f"  attempt: {attempt}")
            self.trial_history.append(attempt)

            # Check success
            if self._evaluate(question):
                print(f"  ✓ SUCCESS on trial {trial + 1}")
                return attempt
            else:
                # Generate reflection
                if "delta=-14" in attempt:
                    r = "Values too small; pick larger numbers"
                elif "delta=-2" in attempt:
                    r = "Close but still short; increase slightly"
                else:
                    r = "Need to try a different combination"
                self.reflections.append(r)
                print(f"  reflection: {r}")

        return "Failed to solve."


# ═══════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════

def demo():
    print("🤖 Anggira Agent — Agent Engineering from Scratch")
    print("=" * 60)

    # ── Set up tools ──
    registry = ToolRegistry()

    @tool("add", "Add two numbers a and b.", {"a": "number", "b": "number"})
    def add_(a=0, b=0):
        return float(a) + float(b)

    @tool("multiply", "Multiply two numbers a and b.", {"a": "number", "b": "number"})
    def multiply_(a=0, b=0):
        return float(a) * float(b)

    @tool("get_time", "Get the current time.")
    def get_time_():
        t = time.gmtime()
        return {"now": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", t),
                "timezone": "UTC"}

    @tool("get_weather", "Get current weather for a city.",
          {"city": "string", "units": "string"})
    def get_weather_(city="Unknown", units="celsius"):
        data = {"london": 15, "paris": 18, "tokyo": 22, "new york": 20,
                "bengaluru": 28, "mumbai": 32, "delhi": 35, "san francisco": 16}
        temp = data.get(city.lower(), 25)
        return {"city": city, "temp": temp, "units": units}

    @tool("kv_set", "Store a key-value pair.", {"key": "string", "value": "string"})
    def kv_set_(key="", value=""):
        return f"stored {key}"

    @tool("kv_get", "Retrieve a value by key.", {"key": "string"})
    def kv_get_(key=""):
        return None

    @tool("calculator", "Evaluate a mathematical expression.",
          {"expr": "string"})
    def calculator_(expr=""):
        # Safe eval of simple expressions
        allowed = {'x': 0}
        try:
            result = eval(expr, {"__builtins__": {}}, allowed)
            return float(result)
        except:
            return {"error": f"cannot evaluate: {expr}"}

    @tool("classify", "Classify a status label.",
          {"status": "string"})
    def classify_(status=""):
        valid = ["open", "closed", "pending"]
        if status not in valid:
            raise ValueError(f"status '{status}' not in {valid}")
        return f"classified as {status}"

    # Register tools
    for t in [add_, multiply_, get_time_, get_weather_,
              kv_set_, kv_get_, calculator_, classify_]:
        registry.register(t)

    print(f"\n  Tools loaded: {[t.name for t in registry.list()]}")

    # ── Demo 1: ReAct Agent ──
    print("\n" + "─" * 50)
    print("  DEMO 1: ReAct Agent Loop")
    print("─" * 50)

    agent = ReActAgent(registry)
    queries = [
        "what is 120 plus 15% tax",
        "what time is it",
        "tell me weather in Bengaluru",
        "store my name as Anggira"
    ]
    for q in queries:
        agent.run(q)
    print(f"  Turns used: {sum(len(a.history) for a in [ReActAgent(registry) for _ in queries])}")

    # ── Demo 2: Reflexion Agent ──
    print("\n" + "─" * 50)
    print("  DEMO 2: Reflexion Agent (self-correcting)")
    print("─" * 50)

    reflex = ReflexionAgent(registry)
    result = reflex.run("pick three ints in [1..9] summing to 20")
    print(f"\n  Final result: {result}")


# ═══════════════════════════════════════════════
# STATE MACHINE (LangGraph-style)
# ═══════════════════════════════════════════════

class StateMachine:
    """Simple state machine with checkpoints and human-in-loop gates."""

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.checkpoints = []

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, from_node, to_node, condition=None):
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append((to_node, condition))

    def run(self, start_node, initial_state, max_steps=20):
        state = dict(initial_state)
        current = start_node
        step = 0
        while current and step < max_steps:
            state["step"] = step
            print(f"  [{step}] node: {current}", end="")
            state = self.nodes[current](state)
            self.checkpoints.append({"node": current, "state": dict(state)})
            if state.get("human_gate"):
                print(f"  ⏸ PAUSED (human gate)")
                state["human_approval"] = True
                state["human_gate"] = False
                print(f"     ✓ Approved")
            else:
                print(f"  ✓")
            if current in self.edges:
                for target, cond in self.edges[current]:
                    if cond is None or cond(state):
                        current = target
                        break
                else:
                    current = None
            else:
                current = None
            step += 1
        return state

    def print_history(self):
        print("  Checkpoints:")
        for c in self.checkpoints:
            print(f"    {c['node']}: {c['state'].get('step')}")


# ═══════════════════════════════════════════════
# ORCHESTRATION PATTERNS
# ═══════════════════════════════════════════════

class SupervisorWorker:
    """Supervisor delegates to workers."""

    def __init__(self, workers):
        self.workers = workers

    def run(self, task):
        w = self.workers.get(task["type"])
        if w:
            r = w(task)
            print(f"    → {r}")
            return r
        return None


class Swarm:
    """Agents hand off to each other."""

    def __init__(self, agents):
        self.agents = agents

    def run(self, task, start):
        current = start
        for _ in range(10):
            result = self.agents[current]["handle"](task)
            print(f"    swarm[{current}] → {result}")
            handoff = self.agents[current].get("handoff")
            if handoff and handoff in self.agents:
                print(f"    swarm[{current}] handoff -> {handoff}")
                current = handoff
            else:
                break


class Debate:
    """Agents debate and converge."""

    def __init__(self, agents):
        self.agents = agents

    def run(self, task):
        from collections import Counter
        proposals = {}
        for name, fn in self.agents.items():
            p = fn(task)
            proposals[name] = p
            print(f"    {name} proposes {p}")
        winner = Counter(proposals.values()).most_common(1)[0][0]
        print(f"    converges → {winner}")
        return winner


# ═══════════════════════════════════════════════
# DEMO 3: Orchestration
# ═══════════════════════════════════════════════

def demo_orchestration():
    print("\n" + "─" * 50)
    print("  DEMO 3: State Machine + Agent Orchestration")
    print("─" * 50)

    # Workers
    def bug_h(t):
        return f"bug logged: {t['desc']}"

    def refund_h(t):
        return f"refund handled: {t['desc']}"

    def sales_h(t):
        return f"quote sent: {t['desc']}"

    # State Machine
    def classify(s):
        t = s["input"]
        s["route"] = "bug" if "crash" in t else "refund" if "refund" in t else "general"
        s["ticket"] = f"{s['route'].upper()}-{t[:12]}"
        return s

    def bug_node(s):
        s["output"] = f"logged {s['ticket']}"
        s["human_gate"] = True
        return s

    def refund_node(s):
        s["output"] = f"processed {s['ticket']}"
        return s

    def final_node(s):
        s.setdefault("output", f"handled {s['ticket']}")
        return s

    sm = StateMachine()
    sm.add_node("classify", classify)
    sm.add_node("bug", bug_node)
    sm.add_node("refund", refund_node)
    sm.add_node("final", final_node)
    sm.add_edge("classify", "bug", lambda s: s["route"] == "bug")
    sm.add_edge("classify", "refund", lambda s: s["route"] == "refund")
    sm.add_edge("bug", "final")
    sm.add_edge("refund", "final")

    print("\n  State Machine:")
    r = sm.run("classify", {"input": "the CLI crashes on ctrl-c, please fix"})
    sm.print_history()
    print(f"  Final: {r}")

    # Supervisor-Worker
    print("\n  Supervisor-Worker:")
    sw = SupervisorWorker({"bug": bug_h, "refund": refund_h, "sales": sales_h})
    sw.run({"type": "bug", "desc": "CLI crash on ctrl-c"})
    sw.run({"type": "refund", "desc": "order damaged"})

    # Swarm
    print("\n  Swarm (handoff):")
    swarm = Swarm({
        "refund": {"handle": refund_h, "handoff": "bug"},
        "bug": {"handle": bug_h, "handoff": None},
    })
    swarm.run({"desc": "widget arrived broken"}, "refund")

    # Debate
    print("\n  Debate:")
    debate = Debate({"alpha": lambda t: t["type"],
                     "beta": lambda t: t["type"],
                     "gamma": lambda t: t["type"]})
    debate.run({"type": "refund", "desc": "defective"})


if __name__ == "__main__":
    demo()
    demo_orchestration()
