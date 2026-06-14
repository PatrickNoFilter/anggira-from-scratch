"""
Anggira Multi-Agent & Swarms — Agent Engineering from Scratch

Phase 16: Multi-Agent Orchestration & Swarm Intelligence

All pure Python stdlib — no external dependencies.

Implements:
  1.  Orchestration Primitives — Sequential, Broadcast, RoundRobin, Supervisor
  2.  Debate / Society of Mind — N-agent debate loop with voting
  3.  Hierarchical Architecture — Parent agent decomposition + workers + synthesis
  4.  Role Specialization Pipeline — Planner → Executor → Critic → Verifier
  5.  Parallel Swarm — Fan-out/fan-in with thread pool
  6.  Group Chat (AutoGen-style) — Speaker selection + message history
  7.  Handoffs & Routines — Agent returns handoff target, executor routes
  8.  Shared Memory / Blackboard — Topic-keyed with read/write/subscribe
  9.  Consensus & Voting — Majority, weighted, PBFT-style (prepare/commit)
  10. Negotiation & Bargaining — Two agents negotiate over resources
  11. Swarm Optimization — Particle Swarm Optimization (PSO)
  12. Generative Agents Simulation — Reflection + planning + spatial movement
  13. Agent Economy — Shapley value attribution + reputation tracking
  14. Failure Mode Detection — Groupthink, coordination, verification failures
  15. Production Patterns — Durable queue with checkpoint/resume for worker crashes
  16. demo() — Run all components
"""
import copy
import json
import math
import random
import statistics
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: Agent wrapper
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Agent:
    """A named callable agent.  Any callable can be wrapped."""
    name: str
    fn: Callable[..., Any]
    description: str = ""

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self):
        return f"Agent({self.name})"


# ==============================================================================
#  1. ORCHESTRATION PRIMITIVES
# ==============================================================================

class Sequential:
    """Chain agents sequentially — each agent receives the previous output."""

    def __init__(self, agents: list[Agent]):
        self.agents = agents

    def run(self, initial_input: Any = None) -> list[dict]:
        results = []
        inp = initial_input
        for agent in self.agents:
            out = agent(inp)
            results.append({"agent": agent.name, "input": inp, "output": out})
            inp = out
        return results


class Broadcast:
    """Send the same input to all agents in parallel and collect outputs."""

    def __init__(self, agents: list[Agent]):
        self.agents = agents

    def run(self, inp: Any = None) -> list[dict]:
        results = []
        with ThreadPoolExecutor(max_workers=len(self.agents)) as pool:
            futures = {pool.submit(a, inp): a for a in self.agents}
            for f in as_completed(futures):
                a = futures[f]
                try:
                    out = f.result()
                    results.append({"agent": a.name, "output": out})
                except Exception as e:
                    results.append({"agent": a.name, "error": str(e)})
        return results


class RoundRobin:
    """Rotate through agents, passing output of one as input to next."""

    def __init__(self, agents: list[Agent]):
        self.agents = agents

    def run(self, initial_input: Any = None, rounds: int = 1) -> list[dict]:
        results = []
        inp = initial_input
        for rnd in range(rounds):
            for agent in self.agents:
                out = agent(inp)
                results.append({"round": rnd, "agent": agent.name, "output": out})
                inp = out
        return results


class Supervisor:
    """Supervisor agent delegates work to specialized workers."""

    def __init__(self, supervisor: Agent, workers: dict[str, Agent]):
        self.supervisor = supervisor
        self.workers = workers

    def run(self, task: Any) -> dict:
        # Supervisor decides which worker(s) to invoke
        decision = self.supervisor(task)
        if isinstance(decision, str):
            decision = {"worker": decision}
        worker_name = decision.get("worker", list(self.workers.keys())[0])
        worker = self.workers.get(worker_name)
        if worker is None:
            return {"error": f"Unknown worker: {worker_name}", "decision": decision}
        result = worker(task)
        return {"worker": worker_name, "decision": decision, "result": result}


# ==============================================================================
#  2. DEBATE / SOCIETY OF MIND
# ==============================================================================

class DebateSocietyOfMind:
    """N-agent debate loop where agents share arguments and vote after N rounds."""

    def __init__(self, agents: list[Agent], rounds: int = 3):
        self.agents = agents
        self.rounds = rounds
        self.history: list[dict] = []

    def run(self, topic: str) -> dict:
        arguments = {a.name: a(topic) for a in self.agents}
        self.history.append({"round": 0, "arguments": dict(arguments)})

        for rnd in range(1, self.rounds + 1):
            # Each agent sees all other arguments from previous round
            shared = "\n".join(
                f"{name}: {arg}" for name, arg in arguments.items()
            )
            new_args = {}
            for agent in self.agents:
                prompt = f"Topic: {topic}\nOther arguments:\n{shared}\n\nYour rebuttal:"
                new_args[agent.name] = agent(prompt)
            arguments = new_args
            self.history.append({"round": rnd, "arguments": dict(arguments)})

        # Final vote — each agent votes
        votes = {}
        for agent in self.agents:
            vote_prompt = (
                f"Topic: {topic}\n"
                f"Final arguments:\n" +
                "\n".join(f"{n}: {a}" for n, a in arguments.items()) +
                "\n\nVote for the best argument (agent name):"
            )
            votes[agent.name] = agent(vote_prompt)

        from collections import Counter
        tally = Counter(votes.values())
        winner = tally.most_common(1)[0][0] if tally else None

        return {
            "topic": topic,
            "rounds": self.rounds,
            "final_arguments": arguments,
            "votes": votes,
            "tally": dict(tally),
            "winner": winner,
            "history": list(self.history),
        }


# ==============================================================================
#  3. HIERARCHICAL ARCHITECTURE
# ==============================================================================

class HierarchicalAgent:
    """Parent agent decomposes tasks, delegates to child workers, and synthesizes results."""

    def __init__(self, parent: Agent, children: list[Agent],
                 decomposer: Optional[Callable] = None,
                 synthesizer: Optional[Callable] = None):
        self.parent = parent
        self.children = {c.name: c for c in children}
        self.decomposer = decomposer or self._default_decomposer
        self.synthesizer = synthesizer or self._default_synthesizer

    @staticmethod
    def _default_decomposer(task: str) -> list[str]:
        """Split a task into subtask descriptions."""
        lines = [l.strip() for l in task.split("\n") if l.strip()]
        if len(lines) <= 1:
            # Simple splitting heuristic
            words = task.split()
            chunk_size = max(1, len(words) // 3)
            return [" ".join(words[i:i + chunk_size])
                    for i in range(0, len(words), chunk_size)]
        return lines

    @staticmethod
    def _default_synthesizer(results: list[dict]) -> str:
        parts = []
        for r in results:
            parts.append(f"{r['worker']}: {r['output']}")
        return "\n".join(parts)

    def run(self, task: Any) -> dict:
        # 1. Parent decomposes
        subtasks = self.decomposer(task)
        decomposition = self.parent(f"decompose: {task}")

        # 2. Dispatch to children
        child_results = []
        with ThreadPoolExecutor(max_workers=len(self.children)) as pool:
            child_map = {}
            for i, subtask in enumerate(subtasks):
                child_name = list(self.children.keys())[i % len(self.children)]
                child = self.children[child_name]
                child_map[pool.submit(child, subtask)] = child_name
            for f in as_completed(child_map):
                cname = child_map[f]
                try:
                    child_results.append({
                        "worker": cname,
                        "subtask": subtasks[list(self.children.keys()).index(cname)
                                            if cname in self.children else 0],
                        "output": f.result(),
                    })
                except Exception as e:
                    child_results.append({"worker": cname, "error": str(e)})

        # 3. Parent synthesizes
        synthesis = self.synthesizer(child_results)
        final = self.parent(f"synthesize: {synthesis}")

        return {
            "decomposition": decomposition,
            "subtasks": subtasks,
            "child_results": child_results,
            "synthesis": synthesis,
            "final": final,
        }


# ==============================================================================
#  4. ROLE SPECIALIZATION PIPELINE
# ==============================================================================

class RolePipeline:
    """
    Planner → Executor → Critic → Verifier chain.
    Each role is a distinct agent that sees the outputs of prior stages.
    """

    def __init__(self, planner: Agent, executor: Agent, critic: Agent, verifier: Agent):
        self.planner = planner
        self.executor = executor
        self.critic = critic
        self.verifier = verifier

    def run(self, task: Any) -> dict:
        # 1. Plan
        plan = self.planner(f"Task: {task}\nCreate a plan:")
        # 2. Execute
        execution = self.executor(f"Plan: {plan}\nExecute the plan:")
        # 3. Critique
        critique = self.critic(f"Plan: {plan}\nExecution: {execution}\nCritique:")
        # 4. Verify (yes/no)
        verification = self.verifier(
            f"Plan: {plan}\nExecution: {execution}\nCritique: {critique}\nVerify:"
        )
        accepted = "accept" in str(verification).lower() or "yes" in str(verification).lower()

        return {
            "task": task,
            "plan": plan,
            "execution": execution,
            "critique": critique,
            "verification": verification,
            "accepted": accepted,
        }


# ==============================================================================
#  5. PARALLEL SWARM — Fan-out/Fan-in
# ==============================================================================

class ParallelSwarm:
    """Fan-out a task to N workers in parallel, then fan-in (collect) results."""

    def __init__(self, workers: list[Agent], max_workers: Optional[int] = None):
        self.workers = workers
        self.max_workers = max_workers or len(workers)

    def fan_out(self, tasks: list[Any]) -> list[dict]:
        """Dispatch each task to a worker (cycling)."""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for i, task in enumerate(tasks):
                worker = self.workers[i % len(self.workers)]
                futures[pool.submit(worker, task)] = {
                    "worker": worker.name,
                    "task_index": i,
                }
            for f in as_completed(futures):
                meta = futures[f]
                try:
                    meta["result"] = f.result()
                except Exception as e:
                    meta["error"] = str(e)
                results.append(meta)
        return results

    def fan_in(self, results: list[dict], aggregator: Optional[Callable] = None) -> Any:
        """Combine results using an aggregator or default concatenation."""
        if aggregator:
            return aggregator(results)
        outputs = [r.get("result", r.get("error", "")) for r in results]
        return outputs

    def run(self, tasks: list[Any], aggregator: Optional[Callable] = None) -> dict:
        results = self.fan_out(tasks)
        aggregated = self.fan_in(results, aggregator)
        return {"results": results, "aggregated": aggregated}


# ==============================================================================
#  6. GROUP CHAT (AutoGen-style)
# ==============================================================================

class GroupChat:
    """
    Multi-agent group chat with speaker selection (round-robin, broadcast, random)
    and message history.
    """

    def __init__(self, agents: list[Agent], max_turns: int = 10):
        self.agents = agents
        self.max_turns = max_turns
        self.messages: list[dict] = []

    def _select_speaker(self, mode: str = "round_robin", turn: int = 0) -> Agent:
        if mode == "round_robin":
            return self.agents[turn % len(self.agents)]
        elif mode == "random":
            return random.choice(self.agents)
        elif mode == "broadcast":
            # broadcast means all speak at same turn — handled separately
            return self.agents[turn % len(self.agents)]
        return self.agents[0]

    def run(self, topic: str, mode: str = "round_robin") -> list[dict]:
        self.messages = [{"role": "system", "content": topic}]
        context = topic

        for turn in range(self.max_turns):
            if mode == "broadcast":
                # All agents speak each turn
                turn_messages = []
                for agent in self.agents:
                    msg = agent(context)
                    entry = {"role": agent.name, "content": msg, "turn": turn}
                    self.messages.append(entry)
                    turn_messages.append(f"{agent.name}: {msg}")
                context = "\n".join(turn_messages)
            else:
                speaker = self._select_speaker(mode, turn)
                msg = speaker(context)
                entry = {"role": speaker.name, "content": msg, "turn": turn}
                self.messages.append(entry)
                context = f"{context}\n{speaker.name}: {msg}"

        return self.messages


# ==============================================================================
#  7. HANDOFFS & ROUTINES
# ==============================================================================

class Handoff:
    """
    An agent can return a Handoff(name, payload) to transfer control
    to another agent.  The HandoffExecutor routes accordingly.
    """

    def __init__(self, target: str, payload: Any = None):
        self.target = target
        self.payload = payload

    def __repr__(self):
        return f"Handoff(to={self.target})"


class HandoffExecutor:
    """
    Executes agent calls and follows Handoff responses to route
    control to the next agent in a chain.
    """

    def __init__(self, agents: dict[str, Agent], max_hops: int = 10):
        self.agents = agents
        self.max_hops = max_hops
        self.trace: list[dict] = []

    def run(self, start: str, inp: Any = None) -> dict:
        current = start
        payload = inp
        for hop in range(self.max_hops):
            agent = self.agents.get(current)
            if agent is None:
                return {"error": f"Unknown agent: {current}", "trace": self.trace}
            result = agent(payload)
            self.trace.append({
                "hop": hop,
                "agent": current,
                "input": payload,
                "result": result,
            })
            if isinstance(result, Handoff):
                payload = result.payload
                current = result.target
            else:
                return {"final_agent": current, "final_result": result,
                        "trace": self.trace, "hops": hop + 1}
        return {"error": "Max hops reached", "trace": self.trace}


# ==============================================================================
#  8. SHARED MEMORY / BLACKBOARD
# ==============================================================================

class Blackboard:
    """
    Topic-keyed blackboard with read/write/subscribe.
    Agents can share state via a common blackboard.
    """

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
        self._history: list[dict] = []

    def write(self, topic: str, value: Any, author: str = "anonymous") -> None:
        with self._lock:
            self._store[topic] = value
            entry = {"topic": topic, "value": value, "author": author,
                     "timestamp": time.time()}
            self._history.append(entry)
            # Notify subscribers
            for cb in self._subscribers.get(topic, []):
                try:
                    cb(entry)
                except Exception:
                    pass

    def read(self, topic: str, default: Any = None) -> Any:
        with self._lock:
            return self._store.get(topic, default)

    def subscribe(self, topic: str, callback: Callable) -> None:
        with self._lock:
            self._subscribers[topic].append(callback)

    def topics(self) -> list[str]:
        with self._lock:
            return list(self._store.keys())

    def history(self, topic: Optional[str] = None) -> list[dict]:
        with self._lock:
            if topic is None:
                return list(self._history)
            return [e for e in self._history if e["topic"] == topic]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._subscribers.clear()
            self._history.clear()


# ==============================================================================
#  9. CONSENSUS & VOTING
# ==============================================================================

class ConsensusMechanism:
    """
    Voting mechanisms: simple majority, weighted vote, PBFT-style.
    """

    @staticmethod
    def majority_vote(votes: list[Any]) -> dict:
        """Simple majority: option with count > len(votes)/2 wins."""
        from collections import Counter
        tally = Counter(votes)
        total = len(votes)
        most_common = tally.most_common()
        if not most_common:
            return {"winner": None, "tally": {}, "consensus": False}
        winner, count = most_common[0]
        return {
            "winner": winner,
            "tally": dict(tally),
            "consensus": count > total / 2,
            "total": total,
        }

    @staticmethod
    def weighted_vote(votes: list[tuple[Any, float]]) -> dict:
        """
        Weighted vote: each vote is (option, weight).
        Option with highest cumulative weight wins.
        """
        tally: dict[Any, float] = defaultdict(float)
        for option, weight in votes:
            tally[option] += weight
        total_weight = sum(tally.values())
        if not tally:
            return {"winner": None, "tally": {}, "consensus": False}
        winner = max(tally, key=tally.get)
        return {
            "winner": winner,
            "tally": dict(tally),
            "consensus": tally[winner] > total_weight / 2 if total_weight else False,
            "total_weight": total_weight,
        }

    @staticmethod
    def pbft_consensus(validators: list[Callable], proposal: Any,
                       f: int = 1) -> dict:
        """
        PBFT-style: prepare phase (validators check proposal),
        commit phase (validators commit after seeing 2f+1 prepares).
        """
        n = len(validators)
        # Phase 1: Prepare
        prepares = []
        for v in validators:
            decision = v(proposal)
            prepares.append(decision)
        prepare_yes = sum(1 for d in prepares if d is True or str(d).lower() in ("yes", "true"))

        # Phase 2: Commit (need 2f+1 prepares to proceed)
        threshold = 2 * f + 1
        if prepare_yes < threshold:
            return {
                "proposal": proposal,
                "prepares": prepares,
                "prepare_yes": prepare_yes,
                "threshold": threshold,
                "commits": [],
                "consensus": False,
                "reason": f"Only {prepare_yes}/{threshold} prepares",
            }

        commits = []
        for v in validators:
            # In PBFT, commit happens after seeing enough prepares
            commit = v(f"commit:{proposal}")
            commits.append(commit)
        commit_yes = sum(1 for d in commits if d is True or str(d).lower() in ("yes", "true"))

        return {
            "proposal": proposal,
            "prepares": prepares,
            "prepare_yes": prepare_yes,
            "threshold": threshold,
            "commits": commits,
            "commit_yes": commit_yes,
            "consensus": commit_yes >= threshold,
        }


# ==============================================================================
#  10. NEGOTIATION & BARGAINING
# ==============================================================================

class Negotiation:
    """
    Two agents negotiate over resources with proposals and counter-proposals.
    """

    def __init__(self, agent_a: Agent, agent_b: Agent,
                 max_rounds: int = 10, deal_threshold: float = 0.1):
        self.agent_a = agent_a
        self.agent_b = agent_b
        self.max_rounds = max_rounds
        self.deal_threshold = deal_threshold
        self.log: list[dict] = []

    def run(self, resources: dict[str, float],
            initial_split: Optional[dict[str, float]] = None) -> dict:
        """
        resources: dict of resource name -> total available
        initial_split: optional dict of agent_a -> {resource: share}
        """
        if initial_split is None:
            # Start with 50/50 split
            initial_split = {
                self.agent_a.name: {r: v / 2 for r, v in resources.items()},
                self.agent_b.name: {r: v / 2 for r, v in resources.items()},
            }

        current_proposal = initial_split[self.agent_a.name]
        last_offer = None

        for rnd in range(self.max_rounds):
            # Agent A proposes
            a_proposal = self.agent_a({
                "round": rnd,
                "resources": resources,
                "current_proposal": current_proposal,
                "role": "proposer",
            })
            if isinstance(a_proposal, dict):
                current_proposal = a_proposal
            self.log.append({
                "round": rnd,
                "proposer": self.agent_a.name,
                "proposal": current_proposal,
            })

            # Agent B evaluates
            b_response = self.agent_b({
                "round": rnd,
                "resources": resources,
                "proposal": current_proposal,
                "role": "evaluator",
            })
            b_accepts = (
                isinstance(b_response, bool) and b_response
                or isinstance(b_response, str) and "accept" in b_response.lower()
            )

            self.log.append({
                "round": rnd,
                "evaluator": self.agent_b.name,
                "response": b_response,
                "accepted": b_accepts,
            })

            if b_accepts:
                return {
                    "agreement": True,
                    "rounds": rnd + 1,
                    "split": {
                        self.agent_a.name: current_proposal,
                        self.agent_b.name: {
                            r: resources[r] - current_proposal.get(r, 0)
                            for r in resources
                        },
                    },
                    "log": self.log,
                }

            # Counter-proposal: roles swap
            current_proposal, last_offer = last_offer, current_proposal
            # Swap roles for next round (agent B proposes)
            self.agent_a, self.agent_b = self.agent_b, self.agent_a

        return {
            "agreement": False,
            "rounds": self.max_rounds,
            "last_proposal": current_proposal,
            "log": self.log,
        }


# ==============================================================================
#  11. SWARM OPTIMIZATION — Particle Swarm Optimization (PSO)
# ==============================================================================

class Particle:
    """A single particle in PSO."""

    def __init__(self, position: list[float], velocity: Optional[list[float]] = None):
        self.position = list(position)
        self.velocity = velocity or [0.0 for _ in position]
        self.best_position = list(position)
        self.best_score = float("inf")

    def __repr__(self):
        return f"Particle(pos={self.position}, best={self.best_score:.4f})"


class SwarmOptimizer:
    """
    Particle Swarm Optimization for function optimization.
    Supports inertia weight, cognitive and social coefficients.
    """

    def __init__(self, n_particles: int, dimensions: int,
                 bounds: list[tuple[float, float]],
                 objective: Callable[[list[float]], float],
                 inertia: float = 0.7,
                 cognitive: float = 1.5,
                 social: float = 1.5):
        self.n_particles = n_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.objective = objective
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        # Initialize particles
        self.particles: list[Particle] = []
        for _ in range(n_particles):
            pos = [random.uniform(b[0], b[1]) for b in bounds]
            vel = [random.uniform(-abs(b[1] - b[0]) * 0.1,
                                  abs(b[1] - b[0]) * 0.1) for b in bounds]
            self.particles.append(Particle(pos, vel))

        self.global_best_position = list(self.particles[0].position)
        self.global_best_score = float("inf")
        self.history: list[float] = []

    def _clamp(self, val: float, idx: int) -> float:
        lo, hi = self.bounds[idx]
        return max(lo, min(hi, val))

    def step(self) -> dict:
        """Run one optimization step. Returns step stats."""
        for p in self.particles:
            score = self.objective(p.position)
            if score < p.best_score:
                p.best_score = score
                p.best_position = list(p.position)
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = list(p.position)

        best_improved = False
        prev_best = self.global_best_score

        for p in self.particles:
            for d in range(self.dimensions):
                r1, r2 = random.random(), random.random()
                cognitive_term = self.cognitive * r1 * (p.best_position[d] - p.position[d])
                social_term = self.social * r2 * (self.global_best_position[d] - p.position[d])
                p.velocity[d] = (self.inertia * p.velocity[d]
                                 + cognitive_term + social_term)
                p.position[d] = self._clamp(p.position[d] + p.velocity[d], d)

        self.history.append(self.global_best_score)
        best_improved = self.global_best_score < prev_best

        return {
            "best_score": self.global_best_score,
            "best_position": list(self.global_best_position),
            "best_improved": best_improved,
        }

    def optimize(self, iterations: int = 100, verbose: bool = False) -> dict:
        """Run optimization for a given number of iterations."""
        for i in range(iterations):
            info = self.step()
            if verbose and (i % 20 == 0 or i == iterations - 1):
                print(f"  iter {i:4d}: best={info['best_score']:.6f}")
        return {
            "best_score": self.global_best_score,
            "best_position": list(self.global_best_position),
            "iterations": iterations,
            "history": self.history,
        }


# ==============================================================================
#  12. GENERATIVE AGENTS SIMULATION
# ==============================================================================

@dataclass
class GenerativeAgent:
    """An agent with memory, reflection, planning, and spatial movement."""
    name: str
    role: str = ""
    position: tuple[int, int] = (0, 0)
    memory: list[str] = field(default_factory=list)
    plans: list[str] = field(default_factory=list)
    reflection: str = ""
    energy: float = 100.0
    inventory: list[str] = field(default_factory=list)

    def observe(self, observation: str) -> None:
        self.memory.append(observation)
        if len(self.memory) > 50:
            self.memory = self.memory[-50:]

    def plan(self, planner: Callable[[str], list[str]]) -> None:
        context = f"Agent {self.name} ({self.role}) at {self.position}. "
        context += f"Energy: {self.energy:.1f}. "
        context += f"Recent memories: {self.memory[-5:]}"
        self.plans = planner(context)

    def reflect(self, reflector: Callable[[str], str]) -> None:
        context = f"Agent {self.name} memories: {self.memory}"
        self.reflection = reflector(context)

    def move(self, direction: str, grid_size: tuple[int, int] = (10, 10)) -> None:
        dx, dy = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0),
                  "ne": (1, -1), "nw": (-1, -1), "se": (1, 1), "sw": (-1, 1),
                  "": (0, 0)}.get(direction, (0, 0))
        x = max(0, min(grid_size[0] - 1, self.position[0] + dx))
        y = max(0, min(grid_size[1] - 1, self.position[1] + dy))
        self.position = (x, y)
        self.energy -= 1.0
        self.observe(f"Moved {direction} to {self.position}")

    def interact(self, other: "GenerativeAgent", interaction: str) -> str:
        msg = f"{self.name} -> {other.name}: \"{interaction}\""
        self.observe(msg)
        other.observe(f"{self.name} says: \"{interaction}\"")
        return msg


class GenerativeSimulation:
    """
    Run a simulation of N generative agents on a grid.
    Each round: agents observe, reflect, plan, act, and move.
    """

    def __init__(self, agents: list[GenerativeAgent],
                 grid_size: tuple[int, int] = (10, 10)):
        self.agents = agents
        self.grid_size = grid_size
        self.round = 0
        self.log: list[dict] = []

    def step(self, planner: Callable[[str], list[str]],
             reflector: Callable[[str], str],
             action_chooser: Callable[[str], str]) -> None:
        """Run one simulation step for all agents."""
        self.round += 1
        round_log = {"round": self.round, "actions": []}

        for agent in self.agents:
            agent.reflect(reflector)
            agent.plan(planner)
            action = action_chooser(
                f"{agent.name} plans: {agent.plans}, reflection: {agent.reflection}"
            )
            round_log["actions"].append({
                "agent": agent.name,
                "action": action,
                "position": agent.position,
                "energy": agent.energy,
            })
            # Execute action if it's a move
            if action.startswith("move_"):
                direction = action.split("_", 1)[1]
                agent.move(direction, self.grid_size)
            elif action.startswith("say_"):
                msg = action.split("_", 1)[1]
                others = [a for a in self.agents if a.name != agent.name]
                if others:
                    target = random.choice(others)
                    agent.interact(target, msg)

        self.log.append(round_log)

    def run(self, steps: int, planner: Callable[[str], list[str]],
            reflector: Callable[[str], str],
            action_chooser: Callable[[str], str]) -> list[dict]:
        for _ in range(steps):
            self.step(planner, reflector, action_chooser)
        return self.log

    def status(self) -> dict:
        return {
            "round": self.round,
            "agents": {
                a.name: {
                    "position": a.position,
                    "energy": a.energy,
                    "memories": len(a.memory),
                    "reflection": a.reflection[:60] if a.reflection else "",
                }
                for a in self.agents
            }
        }


# ==============================================================================
#  13. AGENT ECONOMY — Shapley value + Reputation
# ==============================================================================

class AgentEconomy:
    """
    Shapley value attribution for cooperative tasks + reputation tracking.
    """

    def __init__(self, agents: list[str]):
        self.agents = agents
        self.reputation: dict[str, float] = {a: 1.0 for a in agents}
        self.transaction_log: list[dict] = []

    def shapley_values(self, coalition_results: dict[frozenset, float]) -> dict[str, float]:
        """
        Compute Shapley values for each agent given a characteristic function
        over coalitions. coalition_results maps frozenset -> value.
        """
        n = len(self.agents)
        values = {a: 0.0 for a in self.agents}

        for agent in self.agents:
            for coalition, value in coalition_results.items():
                if agent not in coalition:
                    continue
                coalition_without = frozenset(a for a in coalition if a != agent)
                val_with = value
                val_without = coalition_results.get(coalition_without, 0.0)
                marginal = val_with - val_without
                size = len(coalition)
                weight = math.factorial(size - 1) * math.factorial(n - size) / math.factorial(n)
                values[agent] += marginal * weight

        return values

    def update_reputation(self, agent: str, delta: float) -> None:
        self.reputation[agent] = max(0.0, min(2.0,
                                     self.reputation.get(agent, 1.0) + delta))

    def record_transaction(self, agent: str, action: str, value: float) -> None:
        self.transaction_log.append({
            "agent": agent,
            "action": action,
            "value": value,
            "timestamp": time.time(),
        })

    def report(self) -> dict:
        return {
            "reputation": dict(self.reputation),
            "transactions": len(self.transaction_log),
            "avg_reputation": statistics.mean(self.reputation.values()),
        }


# ==============================================================================
#  14. FAILURE MODE DETECTION
# ==============================================================================

class FailureModeDetector:
    """
    Detect failure modes in multi-agent systems:
    - Groupthink: conformity pressure, cascade effects
    - Coordination failures: conflicting actions, unmet dependencies
    - Verification failures: agents accepting flawed outputs
    """

    @staticmethod
    def detect_groupthink(vote_history: list[dict],
                          conformity_threshold: float = 0.8) -> dict:
        """
        Detect groupthink by measuring conformity pressure.
        If >conformity_threshold of agents follow the majority without deviation,
        groupthink is likely.
        """
        if not vote_history:
            return {"groupthink": False, "reason": "No vote data"}

        rounds = len(vote_history)
        conformities = []

        for rnd_data in vote_history:
            votes = list(rnd_data.values())
            if not votes:
                continue
            from collections import Counter
            tally = Counter(votes)
            majority_count = tally.most_common(1)[0][1]
            conformity = majority_count / len(votes)
            conformities.append(conformity)

        avg_conformity = statistics.mean(conformities) if conformities else 0
        cascade = all(c >= conformity_threshold for c in conformities[-3:]) if len(conformities) >= 3 else False

        return {
            "groupthink": avg_conformity >= conformity_threshold,
            "avg_conformity": avg_conformity,
            "conformity_per_round": conformities,
            "cascade_detected": cascade,
            "reason": (
                f"Avg conformity {avg_conformity:.2%} "
                f"{'exceeds' if avg_conformity >= conformity_threshold else 'below'} "
                f"threshold {conformity_threshold:.0%}"
            ),
        }

    @staticmethod
    def detect_coordination_failure(actions: list[dict]) -> dict:
        """
        Detect coordination failures: agents clashing (same resource, opposite intent),
        circular dependencies, or unmet preconditions.
        """
        failures = []
        # Check for resource conflicts
        resource_claims = defaultdict(list)
        for act in actions:
            resource = act.get("resource")
            if resource:
                resource_claims[resource].append(act["agent"])

        for resource, claimants in resource_claims.items():
            if len(claimants) > 1:
                failures.append({
                    "type": "resource_conflict",
                    "resource": resource,
                    "agents": claimants,
                })

        # Check for circular delegation
        delegation_chain = [a.get("delegates_to") for a in actions
                            if a.get("delegates_to")]
        if len(delegation_chain) >= 2 and delegation_chain[-1] == delegation_chain[0]:
            failures.append({
                "type": "circular_delegation",
                "chain": delegation_chain,
            })

        return {
            "coordination_failure": len(failures) > 0,
            "failures": failures,
            "failure_count": len(failures),
        }

    @staticmethod
    def detect_verification_failure(verification_log: list[dict]) -> dict:
        """
        Detect when flawed outputs pass verification repeatedly.
        """
        false_positives = 0
        false_negatives = 0
        for entry in verification_log:
            actual_quality = entry.get("actual_quality", 0.5)
            passed = entry.get("passed", False)
            if actual_quality < 0.3 and passed:
                false_positives += 1
            if actual_quality >= 0.7 and not passed:
                false_negatives += 1

        total = len(verification_log)
        fp_rate = false_positives / total if total else 0
        fn_rate = false_negatives / total if total else 0

        return {
            "verification_failure": fp_rate > 0.2 or fn_rate > 0.2,
            "false_positive_rate": fp_rate,
            "false_negative_rate": fn_rate,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total": total,
        }


# ==============================================================================
#  15. PRODUCTION PATTERNS — Durable Queue with Checkpoint/Resume
# ==============================================================================

@dataclass
class TaskItem:
    """A task in the durable queue."""
    id: str
    payload: Any
    status: str = "pending"  # pending | running | completed | failed
    result: Any = None
    error: Optional[str] = None
    checkpoint: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "payload": self.payload,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "checkpoint": self.checkpoint,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskItem":
        return cls(**d)


class DurableQueue:
    """
    Queue with checkpoint/resume support for durable execution.
    Tasks are tracked; on worker crash, the queue can resume from
    the last checkpoint.
    """

    def __init__(self):
        self._queue: deque[TaskItem] = deque()
        self._completed: list[TaskItem] = []
        self._failed: list[TaskItem] = []
        self._running: Optional[TaskItem] = None
        self._lock = threading.Lock()
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"task-{int(time.time() * 1000)}-{self._id_counter}"

    def enqueue(self, payload: Any) -> str:
        task = TaskItem(id=self._next_id(), payload=payload)
        with self._lock:
            self._queue.append(task)
        return task.id

    def enqueue_batch(self, payloads: list[Any]) -> list[str]:
        return [self.enqueue(p) for p in payloads]

    def dequeue(self) -> Optional[TaskItem]:
        with self._lock:
            if not self._queue:
                return None
            task = self._queue.popleft()
            task.status = "running"
            self._running = task
            return task

    def complete(self, task_id: str, result: Any, checkpoint: Optional[dict] = None) -> None:
        with self._lock:
            if self._running and self._running.id == task_id:
                self._running.status = "completed"
                self._running.result = result
                self._running.checkpoint = checkpoint
                self._completed.append(self._running)
                self._running = None

    def fail(self, task_id: str, error: str) -> None:
        with self._lock:
            if self._running and self._running.id == task_id:
                self._running.status = "failed"
                self._running.error = error
                self._failed.append(self._running)
                self._running = None

    def resume_from_checkpoint(self, task_id: str,
                               state_restorer: Callable[[dict], Any]) -> Optional[Any]:
        """Resume a failed task from its checkpoint using the state_restorer."""
        with self._lock:
            task = None
            for t in self._failed:
                if t.id == task_id:
                    task = t
                    break
            if task is None or task.checkpoint is None:
                return None
            restored_state = state_restorer(task.checkpoint)
            return restored_state

    def retry_failed(self, worker: Callable[[Any], Any],
                     max_retries: int = 3) -> list[dict]:
        """Retry all failed tasks."""
        results = []
        with self._lock:
            failed = list(self._failed)
            self._failed.clear()

        for task in failed:
            for attempt in range(max_retries):
                try:
                    result = worker(task.payload)
                    task.status = "completed"
                    task.result = result
                    self._completed.append(task)
                    results.append({"task_id": task.id, "status": "completed",
                                    "attempts": attempt + 1})
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        task.status = "failed"
                        task.error = str(e)
                        self._failed.append(task)
                        results.append({"task_id": task.id, "status": "failed",
                                        "error": str(e), "attempts": attempt + 1})
                    # else retry
        return results

    def stats(self) -> dict:
        with self._lock:
            return {
                "pending": len(self._queue),
                "completed": len(self._completed),
                "failed": len(self._failed),
                "running": self._running is not None,
            }

    def snapshot(self) -> dict:
        """Full state snapshot for persistence/recovery."""
        with self._lock:
            return {
                "pending": [t.to_dict() for t in self._queue],
                "completed": [t.to_dict() for t in self._completed],
                "failed": [t.to_dict() for t in self._failed],
                "running": self._running.to_dict() if self._running else None,
                "id_counter": self._id_counter,
            }

    @classmethod
    def from_snapshot(cls, data: dict) -> "DurableQueue":
        q = cls()
        q._queue = deque(TaskItem.from_dict(t) for t in data.get("pending", []))
        q._completed = [TaskItem.from_dict(t) for t in data.get("completed", [])]
        q._failed = [TaskItem.from_dict(t) for t in data.get("failed", [])]
        q._id_counter = data.get("id_counter", 0)
        return q


class DurableExecutor:
    """
    Executor that processes tasks from a DurableQueue with checkpointing.
    On worker crash, tasks can be resumed from their last checkpoint.
    """

    def __init__(self, queue: Optional[DurableQueue] = None):
        self.queue = queue or DurableQueue()
        self.worker_registry: dict[str, Callable] = {}

    def register_worker(self, task_type: str, worker: Callable) -> None:
        self.worker_registry[task_type] = worker

    def process_next(self, task_type: str = "default") -> Optional[dict]:
        task = self.queue.dequeue()
        if task is None:
            return None

        worker = self.worker_registry.get(task_type)
        if worker is None:
            self.queue.fail(task.id, f"No worker for '{task_type}'")
            return {"task_id": task.id, "status": "failed", "error": "No worker"}

        try:
            # Save checkpoint before execution
            checkpoint = {"state": "pre_execution", "payload_hash": hash(str(task.payload))}
            result = worker(task.payload)
            self.queue.complete(task.id, result, checkpoint)
            return {"task_id": task.id, "status": "completed", "result": result}
        except Exception as e:
            self.queue.fail(task.id, f"{type(e).__name__}: {e}")
            return {"task_id": task.id, "status": "failed", "error": str(e)}

    def process_all(self, task_type: str = "default") -> list[dict]:
        results = []
        while True:
            r = self.process_next(task_type)
            if r is None:
                break
            results.append(r)
        return results

    def recover_and_resume(self, task_id: str,
                           state_restorer: Callable[[dict], Any]) -> Optional[Any]:
        return self.queue.resume_from_checkpoint(task_id, state_restorer)

    def stats(self) -> dict:
        return self.queue.stats()


# ==============================================================================
#  DEMO
# ==============================================================================

def demo():
    """Run demonstrations of all 15 swarm/multi-agent components."""
    print("=" * 66)
    print("  🧠  Anggira Multi-Agent & Swarms  —  Demo")
    print("=" * 66)

    # ── Helper agents used across demos ─────────────────────────────────────
    def make_echo(name: str, prefix: str = "") -> Agent:
        return Agent(name, lambda x: f"{prefix}{x}")

    def identity_agent(name: str) -> Agent:
        return Agent(name, lambda x: x)

    agents = {
        "alice": Agent("alice", lambda x: f"alice processed: {x}"),
        "bob": Agent("bob", lambda x: f"bob processed: {x}"),
        "charlie": Agent("charlie", lambda x: f"charlie processed: {x}"),
    }

    # ═══════════════════════════════════════════════════════════════════════
    #  1. Orchestration Primitives
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [1] Orchestration Primitives")
    print("─" * 66)

    seq = Sequential([agents["alice"], agents["bob"], agents["charlie"]])
    seq_results = seq.run("hello")
    for r in seq_results:
        print(f"      {r['agent']}: {r['output']}")

    bc = Broadcast([agents["alice"], agents["bob"]])
    bc_results = bc.run("broadcast test")
    for r in bc_results:
        print(f"      broadcast -> {r['agent']}: {r['output']}")

    rr = RoundRobin([agents["alice"], agents["bob"]])
    rr_results = rr.run("round", rounds=2)
    for r in rr_results:
        print(f"      round {r['round']} {r['agent']}: {r['output']}")

    sup = Supervisor(
        Agent("sup", lambda x: {"worker": "alice" if "task" in str(x) else "bob"}),
        {"alice": agents["alice"], "bob": agents["bob"]},
    )
    sup_result = sup.run("task: do something")
    print(f"      supervisor -> {sup_result['worker']}: {sup_result['result']}")

    # ═══════════════════════════════════════════════════════════════════════
    #  2. Debate / Society of Mind
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [2] Debate / Society of Mind")
    print("─" * 66)

    debate_agents = [
        Agent("pro", lambda x: "I support this position because it increases efficiency."),
        Agent("con", lambda x: "I oppose this position due to risks and costs."),
        Agent("neutral", lambda x: "Let's find a compromise that balances both views."),
    ]
    debate = DebateSocietyOfMind(debate_agents, rounds=2)
    debate_result = debate.run("Should we implement AI safety constraints?")
    print(f"      Winner: {debate_result['winner']}")
    print(f"      Tally:  {debate_result['tally']}")

    # ═══════════════════════════════════════════════════════════════════════
    #  3. Hierarchical Architecture
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [3] Hierarchical Architecture")
    print("─" * 66)

    parent = Agent("parent", lambda x: f"parent decomposed: {x}")
    child_a = Agent("child_A", lambda x: f"A solved: {x}")
    child_b = Agent("child_B", lambda x: f"B solved: {x}")
    hier = HierarchicalAgent(parent, [child_a, child_b])
    hier_result = hier.run("Design a distributed task queue")
    print(f"      Subtasks: {hier_result['subtasks']}")
    for cr in hier_result['child_results']:
        print(f"      {cr['worker']}: {cr.get('output', cr.get('error', ''))}")
    print(f"      Final: {hier_result['final']}")

    # ═══════════════════════════════════════════════════════════════════════
    #  4. Role Specialization Pipeline
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [4] Role Specialization Pipeline")
    print("─" * 66)

    pipeline = RolePipeline(
        Agent("planner", lambda x: "Step 1: Gather requirements\nStep 2: Implement\nStep 3: Test"),
        Agent("executor", lambda x: "Implementation complete: feature X added"),
        Agent("critic", lambda x: "Missing test coverage for edge cases; consider refactoring"),
        Agent("verifier", lambda x: "Accept after addressing test coverage"),
    )
    pipe_result = pipeline.run("Add a new API endpoint")
    print(f"      Plan:       {pipe_result['plan']}")
    print(f"      Execution:  {pipe_result['execution']}")
    print(f"      Critique:   {pipe_result['critique']}")
    print(f"      Verdict:    {'✓ Accepted' if pipe_result['accepted'] else '✗ Rejected'}")

    # ═══════════════════════════════════════════════════════════════════════
    #  5. Parallel Swarm (Fan-out/Fan-in)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [5] Parallel Swarm — Fan-out/Fan-in")
    print("─" * 66)

    workers = [
        Agent("worker_1", lambda x: f"result_1({x})"),
        Agent("worker_2", lambda x: f"result_2({x})"),
        Agent("worker_3", lambda x: f"result_3({x})"),
    ]
    swarm_parallel = ParallelSwarm(workers, max_workers=3)
    swarm_result = swarm_parallel.run(
        tasks=["task_A", "task_B", "task_C", "task_D", "task_E"],
    )
    print(f"      Aggregated ({len(swarm_result['results'])} results):")
    for r in swarm_result['results']:
        print(f"        {r['worker']}: {r.get('result', r.get('error', '?'))}")

    # ═══════════════════════════════════════════════════════════════════════
    #  6. Group Chat (AutoGen-style)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [6] Group Chat (AutoGen-style)")
    print("─" * 66)

    chat_agents = [
        Agent("moderator", lambda x: "Let's stay on topic."),
        Agent("expert1", lambda x: "I recommend using Python for this."),
        Agent("expert2", lambda x: "Consider performance implications."),
    ]
    chat = GroupChat(chat_agents, max_turns=4)
    chat_messages = chat.run("Best language for backend?", mode="round_robin")
    print(f"      {len(chat_messages)} messages exchanged")
    for m in chat_messages[:4]:
        if m["role"] != "system":
            print(f"      [{m['turn']}] {m['role']}: {m['content'][:50]}...")

    # ═══════════════════════════════════════════════════════════════════════
    #  7. Handoffs & Routines
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [7] Handoffs & Routines")
    print("─" * 66)

    def triage_fn(x):
        if "refund" in str(x).lower():
            return Handoff("refund_agent", x)
        return f"general: {x}"

    def refund_fn(x):
        return f"refund processed: {x}"

    handoff_exec = HandoffExecutor({
        "triage": Agent("triage", triage_fn),
        "refund_agent": Agent("refund_agent", refund_fn),
    })
    handoff_result = handoff_exec.run("triage", "Customer requests refund for order #123")
    print(f"      Final agent: {handoff_result['final_agent']}")
    print(f"      Final result: {handoff_result['final_result']}")
    print(f"      Hops: {handoff_result['hops']}")

    # ═══════════════════════════════════════════════════════════════════════
    #  8. Shared Memory / Blackboard
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [8] Shared Memory / Blackboard")
    print("─" * 66)

    bb = Blackboard()
    bb.write("inventory", {"widgets": 42, "gadgets": 17}, author="alice")
    bb.write("pricing", {"widget_price": 9.99}, author="bob")
    inventory = bb.read("inventory")
    print(f"      inventory: {inventory}")
    print(f"      topics: {bb.topics()}")
    print(f"      history entries: {len(bb.history())}")

    # Test subscription
    received = []
    bb.subscribe("alerts", lambda e: received.append(e))
    bb.write("alerts", "low stock: widgets", author="system")
    print(f"      subscriber notified: {len(received)} alert(s)")

    # ═══════════════════════════════════════════════════════════════════════
    #  9. Consensus & Voting
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [9] Consensus & Voting")
    print("─" * 66)

    cm = ConsensusMechanism()
    # Majority
    mv = cm.majority_vote(["A", "A", "B", "A", "C"])
    print(f"      Majority: winner={mv['winner']}, consensus={mv['consensus']}")

    # Weighted
    wv = cm.weighted_vote([("A", 0.8), ("B", 0.5), ("A", 1.2)])
    print(f"      Weighted: winner={wv['winner']}, consensus={wv['consensus']}")

    # PBFT-style
    validators = [
        lambda p: True,
        lambda p: True,
        lambda p: "yes",
        lambda p: False,   # faulty
    ]
    pbft = cm.pbft_consensus(validators, "new_block", f=1)
    print(f"      PBFT: consensus={pbft['consensus']}, "
          f"prepares={pbft['prepare_yes']}/{pbft['threshold']}")

    # ═══════════════════════════════════════════════════════════════════════
    #  10. Negotiation & Bargaining
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [10] Negotiation & Bargaining")
    print("─" * 66)

    def negotiator_a(x):
        # Always proposes 70/30 in its favor, accepts anything >= 40%
        if x.get("role") == "proposer":
            return {"budget": x["resources"]["budget"] * 0.7}
        proposal = x.get("proposal", {})
        if proposal.get("budget", 0) >= x["resources"]["budget"] * 0.4:
            return "accept"
        return "counter"

    def negotiator_b(x):
        if x.get("role") == "proposer":
            return {"budget": x["resources"]["budget"] * 0.6}
        proposal = x.get("proposal", {})
        if proposal.get("budget", 0) >= x["resources"]["budget"] * 0.3:
            return "accept"
        return "counter"

    neg = Negotiation(
        Agent("party_A", negotiator_a),
        Agent("party_B", negotiator_b),
        max_rounds=5,
    )
    neg_result = neg.run({"budget": 1000})
    print(f"      Agreement: {neg_result['agreement']} in {neg_result['rounds']} rounds")
    if neg_result.get("split"):
        for party, split in neg_result["split"].items():
            print(f"        {party}: {split}")

    # ═══════════════════════════════════════════════════════════════════════
    #  11. Swarm Optimization (PSO)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [11] Swarm Optimization (PSO)")
    print("─" * 66)

    def rosenbrock(x: list[float]) -> float:
        """Rosenbrock banana function (minimum at (1,1) = 0)."""
        return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                   for i in range(len(x) - 1))

    pso = SwarmOptimizer(
        n_particles=20,
        dimensions=2,
        bounds=[(-5, 5), (-5, 5)],
        objective=rosenbrock,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
    )
    pso_result = pso.optimize(iterations=50, verbose=False)
    print(f"      Best position: {[f'{v:.4f}' for v in pso_result['best_position']]}")
    print(f"      Best score:    {pso_result['best_score']:.6f}")
    print(f"      (Expected: ~0 at [1, 1])")

    # ═══════════════════════════════════════════════════════════════════════
    #  12. Generative Agents Simulation
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [12] Generative Agents Simulation")
    print("─" * 66)

    gen_agents = [
        GenerativeAgent("Alice", role="explorer"),
        GenerativeAgent("Bob", role="gatherer"),
        GenerativeAgent("Eve", role="builder"),
    ]

    def simple_planner(ctx: str) -> list[str]:
        return ["explore north area", "gather resources", "build shelter"]

    def simple_reflector(ctx: str) -> str:
        return f"Reflecting on: {ctx[:40]}..."

    def simple_action_chooser(ctx: str) -> str:
        actions = ["move_n", "move_e", "move_s", "move_w", "say_hello", "say_look"]
        return random.choice(actions)

    sim = GenerativeSimulation(gen_agents, grid_size=(5, 5))
    sim.run(steps=5, planner=simple_planner, reflector=simple_reflector,
            action_chooser=simple_action_chooser)
    status = sim.status()
    print(f"      Rounds: {status['round']}")
    for name, info in status["agents"].items():
        print(f"        {name}: pos={info['position']}, "
              f"energy={info['energy']:.1f}, memories={info['memories']}")

    # ═══════════════════════════════════════════════════════════════════════
    #  13. Agent Economy
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [13] Agent Economy — Shapley Value & Reputation")
    print("─" * 66)

    economy = AgentEconomy(["alice", "bob", "charlie"])
    coalition_results = {
        frozenset({"alice"}): 10,
        frozenset({"bob"}): 15,
        frozenset({"charlie"}): 5,
        frozenset({"alice", "bob"}): 35,
        frozenset({"alice", "charlie"}): 20,
        frozenset({"bob", "charlie"}): 25,
        frozenset({"alice", "bob", "charlie"}): 50,
    }
    shapley = economy.shapley_values(coalition_results)
    print(f"      Shapley values:")
    for agent, value in sorted(shapley.items()):
        print(f"        {agent}: {value:.2f}")

    economy.update_reputation("alice", 0.2)
    economy.update_reputation("bob", -0.1)
    economy.record_transaction("alice", "completed_task", 10.0)
    report = economy.report()
    print(f"      Reputation: {report['reputation']}")
    print(f"      Avg reputation: {report['avg_reputation']:.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    #  14. Failure Mode Detection
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [14] Failure Mode Detection")
    print("─" * 66)

    fm = FailureModeDetector()

    # Groupthink detection
    vote_history = [
        {"alice": "A", "bob": "A", "charlie": "A", "dave": "A"},
        {"alice": "A", "bob": "A", "charlie": "A", "dave": "B"},
        {"alice": "A", "bob": "A", "charlie": "A", "dave": "A"},
    ]
    gt = fm.detect_groupthink(vote_history, conformity_threshold=0.75)
    print(f"      Groupthink: {gt['groupthink']} (conformity={gt['avg_conformity']:.1%})")
    print(f"        Cascade: {gt['cascade_detected']}")

    # Coordination failure
    actions = [
        {"agent": "alice", "resource": "cpu"},
        {"agent": "bob", "resource": "cpu"},
        {"agent": "charlie", "resource": "memory"},
    ]
    cf = fm.detect_coordination_failure(actions)
    print(f"      Coordination failure: {cf['coordination_failure']}")
    for f in cf["failures"]:
        print(f"        {f['type']}: {f}")

    # Verification failure
    vlog = [
        {"actual_quality": 0.1, "passed": True},   # false positive
        {"actual_quality": 0.9, "passed": False},   # false negative
        {"actual_quality": 0.8, "passed": True},
    ]
    vf = fm.detect_verification_failure(vlog)
    print(f"      Verification failure: {vf['verification_failure']}")
    print(f"        FP rate: {vf['false_positive_rate']:.0%}, "
          f"FN rate: {vf['false_negative_rate']:.0%}")

    # ═══════════════════════════════════════════════════════════════════════
    #  15. Production Patterns — Durable Queue
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 66)
    print("  [15] Production Patterns — Durable Queue")
    print("─" * 66)

    durable_q = DurableQueue()
    exec_system = DurableExecutor(durable_q)
    exec_system.register_worker("default", lambda x: f"processed {x}")

    task_ids = exec_system.queue.enqueue_batch(["order_1", "order_2", "order_3"])
    print(f"      Enqueued: {task_ids}")

    results = exec_system.process_all()
    for r in results:
        print(f"        {r['task_id']}: {r['status']}")

    stats = exec_system.stats()
    print(f"      Queue stats: {stats}")

    # Snapshot + restore
    snap = exec_system.queue.snapshot()
    restored = DurableQueue.from_snapshot(snap)
    print(f"      Restored queue: pending={len(restored._queue)}, "
          f"completed={len(restored._completed)}")

    # ═══════════════════════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 66)
    print("  ✅  All 15 components demonstrated successfully!")
    print("=" * 66)
    print("""
  Components:
   1.  Orchestration Primitives  — Sequential, Broadcast, RoundRobin, Supervisor
   2.  Debate / Society of Mind  — N-agent debate loop
   3.  Hierarchical Architecture — Decompose → children → synthesize
   4.  Role Specialization       — Planner → Executor → Critic → Verifier
   5.  Parallel Swarm            — Fan-out/fan-in with ThreadPoolExecutor
   6.  Group Chat                — AutoGen-style speaker selection
   7.  Handoffs & Routines       — Handoff object + executor routing
   8.  Shared Memory/Blackboard  — Topic-keyed read/write/subscribe
   9.  Consensus & Voting        — Majority, weighted, PBFT-style
  10.  Negotiation & Bargaining  — Proposals & counter-proposals
  11.  Swarm Optimization        — Particle Swarm Optimization (PSO)
  12.  Generative Agents         — Reflection, planning, spatial movement
  13.  Agent Economy             — Shapley value + reputation
  14.  Failure Mode Detection    — Groupthink, coordination, verification
  15.  Production Patterns       — Durable queue with checkpoint/resume
  """)


if __name__ == "__main__":
    demo()
