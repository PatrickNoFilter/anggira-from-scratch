"""
Anggira Capstone — AI Research Agent

Phase 19, Lessons 50-57: AI Research Scientist

Implements:
- HypothesisGenerator — Generate diverse, testable hypotheses
- LiteratureRetriever — Retrieve and summarize related work
- ExperimentRunner — Sandboxed experiment execution
- ResultEvaluator — Verdict from metrics
- PaperWriter — LaTeX paper skeleton
- CriticLoop — Iterative improvement with convergence
- ResearchAgent — End-to-end research loop
"""

import json
import math
import os
import random
import re
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ═══════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════

@dataclass
class Hypothesis:
    """A scientific hypothesis."""
    id: str
    question: str
    claim: str
    prediction: str
    confidence: float = 0.5
    domain: str = "general"
    tags: list[str] = field(default_factory=list)
    timestamp: float = 0.0

    def short(self):
        return f"[{self.id[:6]}] {self.claim[:60]}..."


@dataclass
class LiteratureRef:
    """A reference from literature."""
    title: str
    authors: list[str]
    year: int
    abstract: str
    source: str = ""
    relevance: float = 0.5
    key_finding: str = ""


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    hypothesis_id: str
    metric_name: str
    metric_value: float
    baseline_value: Optional[float] = None
    significance: float = 0.0
    raw_output: str = ""
    duration_ms: float = 0.0


@dataclass
class ResearchPaper:
    """Generated research paper."""
    title: str
    authors: list[str]
    abstract: str
    sections: dict[str, str]
    citations: list[str]
    metrics_summary: dict
    confidence_score: float


# ═══════════════════════════════════════════════
# HypothesisGenerator  (Lesson 50-51)
# ═══════════════════════════════════════════════

class HypothesisGenerator:
    """Generate diverse, testable hypotheses.

    Uses a template-based approach for simulation.
    In production, this would call an LLM.
    """

    TEMPLATES = [
        "Increasing {hyperparam} in {model_class} improves {metric} on {dataset}",
        "{technique} combined with {approach} reduces {problem} in {domain}",
        "The {method} approach outperforms {baseline} when {condition}",
        "Adding {component} to {architecture} improves {attribute} while maintaining {constraint}",
        "{property} of {system} correlates with {outcome} under {setting}",
    ]

    DOMAINS = {
        "nlp": {
            "tokens": ["word embedding", "attention head", "tokenizer"],
            "hyperparams": ["learning rate", "batch size", "hidden dim", "dropout"],
            "metrics": ["perplexity", "BLEU", "ROUGE"],
            "techniques": ["pretraining", "fine-tuning", "prompt engineering"],
            "approaches": ["few-shot", "zero-shot", "instruction tuning"],
            "problems": ["catastrophic forgetting", "hallucination", "overfitting"],
            "models": ["transformer", "GPT", "BERT", "LLaMA"],
            "datasets": ["GLUE", "SuperGLUE", "MMLU", "GSM8K"],
            "architectures": ["encoder-only", "decoder-only", "encoder-decoder"],
            "components": ["positional encoding", "layer norm", "attention mask", "KV cache"],
            "attributes": ["expressiveness", "efficiency", "generalization"],
            "constraints": ["inference cost", "memory budget", "latency SLA"],
            "properties": ["Scale", "Depth", "Width", "Context length"],
            "systems": ["language model", "embedding model", "generative model"],
            "outcomes": ["few-shot performance", "in-context learning", "factual accuracy"],
            "conditions": ["domain shift", "low-resource", "adversarial input"],
            "baselines": ["standard fine-tuning", "zero-shot inference", "random baseline"],
        },
        "vision": {
            "tokens": ["pixel patch", "attention map", "conv kernel"],
            "hyperparams": ["patch size", "embed dim", "num heads"],
            "metrics": ["accuracy", "mAP", "IoU"],
            "techniques": ["data augmentation", "knowledge distillation", "architecture search"],
            "approaches": ["end-to-end", "two-stage", "self-supervised"],
            "problems": ["domain gap", "occlusion", "scale variance"],
            "models": ["ViT", "ResNet", "ConvNeXt", "DINO"],
            "datasets": ["ImageNet", "COCO", "ADE20K"],
            "architectures": ["CNN", "Transformer", "hybrid"],
            "components": ["convolution", "self-attention", "pooling", "skip connection"],
            "attributes": ["spatial reasoning", "texture bias", "shape bias"],
            "constraints": ["GFLOPs", "parameter count", "throughput"],
            "properties": ["Input resolution", "Receptive field", "Model width"],
            "systems": ["image classifier", "detector", "segmenter"],
            "outcomes": ["classification accuracy", "detection precision", "segmentation quality"],
            "conditions": ["distribution shift", "adversarial patch", "ambient occlusion"],
            "baselines": ["supervised baseline", "contrastive baseline"],
        },
    }

    def __init__(self, domain="nlp"):
        self.domain = domain
        self._generated_ids = set()
        self._diversity_bank = defaultdict(set)  # tag -> set of claim fragments

    def generate(self, n=5, diversity_threshold=0.3):
        """Generate n hypotheses with diversity forcing."""
        vocab = self.DOMAINS.get(self.domain, self.DOMAINS["nlp"])
        hypotheses = []

        attempts = 0
        while len(hypotheses) < n and attempts < n * 10:
            attempts += 1
            h = self._single_hypothesis(vocab)
            if self._is_diverse(h, diversity_threshold):
                hypotheses.append(h)
                self._register(h)

        return hypotheses

    def _single_hypothesis(self, vocab):
        """Generate a single hypothesis from templates."""
        template = random.choice(self.TEMPLATES)
        claim = template.format(
            hyperparam=random.choice(vocab.get("hyperparams", ["lr"])),
            model_class=random.choice(vocab.get("models", ["model"])),
            metric=random.choice(vocab.get("metrics", ["accuracy"])),
            dataset=random.choice(vocab.get("datasets", ["data"])),
            technique=random.choice(vocab.get("techniques", ["method"])),
            approach=random.choice(vocab.get("approaches", ["fine-tuning"])),
            problem=random.choice(vocab.get("problems", ["overfitting"])),
            domain=self.domain,
            method=random.choice(vocab.get("techniques", ["method"])),
            baseline=random.choice(vocab.get("baselines", ["baseline"])),
            condition=random.choice(vocab.get("conditions", ["standard"])),
            component=random.choice(vocab.get("components", ["attention"])),
            architecture=random.choice(vocab.get("architectures", ["transformer"])),
            attribute=random.choice(vocab.get("attributes", ["performance"])),
            constraint=random.choice(vocab.get("constraints", ["cost"])),
            property=random.choice(vocab.get("properties", ["Scale"])),
            system=random.choice(vocab.get("systems", ["model"])),
            outcome=random.choice(vocab.get("outcomes", ["performance"])),
            setting=random.choice(vocab.get("conditions", ["standard setting"])),
        )

        question = f"What is the effect of {claim.lower()}?"
        prediction = f"{claim} is expected to show improvement in {random.choice(vocab.get('metrics', ['performance']))}"

        return Hypothesis(
            id=str(uuid.uuid4()),
            question=question,
            claim=claim,
            prediction=prediction,
            confidence=random.uniform(0.3, 0.9),
            domain=self.domain,
            tags=[self.domain, random.choice(list(vocab.keys()))],
            timestamp=time.time(),
        )

    def _is_diverse(self, h, threshold):
        """Check if hypothesis is diverse enough from prior ones."""
        words = set(h.claim.lower().split())
        for existing_words in self._diversity_bank.values():
            overlap = len(words & existing_words) / max(len(words | existing_words), 1)
            if overlap > threshold:
                return False
        return True

    def _register(self, h):
        """Register hypothesis in diversity bank."""
        for tag in h.tags:
            self._diversity_bank[tag].update(h.claim.lower().split())


# ═══════════════════════════════════════════════
# LiteratureRetriever  (Lesson 52)
# ═══════════════════════════════════════════════

class LiteratureRetriever:
    """Retrieve and summarize related work.

    Simulated — in production this queries arXiv / Semantic Scholar.
    """

    def __init__(self):
        self._library = self._build_sample_library()

    def _build_sample_library(self):
        """Build a sample literature library."""
        return [
            LiteratureRef(
                title="Scaling Laws for Neural Language Models",
                authors=["Kaplan", "McCandlish", "Henighan", "Brown", "Chess", "Child", "Gray", "Radford", "Wu", "Amodei"],
                year=2023,
                abstract="We find that language model performance scales as a power-law with model size, dataset size, and compute budget.",
                key_finding="Performance scales as power-law with model size, data, and compute.",
                relevance=0.95,
            ),
            LiteratureRef(
                title="Training Compute-Optimal Large Language Models",
                authors=["Hoffmann", "Borgeaud", "Mensch", "et al."],
                year=2022,
                abstract="We find that for a given compute budget, optimal performance is achieved by training a smaller model on more data than current practice.",
                key_finding="Optimal model size scales slower than previously thought — more data, not bigger models.",
                relevance=0.92,
            ),
            LiteratureRef(
                title="Attention Is All You Need",
                authors=["Vaswani", "Shazeer", "Parmar", "et al."],
                year=2017,
                abstract="We propose a novel network architecture, the Transformer, based solely on attention mechanisms.",
                key_finding="Transformer with multi-head self-attention achieves SOTA on translation.",
                relevance=0.88,
            ),
            LiteratureRef(
                title="Large Language Models are Few-Shot Learners",
                authors=["Brown", "Mann", "Ryder", "et al."],
                year=2020,
                abstract="We show that scaling up language models greatly improves task-agnostic few-shot performance.",
                key_finding="GPT-3 achieves strong few-shot results without fine-tuning.",
                relevance=0.90,
            ),
            LiteratureRef(
                title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
                authors=["Wei", "Wang", "Schuurmans", "et al."],
                year=2022,
                abstract="We show that chain-of-thought reasoning prompts improve performance on arithmetic, commonsense, and symbolic reasoning tasks.",
                key_finding="CoT prompting dramatically improves multi-step reasoning.",
                relevance=0.85,
            ),
            LiteratureRef(
                title="Constitutional AI: Harmlessness from AI Feedback",
                authors=["Bai", "Kadavath", "et al."],
                year=2022,
                abstract="We train a harmless AI assistant via self-improvement from constitutional principles.",
                key_finding="RLAIF can substitute RLHF for harmlessness training.",
                relevance=0.82,
            ),
            LiteratureRef(
                title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
                authors=["Rafailov", "Sharma", "Mitchell", "et al."],
                year=2023,
                abstract="We show that we can directly optimize language models from human preferences without explicit reward modeling.",
                key_finding="DPO matches RLHF without training a reward model.",
                relevance=0.80,
            ),
            LiteratureRef(
                title="LLaMA: Open and Efficient Foundation Language Models",
                authors=["Touvron", "Lavril", "Izacard", "et al."],
                year=2023,
                abstract="We introduce LLaMA, a collection of foundation language models trained on trillions of tokens.",
                key_finding="LLaMA-13B outperforms GPT-3 (175B) on most benchmarks.",
                relevance=0.87,
            ),
            LiteratureRef(
                title="QLoRA: Efficient Finetuning of Quantized Language Models",
                authors=["Dettmers", "Pagnoni", "Holtzman", "et al."],
                year=2023,
                abstract="We present QLoRA, an efficient method to finetune quantized LLMs.",
                key_finding="QLoRA enables 65B model finetuning on a single 48GB GPU.",
                relevance=0.78,
            ),
            LiteratureRef(
                title="The Llama 3 Herd of Models",
                authors=["AI Meta", "Team"],
                year=2024,
                abstract="We introduce Llama 3, a new family of foundation models trained on over 15T tokens.",
                key_finding="Llama 3 405B is the first open-source model to rival frontier models.",
                relevance=0.85,
            ),
        ]

    def search(self, query, top_k=3):
        """Search library for relevant references (BM25-like scoring)."""
        query_terms = set(query.lower().split())
        scored = []
        for ref in self._library:
            text = (ref.title + " " + ref.abstract).lower()
            match_count = sum(1 for t in query_terms if t in text)
            score = match_count / max(len(query_terms), 1) * ref.relevance
            scored.append((score, ref))

        scored.sort(key=lambda x: -x[0])
        return [ref for _, ref in scored[:top_k]]

    def retrieve_for_hypothesis(self, hypothesis: Hypothesis, top_k=3):
        """Retrieve literature relevant to a hypothesis."""
        query = f"{hypothesis.claim} {hypothesis.domain}"
        refs = self.search(query, top_k)
        return refs

    def summarize(self, refs):
        """Summarize a list of literature references."""
        if not refs:
            return "No related work found."

        lines = ["Related Work Summary:"]
        for i, ref in enumerate(refs, 1):
            lines.append(f"  [{i}] {ref.title} ({ref.year})")
            lines.append(f"       Key: {ref.key_finding}")
            lines.append(f"       Relevance: {ref.relevance:.2f}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════
# ExperimentRunner  (Lesson 53)
# ═══════════════════════════════════════════════

class ExperimentRunner:
    """Sandboxed experiment execution.

    Runs experiments (simulated) and returns results.
    In production, this would call a compute sandbox.
    """

    def __init__(self, registry=None):
        self.registry = registry or {}
        self._results = []
        self._debug_log = []

    def register_benchmark(self, name, fn):
        """Register a benchmark function.

        fn(hypothesis) -> ExperimentResult
        """
        self.registry[name] = fn

    def run(self, hypothesis: Hypothesis, benchmark="default", params=None):
        """Run a single experiment."""
        start = time.time()

        if benchmark in self.registry:
            result = self.registry[benchmark](hypothesis)
        else:
            # Default simulated experiment
            metric_value = random.uniform(0.5, 0.95)
            baseline = random.uniform(0.5, 0.85)
            significance = min(1.0, max(0.0, (metric_value - baseline) / baseline))

            result = ExperimentResult(
                hypothesis_id=hypothesis.id,
                metric_name="accuracy",
                metric_value=round(metric_value, 4),
                baseline_value=round(baseline, 4),
                significance=round(significance, 4),
                raw_output=f"Experiment completed for '{hypothesis.claim[:50]}...'",
                duration_ms=(time.time() - start) * 1000,
            )

        self._results.append(result)
        return result

    def run_batch(self, hypotheses, benchmark="default"):
        """Run experiments for a batch of hypotheses."""
        return [self.run(h, benchmark) for h in hypotheses]

    def stats(self):
        return {
            "total_experiments": len(self._results),
            "benchmarks": list(self.registry.keys()),
        }


# ═══════════════════════════════════════════════
# ResultEvaluator  (Lesson 54)
# ═══════════════════════════════════════════════

class ResultEvaluator:
    """Evaluate experiment results and render verdicts."""

    @staticmethod
    def evaluate(result: ExperimentResult) -> dict:
        """Evaluate a single result and return a verdict."""
        verdict = "inconclusive"

        if result.baseline_value is not None:
            improvement = result.metric_value - result.baseline_value
            rel_improvement = improvement / max(result.baseline_value, 1e-12)

            if rel_improvement > 0.1 and result.significance > 0.2:
                verdict = "confirmed"
            elif rel_improvement < -0.05:
                verdict = "refuted"
            elif result.significance < 0.05:
                verdict = "inconclusive"
            else:
                verdict = "weakly_supported"

        return {
            "verdict": verdict,
            "improvement": round(result.metric_value - result.baseline_value, 4),
            "rel_improvement": round((result.metric_value - result.baseline_value)
                                     / max(result.baseline_value, 1e-12), 4),
            "confidence": min(1.0, abs(result.metric_value - result.baseline_value)
                              * result.significance),
            "significance": result.significance,
        }

    @staticmethod
    def compare(results: list[ExperimentResult]) -> dict:
        """Compare multiple results and identify top hypotheses."""
        evaluations = [ResultEvaluator.evaluate(r) for r in results]
        sorted_pairs = sorted(zip(results, evaluations),
                              key=lambda x: x[1]["confidence"], reverse=True)

        top_hypothesis = None
        top_eval = None
        if sorted_pairs:
            top_result, top_eval = sorted_pairs[0]
            top_hypothesis = top_result.hypothesis_id

        confirmed = sum(1 for _, e in sorted_pairs if e["verdict"] == "confirmed")
        refuted = sum(1 for _, e in sorted_pairs if e["verdict"] == "refuted")

        return {
            "top_hypothesis_id": top_hypothesis,
            "top_evaluation": top_eval,
            "confirmed": confirmed,
            "refuted": refuted,
            "inconclusive": len(results) - confirmed - refuted,
            "ranked_results": [(r.hypothesis_id, e["verdict"], e["confidence"])
                              for r, e in sorted_pairs],
        }


# ═══════════════════════════════════════════════
# PaperWriter  (Lesson 55)
# ═══════════════════════════════════════════════

class PaperWriter:
    """Generate LaTeX research paper from hypothesis and results."""

    def __init__(self, authors=None):
        self.authors = authors or ["Anggira Research Team"]

    def write_paper(self, hypothesis: Hypothesis, refs: list[LiteratureRef],
                    results: list[ExperimentResult], evaluations: list[dict]) -> ResearchPaper:
        """Generate a complete research paper."""
        title = self._generate_title(hypothesis)
        abstract = self._generate_abstract(hypothesis, results, evaluations)

        sections = {
            "introduction": self._write_introduction(hypothesis),
            "related_work": self._write_related_work(refs),
            "methodology": self._write_methodology(hypothesis),
            "experiments": self._write_experiments(results),
            "results": self._write_results(results, evaluations),
            "discussion": self._write_discussion(hypothesis, evaluations),
            "conclusion": self._write_conclusion(hypothesis, evaluations),
        }

        citations = [ref.title for ref in refs]

        metrics_summary = {}
        for r in results:
            metrics_summary[f"{r.metric_name}_{r.hypothesis_id[:6]}"] = {
                "value": r.metric_value,
                "baseline": r.baseline_value,
            }

        confidence = sum(e.get("confidence", 0) for e in evaluations) / max(len(evaluations), 1)

        return ResearchPaper(
            title=title,
            authors=self.authors,
            abstract=abstract,
            sections=sections,
            citations=citations,
            metrics_summary=metrics_summary,
            confidence_score=round(confidence, 3),
        )

    def _generate_title(self, hypothesis):
        """Generate a title from the hypothesis."""
        words = hypothesis.claim.split()
        if len(words) > 6:
            title = "A Study on " + " ".join(words[:6]) + "..."
        else:
            title = "An Investigation of " + hypothesis.claim
        return title

    def _generate_abstract(self, hypothesis, results, evaluations):
        """Generate an abstract."""
        confirmed = sum(1 for e in evaluations if e["verdict"] == "confirmed")
        n_results = len(results)
        return (f"This paper investigates the hypothesis: {hypothesis.claim}. "
                f"Through {n_results} experiments, we find that "
                f"{confirmed} out of {n_results} predictions are supported by evidence. "
                f"Our findings contribute to understanding {hypothesis.domain}.")

    def _write_introduction(self, hypothesis):
        return (f"The question of {hypothesis.question} has received significant "
                f"attention in the {hypothesis.domain} community. In this work, we "
                f"systematically investigate the claim: {hypothesis.claim}.")

    def _write_related_work(self, refs):
        lines = [f"Our work builds upon {len(refs)} related studies:"]
        for ref in refs:
            lines.append(f"\\noindent {ref.title} ({ref.year}) — {ref.key_finding}")
        return "\n".join(lines)

    def _write_methodology(self, hypothesis):
        return (f"We evaluate the hypothesis using controlled experiments in the "
                f"{hypothesis.domain} domain. Our methodology follows standard practices "
                f"while controlling for confounding variables. The experimental setup "
                f"is designed to isolate the effect described in the hypothesis.")

    def _write_experiments(self, results):
        lines = [f"We conducted {len(results)} experiments:"]
        for i, r in enumerate(results, 1):
            if r.baseline_value:
                lines.append(f"\\noindent Experiment {i}: {r.metric_name} = {r.metric_value:.3f} "
                            f"(baseline: {r.baseline_value:.3f})")
            else:
                lines.append(f"\\noindent Experiment {i}: {r.metric_name} = {r.metric_value:.3f}")
        return "\n".join(lines)

    def _write_results(self, results, evaluations):
        lines = ["Results Summary:"]
        for r, e in zip(results, evaluations):
            status_emoji = "✓" if e["verdict"] == "confirmed" else "✗"
            lines.append(f"\\noindent {status_emoji} {r.hypothesis_id[:6]}: "
                        f"improvement={e['rel_improvement']:.1%}, "
                        f"verdict={e['verdict']}")
        return "\n".join(lines)

    def _write_discussion(self, hypothesis, evaluations):
        confirmed = sum(1 for e in evaluations if e["verdict"] == "confirmed")
        return (f"Our experiments provide {confirmed}/{len(evaluations)} confirmations "
                f"of the hypothesis. These results suggest that {hypothesis.claim[:50]}..."
                f" has partial empirical support. Future work should investigate "
                f"boundary conditions and possible confounding factors.")

    def _write_conclusion(self, hypothesis, evaluations):
        confirmed = sum(1 for e in evaluations if e["verdict"] == "confirmed")
        return (f"In conclusion, we find {'support' if confirmed > len(evaluations) / 2 else 'limited support'} "
                f"for the hypothesis that {hypothesis.claim[:60]}... "
                f"These findings advance our understanding of {hypothesis.domain}.")

    def to_latex(self, paper: ResearchPaper) -> str:
        """Render paper as LaTeX."""
        sections = []
        sections.append("\\documentclass[11pt]{article}")
        sections.append("\\usepackage{geometry}")
        sections.append("\\title{" + paper.title + "}")
        sections.append("\\author{" + ", ".join(paper.authors) + "}")
        sections.append("\\date{}")
        sections.append("\\begin{document}")
        sections.append("\\maketitle")
        sections.append("\\begin{abstract}")
        sections.append(paper.abstract)
        sections.append("\\end{abstract}")

        section_order = [
            ("Introduction", "introduction"),
            ("Related Work", "related_work"),
            ("Methodology", "methodology"),
            ("Experiments", "experiments"),
            ("Results", "results"),
            ("Discussion", "discussion"),
            ("Conclusion", "conclusion"),
        ]

        for heading, key in section_order:
            sections.append(f"\\section{{{heading}}}")
            sections.append(paper.sections[key])

        if paper.citations:
            sections.append("\\begin{thebibliography}{9}")
            for i, cite in enumerate(paper.citations, 1):
                sections.append(f"\\bibitem{{{i}}} {cite}")
            sections.append("\\end{thebibliography}")

        sections.append("\\end{document}")
        return "\n\n".join(sections)


# ═══════════════════════════════════════════════
# CriticLoop  (Lesson 56)
# ═══════════════════════════════════════════════

class CriticLoop:
    """Iterative improvement loop with convergence detection.

    Alternates between: propose experiment → run → evaluate → critic → refine.
    Converges when score plateaus or max iterations reached.
    """

    def __init__(self, hypothesis_generator, experiment_runner, result_evaluator,
                 max_iterations=5, convergence_threshold=0.01):
        self.generator = hypothesis_generator
        self.runner = experiment_runner
        self.evaluator = result_evaluator
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.history = []

    def run(self, hypothesis: Hypothesis) -> dict:
        """Run the critic loop for a hypothesis."""
        print(f"  CriticLoop starting for '{hypothesis.claim[:50]}...'")

        best_score = -1.0
        converged = False
        final_result = None

        for iteration in range(1, self.max_iterations + 1):
            # Run experiment
            result = self.runner.run(hypothesis)
            evaluation = self.evaluator.evaluate(result)
            score = evaluation["confidence"]

            self.history.append({
                "iteration": iteration,
                "hypothesis_id": hypothesis.id,
                "score": score,
                "verdict": evaluation["verdict"],
            })

            print(f"    Iteration {iteration}: score={score:.3f}, "
                  f"verdict={evaluation['verdict']}")

            # Check convergence
            if score > best_score:
                delta = score - best_score
                best_score = score
                final_result = (result, evaluation)
            else:
                delta = best_score - score

            if delta < self.convergence_threshold and iteration >= 2:
                converged = True
                print(f"    ✓ Converged (score delta={delta:.4f} < threshold)")
                break

        return {
            "hypothesis_id": hypothesis.id,
            "iterations": iteration,
            "converged": converged,
            "best_score": round(best_score, 4),
            "final_verdict": final_result[1]["verdict"] if final_result else "unknown",
            "history": self.history,
        }


# ═══════════════════════════════════════════════
# ResearchAgent  (Lesson 57) — End-to-end
# ═══════════════════════════════════════════════

class ResearchAgent:
    """End-to-end AI research agent.

    Full loop: Generate hypotheses → Retrieve literature →
    Run experiments → Evaluate results → Write paper.
    """

    def __init__(self, domain="nlp"):
        self.hypothesis_generator = HypothesisGenerator(domain=domain)
        self.literature_retriever = LiteratureRetriever()
        self.experiment_runner = ExperimentRunner()
        self.result_evaluator = ResultEvaluator()
        self.critic_loop = CriticLoop(
            self.hypothesis_generator,
            self.experiment_runner,
            self.result_evaluator,
            max_iterations=3,
        )
        self.paper_writer = PaperWriter(authors=["Anggira AI Research Agent"])
        self._session_id = str(uuid.uuid4())[:8]

        # Default benchmarks use simulated path in ExperimentRunner.run()
        pass

    def run_research_cycle(self, n_hypotheses=3):
        """Run a complete research cycle."""
        print(f"\n  Research Session: {self._session_id}")
        print(f"  Domain: {self.hypothesis_generator.domain}")

        # Phase 1: Generate hypotheses
        print(f"\n  Phase 1: Generating {n_hypotheses} hypotheses...")
        hypotheses = self.hypothesis_generator.generate(n=n_hypotheses)
        for h in hypotheses:
            print(f"    → {h.short()} (confidence={h.confidence:.2f})")

        # Phase 2: Retrieve literature
        print(f"\n  Phase 2: Retrieving related literature...")
        all_refs = {}
        for h in hypotheses:
            refs = self.literature_retriever.retrieve_for_hypothesis(h)
            all_refs[h.id] = refs
            print(f"    {h.short()}: {len(refs)} refs")

        # Phase 3: Run critic loop for each hypothesis
        print(f"\n  Phase 3: Running critic loop...")
        all_critic_results = []
        for h in hypotheses:
            result = self.critic_loop.run(h)
            all_critic_results.append(result)

        # Phase 4: Evaluate all results
        print(f"\n  Phase 4: Evaluating results...")
        all_evaluations = []
        all_results = []
        for h in hypotheses:
            for attempt in range(2):
                result = self.experiment_runner.run(h, "classification")
                evaluation = self.result_evaluator.evaluate(result)
                all_results.append(result)
                all_evaluations.append(evaluation)

        comparison = self.result_evaluator.compare(all_results)
        print(f"    Confirmed: {comparison['confirmed']}, "
              f"Refuted: {comparison['refuted']}, "
              f"Inconclusive: {comparison['inconclusive']}")

        # Phase 5: Write paper
        print(f"\n  Phase 5: Writing paper...")
        best_h = max(hypotheses, key=lambda h: h.confidence)
        best_refs = all_refs.get(best_h.id, [])
        paper = self.paper_writer.write_paper(best_h, best_refs, all_results, all_evaluations)
        latex = self.paper_writer.to_latex(paper)

        # Save paper
        paper_path = tempfile.mktemp(suffix=".tex", prefix=f"anggira_paper_{self._session_id}_")
        with open(paper_path, "w") as f:
            f.write(latex)
        print(f"    Paper saved: {paper_path}")

        return {
            "session_id": self._session_id,
            "hypotheses": [h.claim for h in hypotheses],
            "critic_results": all_critic_results,
            "comparison": comparison,
            "paper": {
                "title": paper.title,
                "abstract": paper.abstract,
                "confidence": paper.confidence_score,
                "citations": paper.citations,
            },
            "paper_path": paper_path,
        }

    def total_coverage(self):
        """Report coverage across all sub-modules."""
        return {
            "hypothesis_generator": ["generation", "diversity"],
            "literature_retriever": ["search", "retrieve", "summarize"],
            "experiment_runner": ["run", "run_batch", "register_benchmark"],
            "result_evaluator": ["evaluate", "compare"],
            "critic_loop": ["iterative_improvement", "convergence_detection"],
            "paper_writer": ["latex_generation", "sections", "citations"],
            "research_agent": ["end_to_end_research_cycle"],
        }


# ═══════════════════════════════════════════════
# DEMOS
# ═══════════════════════════════════════════════

def demo_hypothesis_generator():
    """Generate diverse hypotheses."""
    print("=== HypothesisGenerator Demo ===")
    gen = HypothesisGenerator(domain="nlp")
    hypotheses = gen.generate(n=4)
    print(f"  Generated {len(hypotheses)} hypotheses:")
    for h in hypotheses:
        print(f"    {h.id[:6]}: {h.claim[:70]}")
        print(f"           confidence={h.confidence:.2f}, domain={h.domain}")
    return True


def demo_literature_retriever():
    """Retrieve literature for a hypothesis."""
    print("\n=== LiteratureRetriever Demo ===")
    retriever = LiteratureRetriever()
    gen = HypothesisGenerator(domain="nlp")
    h = gen.generate(n=1)[0]

    refs = retriever.retrieve_for_hypothesis(h)
    print(f"  Found {len(refs)} refs for: {h.claim[:50]}...")
    for ref in refs:
        print(f"    [{ref.year}] {ref.title}: {ref.key_finding[:70]}")

    # Summary
    print(f"\n  {retriever.summarize(refs)}")
    return True


def demo_experiment_runner():
    """Run experiments."""
    print("\n=== ExperimentRunner Demo ===")
    gen = HypothesisGenerator(domain="vision")
    h = gen.generate(n=1)[0]

    runner = ExperimentRunner()
    result = runner.run(h, "default")
    print(f"  Experiment for: {h.claim[:60]}...")
    print(f"  Result: {result.metric_name}={result.metric_value:.4f}, "
          f"baseline={result.baseline_value}")
    print(f"  Significance: {result.significance:.3f}")
    return True


def demo_result_evaluator():
    """Evaluate experiment results."""
    print("\n=== ResultEvaluator Demo ===")
    runner = ExperimentRunner()
    gen = HypothesisGenerator(domain="nlp")
    hypotheses = gen.generate(n=3)

    results = runner.run_batch(hypotheses)
    for r in results:
        eval_result = ResultEvaluator.evaluate(r)
        print(f"  {r.hypothesis_id[:6]}: metric={r.metric_value:.3f}, "
              f"baseline={r.baseline_value}, "
              f"verdict={eval_result['verdict']}, "
              f"improvement={eval_result['rel_improvement']:.1%}")

    comparison = ResultEvaluator.compare(results)
    print(f"\n  Summary: {comparison['confirmed']} confirmed, "
          f"{comparison['refuted']} refuted, "
          f"{comparison['inconclusive']} inconclusive")
    return True


def demo_paper_writer():
    """Generate a full research paper."""
    print("\n=== PaperWriter Demo ===")
    gen = HypothesisGenerator(domain="nlp")
    h = gen.generate(n=1)[0]
    retriever = LiteratureRetriever()
    refs = retriever.retrieve_for_hypothesis(h)

    runner = ExperimentRunner()
    results = runner.run_batch([h] * 3)

    evaluations = [ResultEvaluator.evaluate(r) for r in results]

    writer = PaperWriter(authors=["Anggira Research Team"])
    paper = writer.write_paper(h, refs, results, evaluations)
    print(f"  Title: {paper.title}")
    print(f"  Abstract: {paper.abstract[:80]}...")
    print(f"  Sections: {list(paper.sections.keys())}")
    print(f"  Citations: {len(paper.citations)}")
    print(f"  Confidence: {paper.confidence_score:.3f}")

    # Show latex excerpt
    latex = writer.to_latex(paper)
    lines = latex.split("\n")
    print(f"  LaTeX: {len(lines)} lines")

    # Verify it's valid-ish
    has_document = "\\begin{document}" in latex and "\\end{document}" in latex
    print(f"  Valid LaTeX structure: {has_document}")
    return True


def demo_critic_loop():
    """Demo iterative improvement with convergence."""
    print("\n=== CriticLoop Demo ===")
    gen = HypothesisGenerator(domain="nlp")
    h = gen.generate(n=1)[0]

    runner = ExperimentRunner()
    evaluator = ResultEvaluator()
    loop = CriticLoop(gen, runner, evaluator, max_iterations=4)
    result = loop.run(h)
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Best score: {result['best_score']}")
    print(f"  Final verdict: {result['final_verdict']}")
    return True


def demo_research_agent():
    """End-to-end research cycle."""
    print("\n=== ResearchAgent Demo ===")
    agent = ResearchAgent(domain="nlp")
    result = agent.run_research_cycle(n_hypotheses=2)
    print(f"\n  Session: {result['session_id']}")
    print(f"  Paper: {result['paper']['title'][:60]}...")
    print(f"  Paper confidence: {result['paper']['confidence']:.3f}")
    print(f"  Coverage report: {len(agent.total_coverage())} modules")
    return True


def demo():
    """Run all research agent demos."""
    results = []

    print("=" * 60)
    print("Anggira Capstone — AI Research Agent (Phase 19, Lessons 50-57)")
    print("=" * 60)

    for demo_fn in [
        demo_hypothesis_generator,
        demo_literature_retriever,
        demo_experiment_runner,
        demo_result_evaluator,
        demo_paper_writer,
        demo_critic_loop,
        demo_research_agent,
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
