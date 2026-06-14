"""
Anggira — Capstone Projects Orchestrator

Phase 19: Capstone Projects (Lessons 01-87)

Covering 8 tracks via these sub-modules:
- capstone_agent_harness.py (Lessons 20-29): Agent Harness / Tool-Use System
- capstone_training.py     (Lessons 42-49): Training Pipeline Components
- capstone_research.py     (Lessons 50-57): AI Research Scientist
- capstone_distributed.py  (Lessons 76-81): Distributed Training

Existing modules cover the remaining lessons:
- nlp.py  → Tokenizer / BPE (Lessons 30-37)
- safety.py → Finetuning / DPO / Eval (Lessons 38-41)
- multimodal.py / cv.py → Vision-Language (Lessons 58-63)
- llm_eng.py → RAG (Lessons 64-69), Eval Harness (Lessons 70-75)
- safety.py → Safety & Alignment (Lessons 82-87)
- agent.py → Agent concepts (Lessons 01-19 foundational)
"""

import sys


def demo():
    """Run all capstone demos."""
    results = []

    modules = [
        ("Agent Harness (L20-29)", "capstone_agent_harness"),
        ("Training Pipeline (L42-49)", "capstone_training"),
        ("Research Agent (L50-57)", "capstone_research"),
        ("Distributed Training (L76-81)", "capstone_distributed"),
    ]

    print("=" * 60)
    print("Anggira Capstone — Phase 19 All Tracks")
    print("=" * 60)

    for label, module_name in modules:
        print(f"\n{'─' * 60}")
        print(f"Running: {label}")
        print(f"{'─' * 60}")
        try:
            mod = __import__(f"anggira.{module_name}", fromlist=["demo"])
            mod.demo()
            results.append((label, "PASS"))
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((label, f"FAIL: {e}"))

    print("\n" + "=" * 60)
    print("Phase 19 Overall Results:")
    for label, status in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {label}: {status}")
    print("=" * 60)
    return all(r == "PASS" for _, r in results)


if __name__ == "__main__":
    demo()
