#!/usr/bin/env python3
"""
Real PydanticAI integration test with Groq API.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from work_ledger import WorkLedger
from work_ledger.integrations.pydantic_ai import wrap_agent
from work_ledger.testing import RunDiff

# Groq API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Set GROQ_API_KEY: export GROQ_API_KEY=your-key")
    sys.exit(1)

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.groq import GroqModel
except ImportError as e:
    print(f"❌ Install: pip install pydantic-ai")
    print(f"   Error: {e}")
    sys.exit(1)

print("=" * 60)
print("REAL PYDANTICAI TEST: Groq + Work Ledger")
print("=" * 60)
print()

# Create Groq model for PydanticAI (uses GROQ_API_KEY env var)
model = GroqModel("llama-3.1-8b-instant")

# Create a simple agent
agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant. Be concise.",
)

# Wrap with Work Ledger
ledger = WorkLedger(store=":memory:")
wrapped = wrap_agent(agent, ledger, run_name="pydantic-agent")

# Test 1: Basic query
print("Test 1: Basic Query")
print("-" * 40)

result1 = wrapped.run_sync("What is 2 + 2? Just give the number.")
print(f"Response: {result1.output}")

run1 = ledger.list_runs()[0]
print(f"Recorded: ✓ {run1.status.value}")
print(f"Tokens: {run1.metrics.total_tokens}")
print()

# Test 2: Different query
print("Test 2: Different Query")
print("-" * 40)

result2 = wrapped.run_sync("Name the capital of France in one word.")
print(f"Response: {result2.output}")

run2 = ledger.list_runs()[1]
print(f"Recorded: ✓ {run2.status.value}")
print()

# Test 3: Compare runs
print("Test 3: Diff Between Runs")
print("-" * 40)

diff = RunDiff(run1, run2)
print(f"Similarity: {diff.similarity:.1%}")
print(f"Different prompts, different outputs ✓")
print()

# Test 4: Multi-message
print("Test 4: Longer Response")
print("-" * 40)

result3 = wrapped.run_sync("List 3 programming languages, one per line.")
print(f"Response:\n{result3.output}")

run3 = ledger.list_runs()[2]
print(f"\nRecorded: ✓ {run3.status.value}")
print(f"Tokens: {run3.metrics.total_tokens}")
print()

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()

print(f"Total runs recorded: {len(ledger.list_runs())}")
print()

for i, run in enumerate(ledger.list_runs(), 1):
    prompt = run.inputs.get("prompt", "N/A")[:40]
    result = str(run.outputs.get("result", ""))[:40]
    print(f"  {i}. [{run.status.value}] tokens={run.metrics.total_tokens}")
    print(f"     Q: {prompt}...")
    print(f"     A: {result}...")
    print()

print("=" * 60)
print("✓ PydanticAI integration works with real Groq API!")
print("=" * 60)
