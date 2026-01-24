#!/usr/bin/env python3
"""
Real integration test with Groq API.

This tests the actual Work Ledger integration with real API calls.
"""

import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from work_ledger import WorkLedger, wrap_openai
from work_ledger.testing import RunDiff, format_diff

# Groq API key (get free at https://console.groq.com)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Set GROQ_API_KEY: export GROQ_API_KEY=your-key")
    print("   Get free key at: https://console.groq.com")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("❌ Install openai: pip install openai")
    sys.exit(1)

# Create Groq client (OpenAI-compatible)
client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

ledger = WorkLedger(store=":memory:")
wrapped = wrap_openai(client, ledger)

print("=" * 60)
print("REAL API TEST: Groq + Work Ledger")
print("=" * 60)
print()

# Test 1: Basic completion
print("Test 1: Basic Completion")
print("-" * 40)

response = wrapped.chat.completions.create(
    messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
    model="llama-3.1-8b-instant"
)

print(f"Response: {response.choices[0].message.content}")
run1 = ledger.list_runs()[0]
print(f"Recorded: ✓ {run1.status.value}")
print(f"Tokens: {run1.metrics.total_tokens}")
print()

# Test 2: Different prompt, compare runs
print("Test 2: Different Prompt")
print("-" * 40)

response2 = wrapped.chat.completions.create(
    messages=[{"role": "user", "content": "What is 3+3? Answer with just the number."}],
    model="llama-3.1-8b-instant"
)

print(f"Response: {response2.choices[0].message.content}")
run2 = ledger.list_runs()[1]
print(f"Recorded: ✓ {run2.status.value}")
print()

# Test 3: Diff between runs
print("Test 3: Diff Between Runs")
print("-" * 40)

diff = RunDiff(run1, run2)
print(f"Similarity: {diff.similarity:.1%}")
print(f"Inputs changed: {diff.input_diff}")
print(f"Outputs changed: {diff.output_diff}")
print()

# Test 4: Multi-turn conversation
print("Test 4: Multi-turn Conversation")
print("-" * 40)

response3 = wrapped.chat.completions.create(
    messages=[
        {"role": "user", "content": "Remember the number 42."},
        {"role": "assistant", "content": "I'll remember 42."},
        {"role": "user", "content": "What number did I ask you to remember?"},
    ],
    model="llama-3.1-8b-instant"
)

print(f"Response: {response3.choices[0].message.content}")
run3 = ledger.list_runs()[2]
print(f"Messages recorded: {len(run3.inputs.get('messages', []))}")
print()

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total runs recorded: {len(ledger.list_runs())}")
print()

for i, run in enumerate(ledger.list_runs(), 1):
    content = run.outputs.get("content", "")[:50]
    print(f"  {i}. [{run.status.value}] tokens={run.metrics.total_tokens} | {content}...")

print()
print("✓ All real API tests passed!")
print("✓ Work Ledger recording works with real Groq calls!")
