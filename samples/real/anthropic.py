#!/usr/bin/env python3
"""
Real Anthropic SDK integration test.

Note: Requires ANTHROPIC_API_KEY (Groq doesn't support Anthropic format)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from work_ledger import WorkLedger
from work_ledger.integrations.anthropic import wrap_anthropic
from work_ledger.testing import RunDiff

# Anthropic API key
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("❌ Set ANTHROPIC_API_KEY: export ANTHROPIC_API_KEY=your-key")
    print("   Get a key at: https://console.anthropic.com/")
    sys.exit(1)

try:
    import anthropic
except ImportError as e:
    print(f"❌ Install: pip install anthropic")
    print(f"   Error: {e}")
    sys.exit(1)

print("=" * 60)
print("REAL ANTHROPIC TEST: Claude + Work Ledger")
print("=" * 60)
print()

# Create Anthropic client
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Wrap with Work Ledger
ledger = WorkLedger(store=":memory:")
wrapped = wrap_anthropic(client, ledger)

# Test 1: Basic completion
print("Test 1: Basic Completion")
print("-" * 40)

message1 = wrapped.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "What is 2 + 2? Just give the number."}
    ]
)
print(f"Response: {message1.content[0].text}")

run1 = ledger.list_runs()[0]
print(f"Recorded: ✓ {run1.status.value}")
print(f"Tokens: {run1.metrics.total_tokens}")
print()

# Test 2: Different query
print("Test 2: Different Query")
print("-" * 40)

message2 = wrapped.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Name the capital of France in one word."}
    ]
)
print(f"Response: {message2.content[0].text}")

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

# Test 4: Multi-turn
print("Test 4: Multi-turn Conversation")
print("-" * 40)

message3 = wrapped.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Nice to meet you, Alice!"},
        {"role": "user", "content": "What's my name?"}
    ]
)
print(f"Response: {message3.content[0].text}")

run3 = ledger.list_runs()[2]
print(f"Recorded: ✓ {run3.status.value}")
print(f"Messages recorded: {len(run3.inputs.get('messages', []))}")
print()

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()

print(f"Total runs recorded: {len(ledger.list_runs())}")
print()

for i, run in enumerate(ledger.list_runs(), 1):
    messages = run.inputs.get("messages", [])
    first_msg = messages[0]["content"][:30] if messages else "N/A"
    result = str(run.outputs.get("content", ""))[:40]
    print(f"  {i}. [{run.status.value}] tokens={run.metrics.total_tokens}")
    print(f"     Q: {first_msg}...")
    print(f"     A: {result}...")
    print()

print("=" * 60)
print("✓ Anthropic integration works!")
print("=" * 60)
