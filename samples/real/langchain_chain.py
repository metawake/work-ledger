#!/usr/bin/env python3
"""
Real LangChain integration test with Groq API.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from work_ledger import WorkLedger
from work_ledger.integrations.langchain import wrap_chain
from work_ledger.testing import RunDiff

# Groq API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Set GROQ_API_KEY: export GROQ_API_KEY=your-key")
    sys.exit(1)

try:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    print("❌ Install: pip install langchain langchain-groq")
    sys.exit(1)

print("=" * 60)
print("REAL LANGCHAIN TEST: Groq + Work Ledger")
print("=" * 60)
print()

# Create LangChain components
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer in one sentence: {question}"
)

chain = prompt | llm | StrOutputParser()

# Wrap with Work Ledger
ledger = WorkLedger(store=":memory:")
wrapped = wrap_chain(chain, ledger, run_name="qa-chain")

# Test 1: Basic chain
print("Test 1: Basic Chain Invocation")
print("-" * 40)

result1 = wrapped.invoke({"question": "What is Python?"})
print(f"Response: {result1}")

run1 = ledger.list_runs()[0]
print(f"Recorded: ✓ {run1.status.value}")
print(f"Input: {run1.inputs}")
print()

# Test 2: Different question
print("Test 2: Different Question")
print("-" * 40)

result2 = wrapped.invoke({"question": "What is JavaScript?"})
print(f"Response: {result2}")

run2 = ledger.list_runs()[1]
print(f"Recorded: ✓ {run2.status.value}")
print()

# Test 3: Compare runs
print("Test 3: Diff Between Runs")
print("-" * 40)

diff = RunDiff(run1, run2)
print(f"Similarity: {diff.similarity:.1%}")

# Show what changed
if diff.input_diff.get("changed"):
    print(f"Input changed:")
    for key, val in diff.input_diff["changed"].items():
        print(f"  {key}: '{val['expected']}' → '{val['actual']}'")

if diff.output_diff.get("changed"):
    print(f"Output changed: (different answers as expected)")
print()

# Test 4: Chain with error handling
print("Test 4: Multiple Runs Summary")
print("-" * 40)

wrapped.invoke({"question": "What is 2+2?"})
wrapped.invoke({"question": "Name a color."})

print(f"Total runs recorded: {len(ledger.list_runs())}")
print()

for i, run in enumerate(ledger.list_runs(), 1):
    question = run.inputs.get("question", "N/A")
    result = run.outputs.get("result", "")[:60]
    print(f"  {i}. Q: {question}")
    print(f"     A: {result}...")
    print()

print("=" * 60)
print("✓ LangChain integration works with real API!")
print("=" * 60)
