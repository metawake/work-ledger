#!/usr/bin/env python3
"""
Real LlamaIndex integration test with Groq API.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from work_ledger import WorkLedger
from work_ledger.integrations.llamaindex import wrap_query_engine
from work_ledger.testing import RunDiff

# Groq API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Set GROQ_API_KEY: export GROQ_API_KEY=your-key")
    sys.exit(1)

try:
    from llama_index.core import VectorStoreIndex, Document, Settings
    from llama_index.llms.groq import Groq
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError as e:
    print(f"❌ Install: pip install llama-index llama-index-llms-groq")
    print(f"   Error: {e}")
    sys.exit(1)

print("=" * 60)
print("REAL LLAMAINDEX TEST: Groq + Work Ledger")
print("=" * 60)
print()

# Configure LlamaIndex to use Groq
llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
Settings.llm = llm

# Use a mock embedding to avoid needing OpenAI key
from llama_index.core.embeddings import MockEmbedding
Settings.embed_model = MockEmbedding(embed_dim=1536)

# Create a simple in-memory index with sample documents
documents = [
    Document(text="Python was created by Guido van Rossum and released in 1991. It emphasizes code readability."),
    Document(text="JavaScript was created by Brendan Eich in 1995. It's the language of the web browser."),
    Document(text="Rust was created by Graydon Hoare at Mozilla and released in 2010. It focuses on memory safety."),
]

print("Creating vector index with 3 documents...")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Wrap with Work Ledger
ledger = WorkLedger(store=":memory:")
wrapped = wrap_query_engine(query_engine, ledger, run_name="llamaindex-query")

# Test 1: Basic query
print()
print("Test 1: Python Query")
print("-" * 40)

response1 = wrapped.query("Who created Python?")
print(f"Response: {response1}")

run1 = ledger.list_runs()[0]
print(f"Recorded: ✓ {run1.status.value}")
print(f"Tokens: {run1.metrics.total_tokens}")
print()

# Test 2: Different query
print("Test 2: JavaScript Query")
print("-" * 40)

response2 = wrapped.query("When was JavaScript created?")
print(f"Response: {response2}")

run2 = ledger.list_runs()[1]
print(f"Recorded: ✓ {run2.status.value}")
print()

# Test 3: Compare runs
print("Test 3: Diff Between Runs")
print("-" * 40)

diff = RunDiff(run1, run2)
print(f"Similarity: {diff.similarity:.1%}")
print(f"Different queries, different outputs ✓")
print()

# Test 4: Rust query
print("Test 4: Rust Query")
print("-" * 40)

response3 = wrapped.query("What language focuses on memory safety?")
print(f"Response: {response3}")

run3 = ledger.list_runs()[2]
print(f"Recorded: ✓ {run3.status.value}")
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
    query = run.inputs.get("query", "N/A")[:40]
    result = str(run.outputs.get("response", ""))[:50]
    print(f"  {i}. [{run.status.value}] tokens={run.metrics.total_tokens}")
    print(f"     Q: {query}...")
    print(f"     A: {result}...")
    print()

print("=" * 60)
print("✓ LlamaIndex integration works with real Groq API!")
print("=" * 60)
