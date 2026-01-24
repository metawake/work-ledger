#!/usr/bin/env python3
"""
Sample 04: Debug Workflow
=========================

Learn how to understand what your agent actually did.

This sample shows:
- Causal chain analysis (what triggered what)
- Finding specific step types
- Aggregating metrics
- Inspecting run structure

Run:
    python 04_debug_workflow.py
"""

from work_ledger import WorkLedger, StepKind


# --- Scenario: Debug a Complex Agent ---

print("=" * 50)
print("SCENARIO: Debug a Complex Agent Workflow")
print("=" * 50)
print()
print("Your agent processed a user request but took too long.")
print("Let's understand what happened.")
print()


# --- Simulate a Complex Multi-Step Agent ---

ledger = WorkLedger(store=":memory:")

with ledger.run(name="research-agent") as run:
    run.record_input({"query": "Compare Tesla and Rivian stock performance"})
    
    # Step 1: Parse intent
    with run.step(name="parse-intent", kind="model") as s1:
        s1.record_input({"query": "Compare Tesla and Rivian stock performance"})
        s1.record_output({"intent": "stock_comparison", "entities": ["TSLA", "RIVN"]})
        s1.record_metrics(prompt_tokens=50, completion_tokens=20, cost=0.001)
    
    # Step 2: Fetch Tesla data (triggered by parse-intent)
    with run.step(name="fetch-tsla", kind="tool", caused_by=s1.step_id) as s2:
        s2.record_input({"symbol": "TSLA"})
        s2.record_output({"price": 248.50, "change": "+2.3%"})
        s2.record_metrics(latency_ms=150)
    
    # Step 3: Fetch Rivian data (triggered by parse-intent)
    with run.step(name="fetch-rivn", kind="tool", caused_by=s1.step_id) as s3:
        s3.record_input({"symbol": "RIVN"})
        s3.record_output({"price": 14.20, "change": "-1.5%"})
        s3.record_metrics(latency_ms=2500)  # This one was slow!
    
    # Step 4: Retrieve news context
    with run.step(name="retrieve-news", kind="retrieval") as s4:
        s4.record_input({"query": "Tesla Rivian news"})
        s4.record_output({"articles": ["Tesla Q4 earnings beat...", "Rivian layoffs..."]})
        s4.record_metrics(latency_ms=300)
    
    # Step 5: Generate comparison (triggered by all previous)
    with run.step(name="generate-comparison", kind="model", caused_by=s4.step_id) as s5:
        s5.record_input({"tsla": s2.outputs, "rivn": s3.outputs, "news": s4.outputs})
        s5.record_output({"comparison": "Tesla up 2.3% while Rivian down 1.5%..."})
        s5.record_metrics(prompt_tokens=500, completion_tokens=150, cost=0.008)
    
    run.record_output({"response": s5.outputs["comparison"]})

research_run = ledger.get_run(run.run_id)


# --- Debug: Understand the Workflow ---

print("=" * 50)
print("WORKFLOW STRUCTURE")
print("=" * 50)
print()

print(f"Run: {research_run.name}")
print(f"Input: {research_run.inputs}")
print(f"Output: {research_run.outputs}")
print(f"Duration: {research_run.duration_ms:.0f}ms")
print()

print("Steps:")
for i, step in enumerate(research_run.steps, 1):
    caused = f" (caused by: {step.caused_by[:8]}...)" if step.caused_by else ""
    print(f"  {i}. [{step.kind.value:10}] {step.name}{caused}")


# --- Debug: Find Slow Steps ---

print()
print("=" * 50)
print("PERFORMANCE ANALYSIS")
print("=" * 50)
print()

print("Steps by latency:")
steps_with_latency = [
    (s.name, s.metrics.latency_ms)
    for s in research_run.steps
    if s.metrics.latency_ms is not None
]
for name, latency in sorted(steps_with_latency, key=lambda x: x[1], reverse=True):
    indicator = "⚠️ SLOW" if latency > 1000 else "✓"
    print(f"  {name}: {latency:.0f}ms {indicator}")


# --- Debug: Analyze by Step Type ---

print()
print("=" * 50)
print("ANALYSIS BY STEP TYPE")
print("=" * 50)
print()

for kind in [StepKind.MODEL, StepKind.TOOL, StepKind.RETRIEVAL]:
    steps = research_run.get_steps_by_kind(kind)
    if steps:
        total_tokens = sum(s.metrics.total_tokens for s in steps)
        total_cost = sum(s.metrics.cost or 0 for s in steps)
        print(f"{kind.value.upper()}:")
        print(f"  Count: {len(steps)}")
        print(f"  Tokens: {total_tokens}")
        print(f"  Cost: ${total_cost:.4f}")
        print()


# --- Debug: Trace Causal Chain ---

print("=" * 50)
print("CAUSAL CHAIN ANALYSIS")
print("=" * 50)
print()

# Find what caused the final step
final_step = research_run.steps[-1]
print(f"Tracing what led to '{final_step.name}':")
print()

chain = research_run.get_causal_chain(final_step.step_id)
for i, step in enumerate(chain):
    indent = "  " * i
    arrow = "→ " if i > 0 else ""
    print(f"{indent}{arrow}{step.name} ({step.kind.value})")


# --- Debug: Aggregated Metrics ---

print()
print("=" * 50)
print("AGGREGATED METRICS")
print("=" * 50)
print()

metrics = research_run.aggregate_metrics()
print(f"Total tokens:      {metrics.total_tokens}")
print(f"  Prompt tokens:   {metrics.prompt_tokens}")
print(f"  Completion:      {metrics.completion_tokens}")
print(f"Total cost:        ${metrics.cost:.4f}")


print()
print("=" * 50)
print("WHAT YOU LEARNED")
print("=" * 50)
print("""
1. get_steps_by_kind() - Find all model/tool/retrieval calls
2. get_causal_chain() - Trace what triggered what
3. aggregate_metrics() - Total tokens and cost
4. Step timing reveals performance bottlenecks

This is forensic debugging:
- "Why did step X run?"
- "What was slow?"
- "How much did this cost?"
- "What was the decision path?"
""")
