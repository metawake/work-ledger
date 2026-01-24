#!/usr/bin/env python3
"""
Sample 06: LangGraph Integration
=================================

Wrap your LangGraph to automatically record runs.

This sample shows:
- Wrapping a compiled graph with Work Ledger
- Automatic recording of inputs/outputs
- Node execution tracking with stream()
- State change recording

Run:
    python 06_langgraph_integration.py

Note: This sample uses mock objects. With real LangGraph:
    pip install langgraph
"""

from work_ledger import WorkLedger
from work_ledger.integrations.langgraph import wrap_graph


# --- Mock LangGraph (for demo without real dependency) ---

class MockCompiledGraph:
    """Simulates a compiled LangGraph."""
    
    def __init__(self, name="agent-workflow"):
        self.name = name
    
    def invoke(self, state: dict, config: dict = None) -> dict:
        """Simulate graph execution."""
        messages = state.get("messages", [])
        
        # Simulate agent processing
        if "weather" in str(messages).lower():
            return {
                "messages": messages + ["Checking weather...", "It's sunny!"],
                "result": "Weather retrieved successfully"
            }
        return {
            "messages": messages + ["Processing..."],
            "result": "Completed"
        }
    
    def stream(self, state: dict, config: dict = None):
        """Simulate streaming execution with node outputs."""
        # Simulate node-by-node execution
        yield {"agent": {"thought": "Let me process this..."}}
        yield {"tools": {"action": "search", "result": "Found data"}}
        yield {"agent": {"response": "Here's what I found"}}
        yield {"__end__": {"result": "Workflow complete"}}


# --- Demo ---

print("=" * 60)
print("LangGraph Integration Demo")
print("=" * 60)
print()

# Create ledger
ledger = WorkLedger(store=":memory:")

# Create graph (in real usage: StateGraph(...).compile())
graph = MockCompiledGraph(name="research-agent")

# Wrap the graph - this is the key line!
wrapped = wrap_graph(graph, ledger)

print("1. Wrap your graph:")
print("   wrapped = wrap_graph(graph, ledger)")
print()

# Run the graph - it's recorded automatically
print("2. Use it normally:")
print("   result = wrapped.invoke({'messages': ['What is the weather?']})")
print()

result = wrapped.invoke({"messages": ["What is the weather?"]})

print(f"   Result: {result['result']}")
print(f"   Messages: {result['messages']}")
print()

# Check what was recorded
print("=" * 60)
print("What Work Ledger Recorded")
print("=" * 60)
print()

run = ledger.list_runs()[0]

print(f"Run: {run.name}")
print(f"Status: {run.status.value}")
print()
print(f"Input State:")
for k, v in run.inputs.items():
    print(f"  {k}: {v}")
print()
print(f"Output State:")
for k, v in run.outputs.items():
    print(f"  {k}: {v}")
print()

# Streaming with node tracking
print("=" * 60)
print("Stream with Node Tracking")
print("=" * 60)
print()

# Enable stream recording
wrapped_stream = wrap_graph(graph, ledger, record_stream=True)

print("wrapped = wrap_graph(graph, ledger, record_stream=True)")
print()
print("Streaming nodes:")
for event in wrapped_stream.stream({"messages": ["Research AI trends"]}):
    for node, output in event.items():
        print(f"  [{node}] → {output}")
print()

# Check recorded steps
run = ledger.list_runs()[-1]  # Get the streaming run
print(f"Recorded {len(run.steps)} node steps:")
for step in run.steps:
    print(f"  - {step.name}: {step.outputs}")
print()

# Multiple runs
print("=" * 60)
print("Track Multiple Runs")
print("=" * 60)
print()

wrapped.invoke({"query": "analyze data"})
wrapped.invoke({"query": "generate report"})

print(f"Total runs recorded: {len(ledger.list_runs())}")
for r in ledger.list_runs():
    inputs = str(r.inputs)[:40] + "..." if len(str(r.inputs)) > 40 else str(r.inputs)
    print(f"  - {r.run_id[:8]}... | {r.status.value} | {inputs}")

print()
print("=" * 60)
print("Real LangGraph Usage")
print("=" * 60)
print("""
from langgraph.graph import StateGraph, START
from work_ledger import WorkLedger
from work_ledger.integrations.langgraph import wrap_graph

# Your real graph
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
graph = builder.compile()

# Wrap it
ledger = WorkLedger(store="./runs")
wrapped = wrap_graph(graph, ledger)

# Use normally - everything is recorded
result = wrapped.invoke({"messages": [HumanMessage("Hello!")]})

# Later: debug, diff, replay
runs = ledger.list_runs()
""")
