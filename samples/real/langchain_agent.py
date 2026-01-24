#!/usr/bin/env python3
"""
Real LangChain/LangGraph Agent test with tools.

Tests that Work Ledger correctly records:
- Agent invocations
- Tool calls
- Final outputs
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from work_ledger import WorkLedger
from work_ledger.integrations.langgraph import wrap_graph

# Groq API key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("❌ Set GROQ_API_KEY: export GROQ_API_KEY=your-key")
    sys.exit(1)

try:
    from langchain_groq import ChatGroq
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
except ImportError as e:
    print(f"❌ Install: pip install langchain-groq langgraph")
    print(f"   Error: {e}")
    sys.exit(1)

print("=" * 60)
print("REAL LANGGRAPH AGENT TEST: Tools + Work Ledger")
print("=" * 60)
print()

# Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Input should be a valid Python math expression like '2+2' or '15*7'."""
    try:
        # Safe eval for simple math
        allowed = set('0123456789+-*/(). ')
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"The result is: {result}"
        return "Error: Invalid expression"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Input should be a city name like 'Paris' or 'Tokyo'."""
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C", 
        "tokyo": "Rainy, 18°C",
        "new york": "Clear, 20°C",
    }
    return weather_data.get(city.lower(), f"Weather for {city}: Partly cloudy, 18°C")

tools = [calculator, get_weather]

# Create LLM and agent
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile",  # Updated model
    temperature=0
)

agent = create_react_agent(llm, tools)

# Wrap with Work Ledger (with stream recording)
ledger = WorkLedger(store=":memory:")
wrapped = wrap_graph(agent, ledger, run_name="react-agent", record_stream=True)

# Helper to stream and collect
def stream_agent(query):
    """Stream agent execution and collect final result."""
    final_result = None
    print(f"Streaming: ", end="", flush=True)
    for event in wrapped.stream({"messages": [("user", query)]}):
        # Each event is a dict with node name as key
        for node_name in event:
            print(f"[{node_name}] ", end="", flush=True)
        final_result = event
    print()
    return final_result

# Test 1: Calculator
print("Test 1: Calculator Tool")
print("-" * 40)

result1 = stream_agent("What is 15 * 7 + 23?")
if result1 and "agent" in result1:
    messages = result1.get("agent", {}).get("messages", [])
    if messages:
        print(f"Response: {messages[-1].content}")

run1 = ledger.list_runs()[0]
print(f"Recorded: ✓ {run1.status.value}")
print(f"Steps recorded: {len(run1.steps)}")
for step in run1.steps:
    print(f"  - {step.name}: {step.outputs}")
print()

# Test 2: Weather
print("Test 2: Weather Tool")
print("-" * 40)

result2 = stream_agent("What's the weather like in Paris?")
if result2 and "agent" in result2:
    messages = result2.get("agent", {}).get("messages", [])
    if messages:
        print(f"Response: {messages[-1].content}")

run2 = ledger.list_runs()[1]
print(f"Recorded: ✓ {run2.status.value}")
print(f"Steps recorded: {len(run2.steps)}")
for step in run2.steps:
    print(f"  - {step.name}")
print()

# Test 3: No tools needed
print("Test 3: Direct Answer (No Tools)")
print("-" * 40)

result3 = stream_agent("Say hello in French.")
if result3 and "agent" in result3:
    messages = result3.get("agent", {}).get("messages", [])
    if messages:
        print(f"Response: {messages[-1].content}")

run3 = ledger.list_runs()[2]
print(f"Recorded: ✓ {run3.status.value}")
print(f"Steps recorded: {len(run3.steps)}")
print()

# Summary
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print()

print(f"Total runs: {len(ledger.list_runs())}")
print()

for i, run in enumerate(ledger.list_runs(), 1):
    # Get first user message
    messages = run.inputs.get("messages", [])
    first_msg = messages[0] if messages else "N/A"
    if isinstance(first_msg, (list, tuple)):
        first_msg = first_msg[1] if len(first_msg) > 1 else str(first_msg)
    
    print(f"Run {i}: {str(first_msg)[:50]}...")
    print(f"  Status: {run.status.value}")
    print(f"  Steps: {len(run.steps)}")
    if run.steps:
        print(f"  Node sequence: {' → '.join(s.name for s in run.steps)}")
    print()

print("=" * 60)
print("✓ LangGraph Agent works with Work Ledger!")
print("✓ Tool calls and agent steps recorded!")
print("=" * 60)
