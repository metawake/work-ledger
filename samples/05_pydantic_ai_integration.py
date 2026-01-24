#!/usr/bin/env python3
"""
Sample 05: PydanticAI Integration
=================================

Wrap your PydanticAI agent to automatically record runs.

This sample shows:
- Wrapping an agent with Work Ledger
- Automatic recording of inputs/outputs
- Tool call tracking
- Token usage metrics

Run:
    python 05_pydantic_ai_integration.py

Note: This sample uses mock objects. With real PydanticAI:
    pip install pydantic-ai
"""

from work_ledger import WorkLedger
from work_ledger.integrations.pydantic_ai import wrap_agent


# --- Mock PydanticAI Agent (for demo without real API) ---

class MockUsage:
    def __init__(self):
        self.request_tokens = 150
        self.response_tokens = 75
        self.total_tokens = 225


class MockToolCall:
    def __init__(self, tool_name, args, result):
        self.tool_name = tool_name
        self.args = args
        self.result = result


class MockResult:
    def __init__(self, data, tool_calls=None):
        self.data = data
        self._tool_calls = tool_calls or []
        self.usage = MockUsage()
    
    def all_messages_json(self):
        return [{"role": "assistant", "content": self.data}]


class MockAgent:
    """Simulates a PydanticAI Agent."""
    
    def __init__(self, name="weather-agent"):
        self.name = name
    
    def run_sync(self, prompt, **kwargs):
        # Simulate tool call + response
        if "weather" in prompt.lower():
            return MockResult(
                data="It's 22°C and sunny in Paris!",
                tool_calls=[
                    MockToolCall(
                        tool_name="get_weather",
                        args={"city": "Paris"},
                        result="sunny, 22°C"
                    )
                ]
            )
        return MockResult(data=f"Response to: {prompt}")


# --- Demo ---

print("=" * 60)
print("PydanticAI Integration Demo")
print("=" * 60)
print()

# Create ledger
ledger = WorkLedger(store=":memory:")

# Create agent (in real usage: Agent("openai:gpt-4"))
agent = MockAgent(name="weather-agent")

# Wrap the agent - this is the key line!
wrapped = wrap_agent(agent, ledger)

print("1. Wrap your agent:")
print("   wrapped = wrap_agent(agent, ledger)")
print()

# Run the agent - it's recorded automatically
print("2. Use it normally:")
print("   result = wrapped.run_sync('What is the weather in Paris?')")
print()

result = wrapped.run_sync("What is the weather in Paris?")

print(f"   Result: {result.data}")
print()

# Check what was recorded
print("=" * 60)
print("What Work Ledger Recorded")
print("=" * 60)
print()

run = ledger.list_runs()[0]

print(f"Run: {run.name}")
print(f"Status: {run.status.value}")
print(f"Input: {run.inputs}")
print(f"Output: {run.outputs}")
print()

print("Steps:")
for step in run.steps:
    print(f"  - [{step.kind.value}] {step.name}")
    if step.inputs:
        print(f"    Input: {step.inputs}")
    if step.outputs:
        print(f"    Output: {step.outputs}")
    if step.metrics.total_tokens > 0:
        print(f"    Tokens: {step.metrics.total_tokens}")
print()

print("Metrics:")
print(f"  Prompt tokens: {run.metrics.prompt_tokens}")
print(f"  Completion tokens: {run.metrics.completion_tokens}")
print(f"  Total tokens: {run.metrics.total_tokens}")
print()

# Multiple runs
print("=" * 60)
print("Track Multiple Runs")
print("=" * 60)
print()

wrapped.run_sync("What's the capital of France?")
wrapped.run_sync("Tell me a joke")

print(f"Total runs recorded: {len(ledger.list_runs())}")
for r in ledger.list_runs():
    print(f"  - {r.run_id[:8]}... | {r.inputs.get('prompt', '')[:30]}...")

print()
print("=" * 60)
print("Real PydanticAI Usage")
print("=" * 60)
print("""
from pydantic_ai import Agent
from work_ledger import WorkLedger
from work_ledger.integrations.pydantic_ai import wrap_agent

# Your real agent
agent = Agent("openai:gpt-4o")

# Wrap it
ledger = WorkLedger(store="./runs")
wrapped = wrap_agent(agent, ledger)

# Use normally - everything is recorded
result = await wrapped.run("Analyze this document...")

# Later: debug, diff, replay
runs = ledger.list_runs()
""")
