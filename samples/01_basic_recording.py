#!/usr/bin/env python3
"""
Sample 01: Basic Recording
==========================

Learn how to record agent runs in 2 minutes.

Run:
    python 01_basic_recording.py
"""

from work_ledger import WorkLedger

# Create a ledger (in-memory for this demo)
ledger = WorkLedger(store=":memory:")


# Simulate an agent workflow
def fetch_weather(city: str) -> dict:
    """Pretend to call a weather API."""
    return {"city": city, "temp": 22, "condition": "sunny"}


def generate_response(weather: dict) -> str:
    """Pretend to call an LLM."""
    return f"It's {weather['temp']}°C and {weather['condition']} in {weather['city']}."


# Record the run
with ledger.run(name="weather-query") as run:
    # Record what triggered this run
    run.record_input({"user_query": "What's the weather in Paris?"})
    
    # Record each step
    with run.step(name="fetch-weather", kind="tool") as step:
        weather = fetch_weather("Paris")
        step.record_input({"city": "Paris"})
        step.record_output(weather)
    
    with run.step(name="generate-response", kind="model") as step:
        response = generate_response(weather)
        step.record_output({"response": response})
    
    # Record final output
    run.record_output({"response": response})

# That's it! Now let's see what we recorded
print("=" * 50)
print("RECORDED RUN")
print("=" * 50)
print(f"Run ID:  {run.run_id}")
print(f"Name:    {run.name}")
print(f"Status:  {run.status.value}")
print(f"Steps:   {len(run.steps)}")
print()

print("Steps:")
for i, step in enumerate(run.steps, 1):
    print(f"  {i}. {step.name} ({step.kind.value})")
    print(f"     Input:  {step.inputs}")
    print(f"     Output: {step.outputs}")
print()

print(f"Final Output: {run.outputs}")
print()

# Retrieve the run later
retrieved = ledger.get_run(run.run_id)
print(f"Retrieved: {retrieved.name} with {len(retrieved.steps)} steps")

print()
print("✓ You just recorded your first agent run!")
print("  Next: python 02_testing_basics.py")
