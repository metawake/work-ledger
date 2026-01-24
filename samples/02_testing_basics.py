#!/usr/bin/env python3
"""
Sample 02: Testing Basics
=========================

Learn how to test agent workflows without making API calls.

This sample shows:
- Recording a "golden" run
- Replaying without external calls
- Comparing runs with assertions

Run:
    python 02_testing_basics.py
"""

import tempfile
from pathlib import Path

from work_ledger import WorkLedger, Run, RunStatus
from work_ledger.testing import (
    Recording,
    Fixture,
    save_recording,
    load_recording,
    assert_run_matches,
    assert_output_matches,
)


# --- Step 1: Record a Golden Run ---

print("=" * 50)
print("STEP 1: Record a Golden Run")
print("=" * 50)

ledger = WorkLedger(store=":memory:")

# Simulate your agent
def my_agent(query: str) -> dict:
    """Your actual agent logic."""
    with ledger.run(name="my-agent") as run:
        run.record_input({"query": query})
        
        # Step 1: Retrieve context
        with run.step(name="retrieve", kind="retrieval") as step:
            docs = ["Paris is the capital of France", "Population: 2.1M"]
            step.record_output({"docs": docs})
        
        # Step 2: Generate response
        with run.step(name="generate", kind="model") as step:
            response = f"Based on my knowledge: {docs[0]}"
            step.record_output({"response": response})
        
        run.record_output({"answer": response})
        return {"answer": response, "run": run}


# Run the agent and capture the result
result = my_agent("Tell me about Paris")
run_context = result["run"]

# Get the actual Run object from the ledger
golden_run = ledger.get_run(run_context.run_id)

print(f"Recorded run: {golden_run.run_id}")
print(f"Output: {golden_run.outputs}")
print()


# --- Step 2: Save as a Fixture ---

print("=" * 50)
print("STEP 2: Save as a Fixture")
print("=" * 50)

# Create a temporary directory for fixtures
fixture_dir = Path(tempfile.mkdtemp())
fixture_path = fixture_dir / "golden.json"

# Save the recording
recording = Recording(
    run=golden_run,
    fixtures=[],  # Would contain LLM responses in real usage
    metadata={"version": "1.0", "description": "Golden run for Paris query"}
)
save_recording(fixture_path, recording)

print(f"Saved fixture to: {fixture_path}")
print()


# --- Step 3: Load and Compare ---

print("=" * 50)
print("STEP 3: Load and Compare")
print("=" * 50)

# Load the fixture
loaded = load_recording(fixture_path)
print(f"Loaded run: {loaded.run.name}")
print(f"Steps: {[s.name for s in loaded.run.steps]}")
print()


# --- Step 4: Use Assertions ---

print("=" * 50)
print("STEP 4: Use Assertions")
print("=" * 50)

# Run the agent again
result2 = my_agent("Tell me about Paris")
new_run = ledger.get_run(result2["run"].run_id)

# Assert outputs match
try:
    assert_output_matches(
        new_run.outputs,
        expected_keys=["answer"]
    )
    print("✓ Output has required keys")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")

# Assert runs are similar
try:
    assert_run_matches(
        new_run,
        golden_run,
        ignore_timing=True,
        ignore_ids=True,
    )
    print("✓ Runs match (ignoring timing and IDs)")
except AssertionError as e:
    print(f"✗ Runs differ: {e}")

print()
print("✓ You learned the testing basics!")
print("  Next: python 03_detect_regression.py")
