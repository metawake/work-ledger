#!/usr/bin/env python3
"""
Sample 03: Detect Regression
============================

Learn how to catch when agent behavior changes.

This sample shows:
- Recording a baseline
- Detecting output drift
- Detecting step changes
- Understanding what broke

Run:
    python 03_detect_regression.py
"""

from work_ledger import WorkLedger
from work_ledger.testing import (
    RunDiff,
    format_diff,
    assert_no_regression,
)


# --- Scenario: Your Agent Changes Behavior ---

print("=" * 50)
print("SCENARIO: Detect When Your Agent Changes")
print("=" * 50)
print()
print("Imagine you recorded a working agent last week.")
print("Today, after updating the model, it behaves differently.")
print("Work Ledger helps you see exactly what changed.")
print()


# --- Create "Last Week's" Baseline ---

ledger = WorkLedger(store=":memory:")

with ledger.run(name="summarize-doc") as baseline:
    baseline.record_input({"doc": "Long document about AI..."})
    
    with baseline.step(name="chunk-doc", kind="tool") as s1:
        s1.record_output({"chunks": ["chunk1", "chunk2", "chunk3"]})
    
    with baseline.step(name="summarize-chunks", kind="model") as s2:
        s2.record_output({"summary": "AI is transforming industries."})
        s2.record_metrics(prompt_tokens=500, completion_tokens=50, total_tokens=550)
    
    baseline.record_output({"summary": "AI is transforming industries."})

baseline_run = ledger.get_run(baseline.run_id)


# --- Create "Today's" Run (with changes) ---

with ledger.run(name="summarize-doc") as current:
    current.record_input({"doc": "Long document about AI..."})
    
    # Same chunking step
    with current.step(name="chunk-doc", kind="tool") as s1:
        s1.record_output({"chunks": ["chunk1", "chunk2", "chunk3"]})
    
    # NEW: Added a retrieval step (model now uses RAG)
    with current.step(name="retrieve-context", kind="retrieval") as s2:
        s2.record_output({"context": ["related doc 1", "related doc 2"]})
    
    # Summary is different!
    with current.step(name="summarize-chunks", kind="model") as s3:
        s3.record_output({"summary": "AI and ML are reshaping the future."})
        s3.record_metrics(prompt_tokens=800, completion_tokens=60, total_tokens=860)  # More tokens!
    
    current.record_output({"summary": "AI and ML are reshaping the future."})

current_run = ledger.get_run(current.run_id)


# --- Compare the Runs ---

print("=" * 50)
print("DIFF: What Changed?")
print("=" * 50)
print()

diff = RunDiff(baseline_run, current_run, ignore_timing=True, ignore_ids=True)

print(format_diff(diff))
print()


# --- Detailed Analysis ---

print("=" * 50)
print("DETAILED ANALYSIS")
print("=" * 50)
print()

print(f"Output changed: {diff.output_changed}")
print(f"Steps added:    {diff.steps_added}")
print(f"Steps removed:  {diff.steps_removed}")
print(f"Token diff:     +{diff.token_diff} tokens")
print(f"Similarity:     {diff.similarity:.1%}")
print()


# --- Regression Check ---

print("=" * 50)
print("REGRESSION CHECK")
print("=" * 50)
print()

try:
    assert_no_regression(
        current_run,
        baseline_run,
        allow_new_steps=False,  # Fail if new steps added
        allow_metric_increase=0.1,  # Allow 10% token increase
    )
    print("✓ No regression detected")
except AssertionError as e:
    print(f"✗ Regression detected:")
    print(f"  {e}")

print()

# Try with relaxed settings
try:
    assert_no_regression(
        current_run,
        baseline_run,
        allow_new_steps=True,  # Allow new steps
        allow_metric_increase=1.0,  # Allow 100% token increase
    )
    print("✓ Passes with relaxed settings (new steps allowed)")
except AssertionError as e:
    print(f"✗ Still failing: {e}")

print()
print("=" * 50)
print("WHAT YOU LEARNED")
print("=" * 50)
print("""
1. RunDiff shows exactly what changed between runs
2. assert_no_regression catches behavioral changes
3. You can tune thresholds for token/cost drift
4. New steps, removed steps, and output changes are tracked

This is how you catch regressions after:
- Changing the model
- Updating the prompt
- Modifying the retrieval strategy
- Refactoring agent code
""")
print("  Next: python 04_debug_workflow.py")
