"""Test assertions for agent runs.

This module provides pytest-style assertions for testing
agent behavior, outputs, and regressions.
"""

from __future__ import annotations

from typing import Any

from work_ledger.core.models import Run, Step, StepKind
from work_ledger.testing.diff import RunDiff, format_diff


def assert_run_matches(
    actual: Run,
    expected: Run,
    ignore_timing: bool = True,
    ignore_ids: bool = True,
) -> None:
    """Assert two runs match.
    
    Compares runs and raises AssertionError with detailed diff
    if they don't match.
    
    Args:
        actual: The actual run
        expected: The expected run
        ignore_timing: Ignore started_at/ended_at differences
        ignore_ids: Ignore run_id/step_id differences
        
    Raises:
        AssertionError: If runs don't match
        
    Example:
        >>> assert_run_matches(actual_run, expected_run)
    """
    diff = RunDiff(
        expected,
        actual,
        ignore_timing=ignore_timing,
        ignore_ids=ignore_ids,
    )
    
    if diff.has_changes:
        raise AssertionError(f"Runs don't match:\n{format_diff(diff)}")


def assert_steps_match(
    actual: list[Step],
    expected: list[Step],
    check_order: bool = True,
) -> None:
    """Assert step sequences match.
    
    Args:
        actual: Actual steps
        expected: Expected steps
        check_order: If True, order must match
        
    Raises:
        AssertionError: If steps don't match
        
    Example:
        >>> assert_steps_match(run.steps, expected_steps)
    """
    if len(actual) != len(expected):
        raise AssertionError(
            f"Step count mismatch: got {len(actual)}, expected {len(expected)}"
        )
    
    if check_order:
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a.name != e.name or a.kind != e.kind:
                raise AssertionError(
                    f"Step {i} mismatch:\n"
                    f"  Actual: {a.name} ({a.kind.value})\n"
                    f"  Expected: {e.name} ({e.kind.value})"
                )
    else:
        actual_set = {(s.name, s.kind) for s in actual}
        expected_set = {(s.name, s.kind) for s in expected}
        
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        
        if missing or extra:
            msg = "Step mismatch:"
            if missing:
                msg += f"\n  Missing: {missing}"
            if extra:
                msg += f"\n  Extra: {extra}"
            raise AssertionError(msg)


def assert_output_matches(
    actual: dict[str, Any],
    expected: dict[str, Any] | None = None,
    expected_keys: list[str] | None = None,
    forbidden_keys: list[str] | None = None,
) -> None:
    """Assert output matches expectations.
    
    Flexible assertion that can check:
    - Exact output match
    - Presence of required keys
    - Absence of forbidden keys
    
    Args:
        actual: Actual output dict
        expected: Expected exact output (optional)
        expected_keys: Keys that must be present (optional)
        forbidden_keys: Keys that must not be present (optional)
        
    Raises:
        AssertionError: If assertions fail
        
    Example:
        >>> assert_output_matches(run.outputs, expected_keys=["response"])
        >>> assert_output_matches(run.outputs, forbidden_keys=["error"])
    """
    if expected is not None:
        if actual != expected:
            raise AssertionError(
                f"Output mismatch:\n"
                f"  Actual: {actual}\n"
                f"  Expected: {expected}"
            )
    
    if expected_keys:
        missing = set(expected_keys) - set(actual.keys())
        if missing:
            raise AssertionError(f"Missing output keys: {missing}")
    
    if forbidden_keys:
        present = set(forbidden_keys) & set(actual.keys())
        if present:
            raise AssertionError(f"Forbidden keys present: {present}")


def assert_no_regression(
    actual: Run,
    baseline: Run,
    allow_new_steps: bool = False,
    allow_metric_increase: float = 0.1,
) -> None:
    """Assert run hasn't regressed from baseline.
    
    Checks for regressions:
    - No steps removed (steps must still exist)
    - Optionally, no new steps added
    - Metrics haven't increased beyond threshold
    
    Args:
        actual: Current run
        baseline: Baseline run to compare against
        allow_new_steps: If True, new steps are OK
        allow_metric_increase: Allowed fractional increase (0.1 = 10%)
        
    Raises:
        AssertionError: If regression detected
        
    Example:
        >>> assert_no_regression(current_run, baseline_run)
    """
    # Check no steps removed
    baseline_steps = {s.name for s in baseline.steps}
    actual_steps = {s.name for s in actual.steps}
    
    removed = baseline_steps - actual_steps
    if removed:
        raise AssertionError(f"Regression: steps removed: {removed}")
    
    # Check for new steps
    if not allow_new_steps:
        added = actual_steps - baseline_steps
        if added:
            raise AssertionError(f"Regression: unexpected new steps: {added}")
    
    # Check token increase
    if baseline.metrics.total_tokens > 0:
        token_increase = (
            (actual.metrics.total_tokens - baseline.metrics.total_tokens)
            / baseline.metrics.total_tokens
        )
        if token_increase > allow_metric_increase:
            raise AssertionError(
                f"Regression: token usage increased by {token_increase:.1%} "
                f"(threshold: {allow_metric_increase:.1%})"
            )
    
    # Check cost increase
    baseline_cost = baseline.metrics.cost
    actual_cost = actual.metrics.cost
    
    if baseline_cost and baseline_cost > 0:
        cost_increase = (
            ((actual_cost or 0) - baseline_cost) / baseline_cost
        )
        if cost_increase > allow_metric_increase:
            raise AssertionError(
                f"Regression: cost increased by {cost_increase:.1%} "
                f"(threshold: {allow_metric_increase:.1%})"
            )
