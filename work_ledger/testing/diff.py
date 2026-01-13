"""Run and step diff functionality for testing.

This module provides diff computation and formatting for comparing
runs and steps, useful for regression testing and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from work_ledger.core.models import Run, Step


@dataclass
class StepDiff:
    """Diff between two steps.
    
    Attributes:
        step_name: Name of the step being compared
        expected: The expected step (baseline)
        actual: The actual step
        input_changed: Whether inputs differ
        output_changed: Whether outputs differ
        metrics_changed: Whether metrics differ
    """
    step_name: str
    expected: Step | None
    actual: Step | None
    input_changed: bool = False
    output_changed: bool = False
    metrics_changed: bool = False
    input_diff: dict[str, Any] = field(default_factory=dict)
    output_diff: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        expected: Step | None,
        actual: Step | None,
        ignore_timing: bool = True,
    ) -> None:
        """Initialize step diff.
        
        Args:
            expected: The expected step
            actual: The actual step
            ignore_timing: Whether to ignore timing differences
        """
        self.expected = expected
        self.actual = actual
        self.step_name = (expected.name if expected else 
                          actual.name if actual else "unknown")
        
        self.input_changed = False
        self.output_changed = False
        self.metrics_changed = False
        self.input_diff = {}
        self.output_diff = {}
        
        if expected and actual:
            self._compute_diff(expected, actual)

    def _compute_diff(self, expected: Step, actual: Step) -> None:
        """Compute differences between steps."""
        # Compare inputs
        if expected.inputs != actual.inputs:
            self.input_changed = True
            self.input_diff = _dict_diff(expected.inputs, actual.inputs)
        
        # Compare outputs
        if expected.outputs != actual.outputs:
            self.output_changed = True
            self.output_diff = _dict_diff(expected.outputs, actual.outputs)
        
        # Compare metrics
        if (expected.metrics.total_tokens != actual.metrics.total_tokens or
            expected.metrics.cost != actual.metrics.cost):
            self.metrics_changed = True

    @property
    def has_changes(self) -> bool:
        """Whether there are any differences."""
        if self.expected is None or self.actual is None:
            return True
        return self.input_changed or self.output_changed or self.metrics_changed


@dataclass
class RunDiff:
    """Diff between two runs.
    
    Computes and stores differences between a baseline (expected)
    run and an actual run for comparison and regression testing.
    
    Example:
        >>> diff = RunDiff(baseline_run, actual_run)
        >>> if diff.has_changes:
        ...     print(format_diff(diff))
    """
    expected: Run
    actual: Run
    ignore_timing: bool = True
    ignore_ids: bool = True
    
    # Computed diffs
    input_changed: bool = False
    output_changed: bool = False
    status_changed: bool = False
    metrics_changed: bool = False
    
    input_diff: dict[str, Any] = field(default_factory=dict)
    output_diff: dict[str, Any] = field(default_factory=dict)
    
    steps_added: int = 0
    steps_removed: int = 0
    step_diffs: list[StepDiff] = field(default_factory=list)
    
    token_diff: int = 0
    cost_diff: float = 0.0

    def __init__(
        self,
        expected: Run,
        actual: Run,
        ignore_timing: bool = True,
        ignore_ids: bool = True,
    ) -> None:
        """Initialize run diff.
        
        Args:
            expected: The expected (baseline) run
            actual: The actual run
            ignore_timing: Whether to ignore timing differences
            ignore_ids: Whether to ignore ID differences
        """
        self.expected = expected
        self.actual = actual
        self.ignore_timing = ignore_timing
        self.ignore_ids = ignore_ids
        
        self.input_changed = False
        self.output_changed = False
        self.status_changed = False
        self.metrics_changed = False
        
        self.input_diff = {}
        self.output_diff = {}
        
        self.steps_added = 0
        self.steps_removed = 0
        self.step_diffs = []
        
        self.token_diff = 0
        self.cost_diff = 0.0
        
        self._compute_diff()

    def _compute_diff(self) -> None:
        """Compute all differences between runs."""
        # Compare inputs
        if self.expected.inputs != self.actual.inputs:
            self.input_changed = True
            self.input_diff = _dict_diff(self.expected.inputs, self.actual.inputs)
        
        # Compare outputs
        if self.expected.outputs != self.actual.outputs:
            self.output_changed = True
            self.output_diff = _dict_diff(self.expected.outputs, self.actual.outputs)
        
        # Compare status
        if self.expected.status != self.actual.status:
            self.status_changed = True
        
        # Compare IDs (if not ignored)
        if not self.ignore_ids:
            if self.expected.run_id != self.actual.run_id:
                self.input_changed = True  # Use input_changed as proxy
        
        # Compare timing (if not ignored)
        if not self.ignore_timing:
            if self.expected.started_at != self.actual.started_at:
                self.input_changed = True
        
        # Compare metrics
        self.token_diff = (
            self.actual.metrics.total_tokens - self.expected.metrics.total_tokens
        )
        
        expected_cost = self.expected.metrics.cost or 0.0
        actual_cost = self.actual.metrics.cost or 0.0
        self.cost_diff = actual_cost - expected_cost
        
        if self.token_diff != 0 or self.cost_diff != 0:
            self.metrics_changed = True
        
        # Compare steps
        self._compare_steps()

    def _compare_steps(self) -> None:
        """Compare steps between runs."""
        expected_steps = {s.name: s for s in self.expected.steps}
        actual_steps = {s.name: s for s in self.actual.steps}
        
        expected_names = set(expected_steps.keys())
        actual_names = set(actual_steps.keys())
        
        # Count added/removed
        self.steps_added = len(actual_names - expected_names)
        self.steps_removed = len(expected_names - actual_names)
        
        # Compute diffs for common steps
        common = expected_names & actual_names
        for name in common:
            step_diff = StepDiff(
                expected_steps[name],
                actual_steps[name],
                ignore_timing=self.ignore_timing,
            )
            if step_diff.has_changes:
                self.step_diffs.append(step_diff)

    @property
    def has_changes(self) -> bool:
        """Whether there are any differences."""
        return (
            self.input_changed or
            self.output_changed or
            self.status_changed or
            self.metrics_changed or
            self.steps_added > 0 or
            self.steps_removed > 0 or
            len(self.step_diffs) > 0
        )

    @property
    def similarity(self) -> float:
        """Compute similarity score between 0 and 1.
        
        1.0 means identical, 0.0 means completely different.
        """
        if not self.has_changes:
            return 1.0
        
        # Simple scoring: count changed fields
        total_fields = 4  # inputs, outputs, status, metrics
        changed_fields = sum([
            self.input_changed,
            self.output_changed,
            self.status_changed,
            self.metrics_changed,
        ])
        
        # Add step differences
        total_steps = max(
            len(self.expected.steps),
            len(self.actual.steps),
            1  # Avoid division by zero
        )
        step_similarity = 1.0 - (
            (self.steps_added + self.steps_removed + len(self.step_diffs)) /
            (total_steps * 2)  # Normalize
        )
        step_similarity = max(0.0, step_similarity)
        
        # Combine
        field_similarity = 1.0 - (changed_fields / total_fields)
        
        return (field_similarity + step_similarity) / 2

    @property
    def difference(self) -> float:
        """Compute difference score (1 - similarity)."""
        return 1.0 - self.similarity


def _dict_diff(expected: dict, actual: dict) -> dict[str, Any]:
    """Compute diff between two dictionaries.
    
    Returns a dict with keys:
    - added: keys in actual but not expected
    - removed: keys in expected but not actual
    - changed: keys with different values
    """
    expected_keys = set(expected.keys())
    actual_keys = set(actual.keys())
    
    diff = {
        "added": {k: actual[k] for k in actual_keys - expected_keys},
        "removed": {k: expected[k] for k in expected_keys - actual_keys},
        "changed": {},
    }
    
    for key in expected_keys & actual_keys:
        if expected[key] != actual[key]:
            diff["changed"][key] = {
                "expected": expected[key],
                "actual": actual[key],
            }
    
    return diff


def format_diff(diff: RunDiff) -> str:
    """Format a run diff for display.
    
    Args:
        diff: The diff to format
        
    Returns:
        Human-readable string representation
    """
    if not diff.has_changes:
        return "Runs are identical (no changes detected)"
    
    lines = ["Run Diff:"]
    lines.append(f"  Similarity: {diff.similarity:.1%}")
    lines.append("")
    
    if diff.input_changed:
        lines.append("  Inputs changed:")
        _format_dict_diff(diff.input_diff, lines, indent=4)
    
    if diff.output_changed:
        lines.append("  Outputs changed:")
        _format_dict_diff(diff.output_diff, lines, indent=4)
    
    if diff.status_changed:
        lines.append(f"  Status: {diff.expected.status.value} → {diff.actual.status.value}")
    
    if diff.metrics_changed:
        lines.append(f"  Metrics:")
        if diff.token_diff != 0:
            sign = "+" if diff.token_diff > 0 else ""
            lines.append(f"    Tokens: {sign}{diff.token_diff}")
        if diff.cost_diff != 0:
            sign = "+" if diff.cost_diff > 0 else ""
            lines.append(f"    Cost: {sign}${diff.cost_diff:.4f}")
    
    if diff.steps_added > 0 or diff.steps_removed > 0:
        lines.append(f"  Steps:")
        if diff.steps_added > 0:
            lines.append(f"    Added: {diff.steps_added}")
        if diff.steps_removed > 0:
            lines.append(f"    Removed: {diff.steps_removed}")
    
    if diff.step_diffs:
        lines.append("  Step changes:")
        for step_diff in diff.step_diffs:
            lines.append(f"    {step_diff.step_name}:")
            if step_diff.input_changed:
                lines.append("      Inputs changed")
            if step_diff.output_changed:
                lines.append("      Outputs changed")
    
    return "\n".join(lines)


def _format_dict_diff(diff: dict, lines: list[str], indent: int = 0) -> None:
    """Format dict diff into lines."""
    prefix = " " * indent
    
    if diff.get("added"):
        for key, value in diff["added"].items():
            lines.append(f"{prefix}+ {key}: {value}")
    
    if diff.get("removed"):
        for key, value in diff["removed"].items():
            lines.append(f"{prefix}- {key}: {value}")
    
    if diff.get("changed"):
        for key, change in diff["changed"].items():
            lines.append(f"{prefix}~ {key}: {change['expected']} → {change['actual']}")
