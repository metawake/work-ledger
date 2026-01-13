"""Tests for testing assertions."""

import pytest

from work_ledger.core.models import Run, Step, StepKind, Metrics, RunStatus
from work_ledger.testing.assertions import (
    assert_run_matches,
    assert_steps_match,
    assert_output_matches,
    assert_no_regression,
)


class TestAssertRunMatches:
    """Tests for assert_run_matches."""

    def test_matching_runs_pass(self):
        """Matching runs don't raise."""
        run1 = Run(name="test", status=RunStatus.SUCCESS)
        run1.inputs = {"query": "test"}
        run1.outputs = {"result": "ok"}
        
        run2 = Run(name="test", status=RunStatus.SUCCESS)
        run2.inputs = {"query": "test"}
        run2.outputs = {"result": "ok"}
        
        # Should not raise
        assert_run_matches(run1, run2)

    def test_different_outputs_fail(self):
        """Different outputs raise AssertionError."""
        run1 = Run(name="test", status=RunStatus.SUCCESS)
        run1.outputs = {"result": "old"}
        
        run2 = Run(name="test", status=RunStatus.SUCCESS)
        run2.outputs = {"result": "new"}
        
        with pytest.raises(AssertionError) as exc_info:
            assert_run_matches(run1, run2)
        
        assert "match" in str(exc_info.value).lower()

    def test_ignore_timing(self):
        """Timing differences can be ignored."""
        from datetime import datetime, timezone
        
        run1 = Run(name="test", started_at=datetime(2024, 1, 1, tzinfo=timezone.utc))
        run2 = Run(name="test", started_at=datetime(2024, 1, 2, tzinfo=timezone.utc))
        
        # Should not raise with ignore_timing=True
        assert_run_matches(run1, run2, ignore_timing=True)

    def test_ignore_ids(self):
        """ID differences can be ignored."""
        run1 = Run(run_id="id-1", name="test")
        run2 = Run(run_id="id-2", name="test")
        
        # Should not raise with ignore_ids=True
        assert_run_matches(run1, run2, ignore_ids=True)


class TestAssertStepsMatch:
    """Tests for assert_steps_match."""

    def test_matching_steps_pass(self):
        """Matching steps don't raise."""
        steps1 = [
            Step(name="s1", kind=StepKind.MODEL),
            Step(name="s2", kind=StepKind.TOOL),
        ]
        steps2 = [
            Step(name="s1", kind=StepKind.MODEL),
            Step(name="s2", kind=StepKind.TOOL),
        ]
        
        assert_steps_match(steps1, steps2)

    def test_different_count_fails(self):
        """Different step counts raise."""
        steps1 = [Step(name="s1", kind=StepKind.MODEL)]
        steps2 = [
            Step(name="s1", kind=StepKind.MODEL),
            Step(name="s2", kind=StepKind.TOOL),
        ]
        
        with pytest.raises(AssertionError) as exc_info:
            assert_steps_match(steps1, steps2)
        
        assert "count" in str(exc_info.value).lower()

    def test_different_order_fails_when_checked(self):
        """Different order fails when check_order=True."""
        steps1 = [
            Step(name="s1", kind=StepKind.MODEL),
            Step(name="s2", kind=StepKind.TOOL),
        ]
        steps2 = [
            Step(name="s2", kind=StepKind.TOOL),
            Step(name="s1", kind=StepKind.MODEL),
        ]
        
        with pytest.raises(AssertionError):
            assert_steps_match(steps1, steps2, check_order=True)

    def test_different_order_passes_when_not_checked(self):
        """Different order passes when check_order=False."""
        steps1 = [
            Step(name="s1", kind=StepKind.MODEL),
            Step(name="s2", kind=StepKind.TOOL),
        ]
        steps2 = [
            Step(name="s2", kind=StepKind.TOOL),
            Step(name="s1", kind=StepKind.MODEL),
        ]
        
        assert_steps_match(steps1, steps2, check_order=False)


class TestAssertOutputMatches:
    """Tests for assert_output_matches."""

    def test_exact_match_passes(self):
        """Exact match passes."""
        actual = {"result": "ok", "count": 5}
        expected = {"result": "ok", "count": 5}
        
        assert_output_matches(actual, expected)

    def test_different_value_fails(self):
        """Different values fail."""
        actual = {"result": "ok"}
        expected = {"result": "different"}
        
        with pytest.raises(AssertionError):
            assert_output_matches(actual, expected)

    def test_expected_keys_present(self):
        """Required keys must be present."""
        actual = {"result": "ok", "count": 5}
        
        assert_output_matches(actual, expected_keys=["result", "count"])

    def test_expected_keys_missing_fails(self):
        """Missing required keys fail."""
        actual = {"result": "ok"}
        
        with pytest.raises(AssertionError) as exc_info:
            assert_output_matches(actual, expected_keys=["result", "missing_key"])
        
        assert "missing" in str(exc_info.value).lower()

    def test_forbidden_keys_absent(self):
        """Forbidden keys must be absent."""
        actual = {"result": "ok"}
        
        assert_output_matches(actual, forbidden_keys=["error", "debug"])

    def test_forbidden_keys_present_fails(self):
        """Present forbidden keys fail."""
        actual = {"result": "ok", "error": "something"}
        
        with pytest.raises(AssertionError) as exc_info:
            assert_output_matches(actual, forbidden_keys=["error"])
        
        assert "forbidden" in str(exc_info.value).lower()


class TestAssertNoRegression:
    """Tests for assert_no_regression."""

    def test_identical_runs_pass(self):
        """Identical runs pass."""
        baseline = Run(name="test", status=RunStatus.SUCCESS)
        baseline.add_step(Step(name="s1", kind=StepKind.MODEL))
        baseline.metrics = Metrics(total_tokens=100)
        
        actual = Run(name="test", status=RunStatus.SUCCESS)
        actual.add_step(Step(name="s1", kind=StepKind.MODEL))
        actual.metrics = Metrics(total_tokens=100)
        
        assert_no_regression(actual, baseline)

    def test_removed_step_fails(self):
        """Removed steps fail regression check."""
        baseline = Run(name="test")
        baseline.add_step(Step(name="s1", kind=StepKind.MODEL))
        baseline.add_step(Step(name="s2", kind=StepKind.TOOL))
        
        actual = Run(name="test")
        actual.add_step(Step(name="s1", kind=StepKind.MODEL))
        # s2 is missing
        
        with pytest.raises(AssertionError) as exc_info:
            assert_no_regression(actual, baseline)
        
        assert "removed" in str(exc_info.value).lower()

    def test_new_step_allowed(self):
        """New steps pass when allowed."""
        baseline = Run(name="test")
        baseline.add_step(Step(name="s1", kind=StepKind.MODEL))
        
        actual = Run(name="test")
        actual.add_step(Step(name="s1", kind=StepKind.MODEL))
        actual.add_step(Step(name="s2", kind=StepKind.TOOL))  # new step
        
        assert_no_regression(actual, baseline, allow_new_steps=True)

    def test_new_step_fails_when_not_allowed(self):
        """New steps fail when not allowed."""
        baseline = Run(name="test")
        baseline.add_step(Step(name="s1", kind=StepKind.MODEL))
        
        actual = Run(name="test")
        actual.add_step(Step(name="s1", kind=StepKind.MODEL))
        actual.add_step(Step(name="s2", kind=StepKind.TOOL))
        
        with pytest.raises(AssertionError):
            assert_no_regression(actual, baseline, allow_new_steps=False)

    def test_token_increase_within_threshold(self):
        """Token increase within threshold passes."""
        baseline = Run(name="test")
        baseline.metrics = Metrics(total_tokens=100)
        
        actual = Run(name="test")
        actual.metrics = Metrics(total_tokens=105)  # 5% increase
        
        assert_no_regression(actual, baseline, allow_metric_increase=0.1)

    def test_token_increase_exceeds_threshold(self):
        """Token increase exceeding threshold fails."""
        baseline = Run(name="test")
        baseline.metrics = Metrics(total_tokens=100)
        
        actual = Run(name="test")
        actual.metrics = Metrics(total_tokens=150)  # 50% increase
        
        with pytest.raises(AssertionError) as exc_info:
            assert_no_regression(actual, baseline, allow_metric_increase=0.1)
        
        assert "token" in str(exc_info.value).lower()

    def test_cost_increase_exceeds_threshold(self):
        """Cost increase exceeding threshold fails."""
        baseline = Run(name="test")
        baseline.metrics = Metrics(cost=0.01)
        
        actual = Run(name="test")
        actual.metrics = Metrics(cost=0.02)  # 100% increase
        
        with pytest.raises(AssertionError) as exc_info:
            assert_no_regression(actual, baseline, allow_metric_increase=0.1)
        
        assert "cost" in str(exc_info.value).lower()
