"""Tests for run diff functionality."""

import pytest

from work_ledger.core.models import Run, Step, StepKind, Metrics, RunStatus
from work_ledger.testing.diff import RunDiff, StepDiff, format_diff


class TestRunDiff:
    """Tests for RunDiff class."""

    def test_identical_runs_no_changes(self):
        """Identical runs have no changes."""
        run1 = Run(run_id="r1", name="test", status=RunStatus.SUCCESS)
        run1.inputs = {"query": "test"}
        run1.outputs = {"result": "ok"}
        
        run2 = Run(run_id="r2", name="test", status=RunStatus.SUCCESS)
        run2.inputs = {"query": "test"}
        run2.outputs = {"result": "ok"}
        
        diff = RunDiff(run1, run2)
        assert not diff.has_changes
        assert diff.similarity == 1.0

    def test_different_outputs(self):
        """Different outputs are detected."""
        run1 = Run(name="test", status=RunStatus.SUCCESS)
        run1.outputs = {"result": "old"}
        
        run2 = Run(name="test", status=RunStatus.SUCCESS)
        run2.outputs = {"result": "new"}
        
        diff = RunDiff(run1, run2)
        assert diff.has_changes
        assert diff.output_changed
        assert "result" in diff.output_diff["changed"]

    def test_different_inputs(self):
        """Different inputs are detected."""
        run1 = Run(name="test")
        run1.inputs = {"query": "old"}
        
        run2 = Run(name="test")
        run2.inputs = {"query": "new"}
        
        diff = RunDiff(run1, run2)
        assert diff.has_changes
        assert diff.input_changed

    def test_different_status(self):
        """Different status is detected."""
        run1 = Run(name="test", status=RunStatus.SUCCESS)
        run2 = Run(name="test", status=RunStatus.FAILED)
        
        diff = RunDiff(run1, run2)
        assert diff.has_changes
        assert diff.status_changed

    def test_step_count_diff(self):
        """Different step counts are detected."""
        run1 = Run(name="test")
        run1.add_step(Step(name="s1", kind=StepKind.MODEL))
        
        run2 = Run(name="test")
        run2.add_step(Step(name="s1", kind=StepKind.MODEL))
        run2.add_step(Step(name="s2", kind=StepKind.TOOL))
        
        diff = RunDiff(run1, run2)
        assert diff.has_changes
        assert diff.steps_added == 1
        assert diff.steps_removed == 0

    def test_step_removed(self):
        """Removed steps are detected."""
        run1 = Run(name="test")
        run1.add_step(Step(name="s1", kind=StepKind.MODEL))
        run1.add_step(Step(name="s2", kind=StepKind.TOOL))
        
        run2 = Run(name="test")
        run2.add_step(Step(name="s1", kind=StepKind.MODEL))
        
        diff = RunDiff(run1, run2)
        assert diff.has_changes
        assert diff.steps_removed == 1
        assert diff.steps_added == 0

    def test_step_output_changed(self):
        """Changed step outputs are detected."""
        run1 = Run(name="test")
        step1 = Step(name="llm", kind=StepKind.MODEL)
        step1.outputs = {"response": "old answer"}
        run1.add_step(step1)
        
        run2 = Run(name="test")
        step2 = Step(name="llm", kind=StepKind.MODEL)
        step2.outputs = {"response": "new answer"}
        run2.add_step(step2)
        
        diff = RunDiff(run1, run2)
        assert diff.has_changes
        assert len(diff.step_diffs) == 1
        assert diff.step_diffs[0].output_changed

    def test_metrics_diff(self):
        """Metrics differences are detected."""
        run1 = Run(name="test")
        run1.metrics = Metrics(total_tokens=100, cost=0.001)
        
        run2 = Run(name="test")
        run2.metrics = Metrics(total_tokens=200, cost=0.002)
        
        diff = RunDiff(run1, run2)
        assert diff.metrics_changed
        assert diff.token_diff == 100
        assert diff.cost_diff == 0.001

    def test_ignore_timing(self):
        """Timing differences can be ignored."""
        from datetime import datetime, timezone, timedelta
        
        t1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        t2 = t1 + timedelta(hours=1)
        
        run1 = Run(name="test", started_at=t1)
        run2 = Run(name="test", started_at=t2)
        
        diff = RunDiff(run1, run2, ignore_timing=True)
        assert not diff.has_changes
        
        diff2 = RunDiff(run1, run2, ignore_timing=False)
        assert diff2.has_changes

    def test_ignore_ids(self):
        """ID differences can be ignored."""
        run1 = Run(run_id="id-1", name="test")
        run2 = Run(run_id="id-2", name="test")
        
        diff = RunDiff(run1, run2, ignore_ids=True)
        assert not diff.has_changes
        
        diff2 = RunDiff(run1, run2, ignore_ids=False)
        assert diff2.has_changes

    def test_similarity_score(self):
        """Similarity score is calculated correctly."""
        run1 = Run(name="test", status=RunStatus.SUCCESS)
        run1.outputs = {"a": 1, "b": 2, "c": 3}
        
        run2 = Run(name="test", status=RunStatus.SUCCESS)
        run2.outputs = {"a": 1, "b": 2, "c": 999}  # 1/3 changed
        
        diff = RunDiff(run1, run2)
        assert 0.5 < diff.similarity < 1.0


class TestStepDiff:
    """Tests for StepDiff class."""

    def test_identical_steps(self):
        """Identical steps have no diff."""
        s1 = Step(name="test", kind=StepKind.MODEL)
        s1.inputs = {"prompt": "hello"}
        s1.outputs = {"response": "hi"}
        
        s2 = Step(name="test", kind=StepKind.MODEL)
        s2.inputs = {"prompt": "hello"}
        s2.outputs = {"response": "hi"}
        
        diff = StepDiff(s1, s2)
        assert not diff.has_changes

    def test_output_changed(self):
        """Output changes are detected."""
        s1 = Step(name="test", kind=StepKind.MODEL)
        s1.outputs = {"response": "old"}
        
        s2 = Step(name="test", kind=StepKind.MODEL)
        s2.outputs = {"response": "new"}
        
        diff = StepDiff(s1, s2)
        assert diff.has_changes
        assert diff.output_changed

    def test_input_changed(self):
        """Input changes are detected."""
        s1 = Step(name="test", kind=StepKind.TOOL)
        s1.inputs = {"url": "old.com"}
        
        s2 = Step(name="test", kind=StepKind.TOOL)
        s2.inputs = {"url": "new.com"}
        
        diff = StepDiff(s1, s2)
        assert diff.has_changes
        assert diff.input_changed


class TestFormatDiff:
    """Tests for diff formatting."""

    def test_format_no_changes(self):
        """Format shows no changes."""
        run1 = Run(name="test", status=RunStatus.SUCCESS)
        run2 = Run(name="test", status=RunStatus.SUCCESS)
        
        diff = RunDiff(run1, run2)
        output = format_diff(diff)
        
        assert "no changes" in output.lower() or "identical" in output.lower()

    def test_format_with_changes(self):
        """Format shows changes."""
        run1 = Run(name="test", status=RunStatus.SUCCESS)
        run1.outputs = {"result": "old"}
        
        run2 = Run(name="test", status=RunStatus.SUCCESS)
        run2.outputs = {"result": "new"}
        
        diff = RunDiff(run1, run2)
        output = format_diff(diff)
        
        assert "output" in output.lower()
        assert "old" in output or "new" in output

    def test_format_step_changes(self):
        """Format shows step changes."""
        run1 = Run(name="test")
        run1.add_step(Step(name="s1", kind=StepKind.MODEL))
        
        run2 = Run(name="test")
        run2.add_step(Step(name="s1", kind=StepKind.MODEL))
        run2.add_step(Step(name="s2", kind=StepKind.TOOL))
        
        diff = RunDiff(run1, run2)
        output = format_diff(diff)
        
        assert "step" in output.lower()
