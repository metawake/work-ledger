"""Tests for WorkLedger context manager API."""

from datetime import datetime, timezone
import tempfile
from pathlib import Path

import pytest

from work_ledger.core.ledger import WorkLedger
from work_ledger.core.models import Run, RunStatus, Step, StepKind, Metrics


class TestWorkLedgerBasic:
    """Basic WorkLedger functionality tests."""

    def test_create_ledger_with_memory_store(self):
        """WorkLedger can be created with in-memory store."""
        ledger = WorkLedger(store=":memory:")
        assert ledger is not None

    def test_create_ledger_with_path(self):
        """WorkLedger can be created with a file path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ledger = WorkLedger(store=tmpdir)
            assert ledger is not None


class TestRunContextManager:
    """Tests for run context manager API."""

    def test_run_context_manager_basic(self):
        """Run can be used as context manager."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test-run") as run:
            assert run.name == "test-run"
            assert run.status == RunStatus.RUNNING
            assert run.started_at is not None
        
        # After context exits, run should be completed
        assert run.status == RunStatus.SUCCESS
        assert run.ended_at is not None

    def test_run_records_inputs_outputs(self):
        """Run records inputs and outputs."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test") as run:
            run.record_input({"query": "test query"})
            run.record_output({"response": "test response"})
        
        assert run.inputs == {"query": "test query"}
        assert run.outputs == {"response": "test response"}

    def test_run_fails_on_exception(self):
        """Run status is FAILED when exception occurs."""
        ledger = WorkLedger(store=":memory:")
        
        with pytest.raises(ValueError):
            with ledger.run(name="test") as run:
                raise ValueError("Something went wrong")
        
        assert run.status == RunStatus.FAILED

    def test_run_captures_error_info(self):
        """Run captures error information on failure."""
        ledger = WorkLedger(store=":memory:")
        
        with pytest.raises(ValueError):
            with ledger.run(name="test") as run:
                raise ValueError("Specific error message")
        
        assert "error" in run.annotations
        assert "ValueError" in run.annotations["error"]["type"]
        assert "Specific error message" in run.annotations["error"]["message"]


class TestStepContextManager:
    """Tests for step context manager API."""

    def test_step_context_manager_basic(self):
        """Step can be used as context manager within a run."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test") as run:
            with run.step(name="my-step", kind="tool") as step:
                assert step.name == "my-step"
                assert step.kind == StepKind.TOOL
                assert step.started_at is not None
            
            # Step should be ended and added to run
            assert step.ended_at is not None
            assert len(run.steps) == 1
            assert run.steps[0].step_id == step.step_id
            assert run.steps[0].name == step.name

    def test_step_records_inputs_outputs(self):
        """Step records inputs and outputs."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test") as run:
            with run.step(name="fetch", kind="retrieval") as step:
                step.record_input({"query": "search term"})
                step.record_output({"docs": ["doc1", "doc2"]})
        
        assert step.inputs == {"query": "search term"}
        assert step.outputs == {"docs": ["doc1", "doc2"]}

    def test_step_records_metrics(self):
        """Step can record metrics."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test") as run:
            with run.step(name="llm-call", kind="model") as step:
                step.record_metrics(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150,
                    cost=0.001,
                )
        
        assert step.metrics.prompt_tokens == 100
        assert step.metrics.completion_tokens == 50
        assert step.metrics.cost == 0.001

    def test_step_accepts_kind_as_string(self):
        """Step accepts kind as string."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test") as run:
            with run.step(name="s1", kind="model") as step:
                pass
        
        assert step.kind == StepKind.MODEL

    def test_step_accepts_kind_as_enum(self):
        """Step accepts kind as StepKind enum."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test") as run:
            with run.step(name="s1", kind=StepKind.RETRIEVAL) as step:
                pass
        
        assert step.kind == StepKind.RETRIEVAL

    def test_multiple_steps(self):
        """Multiple steps can be added to a run."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test") as run:
            with run.step(name="step1", kind="tool") as s1:
                s1.record_output({"result": 1})
            
            with run.step(name="step2", kind="model") as s2:
                s2.record_output({"result": 2})
            
            with run.step(name="step3", kind="retrieval") as s3:
                s3.record_output({"result": 3})
        
        assert len(run.steps) == 3
        assert run.steps[0].name == "step1"
        assert run.steps[1].name == "step2"
        assert run.steps[2].name == "step3"


class TestCausalLinks:
    """Tests for causal link recording."""

    def test_step_with_caused_by(self):
        """Steps can specify causal relationships."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test") as run:
            with run.step(name="trigger", kind="custom") as s1:
                pass
            
            with run.step(name="response", kind="model", caused_by=s1.step_id) as s2:
                pass
        
        assert s2.caused_by == s1.step_id

    def test_run_with_parent(self):
        """Runs can specify parent run relationship."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="parent") as parent_run:
            pass
        
        with ledger.run(name="child", parent_run_id=parent_run.run_id) as child_run:
            pass
        
        assert child_run.links.parent_run_id == parent_run.run_id

    def test_run_with_correlation_id(self):
        """Runs can have correlation IDs."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test", correlation_id="request-123") as run:
            pass
        
        assert run.links.correlation_id == "request-123"


class TestAnnotations:
    """Tests for run annotations."""

    def test_run_annotate(self):
        """Runs can be annotated with custom metadata."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test") as run:
            run.annotate({"version": "1.0", "environment": "test"})
        
        assert run.annotations["version"] == "1.0"
        assert run.annotations["environment"] == "test"

    def test_run_annotate_merges(self):
        """Multiple annotate calls merge annotations."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test") as run:
            run.annotate({"key1": "value1"})
            run.annotate({"key2": "value2"})
        
        assert run.annotations["key1"] == "value1"
        assert run.annotations["key2"] == "value2"

    def test_ars_annotations(self):
        """ARS-style annotations work correctly."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test") as run:
            run.annotate({
                "ars.activation_type": "scheduled",
                "ars.surface_id": "slack-123",
            })
        
        assert run.annotations["ars.activation_type"] == "scheduled"
        assert run.annotations["ars.surface_id"] == "slack-123"


class TestRunRetrieval:
    """Tests for retrieving recorded runs."""

    def test_get_run_by_id(self):
        """Runs can be retrieved by ID."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="test") as run:
            run_id = run.run_id
        
        retrieved = ledger.get_run(run_id)
        assert retrieved is not None
        assert retrieved.run_id == run_id
        assert retrieved.name == "test"

    def test_get_run_not_found(self):
        """Getting non-existent run returns None."""
        ledger = WorkLedger(store=":memory:")
        
        result = ledger.get_run("non-existent-id")
        assert result is None

    def test_list_runs(self):
        """All runs can be listed."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="run1"):
            pass
        with ledger.run(name="run2"):
            pass
        with ledger.run(name="run3"):
            pass
        
        runs = ledger.list_runs()
        assert len(runs) == 3
        names = {r.name for r in runs}
        assert names == {"run1", "run2", "run3"}

    def test_list_runs_by_name(self):
        """Runs can be filtered by name."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="process-request"):
            pass
        with ledger.run(name="process-request"):
            pass
        with ledger.run(name="other-task"):
            pass
        
        runs = ledger.list_runs(name="process-request")
        assert len(runs) == 2

    def test_list_runs_by_status(self):
        """Runs can be filtered by status."""
        ledger = WorkLedger(store=":memory:")
        
        with ledger.run(name="success1"):
            pass
        with ledger.run(name="success2"):
            pass
        with pytest.raises(ValueError):
            with ledger.run(name="failed"):
                raise ValueError("error")
        
        successful = ledger.list_runs(status=RunStatus.SUCCESS)
        assert len(successful) == 2
        
        failed = ledger.list_runs(status=RunStatus.FAILED)
        assert len(failed) == 1
