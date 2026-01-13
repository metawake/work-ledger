"""Tests for core data models: Run, Step, Metrics."""

from datetime import datetime, timezone
from uuid import UUID

import pytest

from work_ledger.core.models import (
    Metrics,
    Run,
    RunStatus,
    Step,
    StepKind,
    CausalLink,
)


class TestMetrics:
    """Tests for the Metrics dataclass."""

    def test_create_empty_metrics(self):
        """Metrics can be created with default values."""
        metrics = Metrics()
        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.total_tokens == 0
        assert metrics.latency_ms is None
        assert metrics.cost is None
        assert metrics.retries == 0

    def test_create_metrics_with_values(self):
        """Metrics can be created with specific values."""
        metrics = Metrics(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=1234.5,
            cost=0.0015,
            retries=1,
        )
        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 50
        assert metrics.total_tokens == 150
        assert metrics.latency_ms == 1234.5
        assert metrics.cost == 0.0015
        assert metrics.retries == 1

    def test_metrics_add(self):
        """Metrics can be added together."""
        m1 = Metrics(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.001)
        m2 = Metrics(prompt_tokens=200, completion_tokens=100, total_tokens=300, cost=0.002)
        combined = m1 + m2
        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 150
        assert combined.total_tokens == 450
        assert combined.cost == 0.003

    def test_metrics_to_dict(self):
        """Metrics can be serialized to dict."""
        metrics = Metrics(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        d = metrics.to_dict()
        assert d["prompt_tokens"] == 100
        assert d["completion_tokens"] == 50
        assert d["total_tokens"] == 150

    def test_metrics_from_dict(self):
        """Metrics can be deserialized from dict."""
        d = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        metrics = Metrics.from_dict(d)
        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 50
        assert metrics.total_tokens == 150


class TestStepKind:
    """Tests for StepKind enum."""

    def test_step_kinds_exist(self):
        """All required step kinds exist."""
        assert StepKind.MODEL.value == "model"
        assert StepKind.TOOL.value == "tool"
        assert StepKind.RETRIEVAL.value == "retrieval"
        assert StepKind.CUSTOM.value == "custom"


class TestStep:
    """Tests for the Step dataclass."""

    def test_create_step_minimal(self):
        """Step can be created with minimal required fields."""
        step = Step(name="test-step", kind=StepKind.TOOL)
        assert step.name == "test-step"
        assert step.kind == StepKind.TOOL
        assert step.step_id is not None
        # Verify it's a valid UUID
        UUID(step.step_id)

    def test_create_step_with_all_fields(self):
        """Step can be created with all fields."""
        now = datetime.now(timezone.utc)
        step = Step(
            step_id="step-123",
            name="call-api",
            kind=StepKind.TOOL,
            inputs={"url": "https://api.example.com"},
            outputs={"status": 200},
            metrics=Metrics(latency_ms=150),
            started_at=now,
            ended_at=now,
            caused_by="step-000",
        )
        assert step.step_id == "step-123"
        assert step.name == "call-api"
        assert step.kind == StepKind.TOOL
        assert step.inputs == {"url": "https://api.example.com"}
        assert step.outputs == {"status": 200}
        assert step.metrics.latency_ms == 150
        assert step.caused_by == "step-000"

    def test_step_auto_generates_id(self):
        """Step generates a unique ID if not provided."""
        step1 = Step(name="step1", kind=StepKind.MODEL)
        step2 = Step(name="step2", kind=StepKind.MODEL)
        assert step1.step_id != step2.step_id

    def test_step_to_dict(self):
        """Step can be serialized to dict."""
        step = Step(
            step_id="step-123",
            name="test",
            kind=StepKind.TOOL,
            inputs={"x": 1},
            outputs={"y": 2},
        )
        d = step.to_dict()
        assert d["step_id"] == "step-123"
        assert d["name"] == "test"
        assert d["kind"] == "tool"
        assert d["inputs"] == {"x": 1}
        assert d["outputs"] == {"y": 2}

    def test_step_from_dict(self):
        """Step can be deserialized from dict."""
        d = {
            "step_id": "step-123",
            "name": "test",
            "kind": "tool",
            "inputs": {"x": 1},
            "outputs": {"y": 2},
        }
        step = Step.from_dict(d)
        assert step.step_id == "step-123"
        assert step.name == "test"
        assert step.kind == StepKind.TOOL
        assert step.inputs == {"x": 1}
        assert step.outputs == {"y": 2}


class TestRunStatus:
    """Tests for RunStatus enum."""

    def test_run_statuses_exist(self):
        """All required run statuses exist."""
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.SUCCESS.value == "success"
        assert RunStatus.FAILED.value == "failed"
        assert RunStatus.CANCELLED.value == "cancelled"


class TestCausalLink:
    """Tests for CausalLink dataclass."""

    def test_create_causal_link(self):
        """CausalLink can be created with all fields."""
        link = CausalLink(
            caused_by="step-001",
            correlation_id="corr-123",
            parent_run_id="run-parent",
            parent_step_id="step-parent",
        )
        assert link.caused_by == "step-001"
        assert link.correlation_id == "corr-123"
        assert link.parent_run_id == "run-parent"
        assert link.parent_step_id == "step-parent"

    def test_create_causal_link_minimal(self):
        """CausalLink can be created with no fields (all optional)."""
        link = CausalLink()
        assert link.caused_by is None
        assert link.correlation_id is None
        assert link.parent_run_id is None
        assert link.parent_step_id is None

    def test_causal_link_to_dict(self):
        """CausalLink can be serialized to dict."""
        link = CausalLink(caused_by="step-001", correlation_id="corr-123")
        d = link.to_dict()
        assert d["caused_by"] == "step-001"
        assert d["correlation_id"] == "corr-123"

    def test_causal_link_from_dict(self):
        """CausalLink can be deserialized from dict."""
        d = {"caused_by": "step-001", "correlation_id": "corr-123"}
        link = CausalLink.from_dict(d)
        assert link.caused_by == "step-001"
        assert link.correlation_id == "corr-123"


class TestRun:
    """Tests for the Run dataclass."""

    def test_create_run_minimal(self):
        """Run can be created with minimal required fields."""
        run = Run(name="test-run")
        assert run.name == "test-run"
        assert run.run_id is not None
        UUID(run.run_id)
        assert run.status == RunStatus.PENDING
        assert run.steps == []
        assert run.inputs == {}
        assert run.outputs == {}

    def test_create_run_with_all_fields(self):
        """Run can be created with all fields."""
        now = datetime.now(timezone.utc)
        run = Run(
            run_id="run-123",
            name="process-request",
            started_at=now,
            ended_at=now,
            inputs={"query": "test"},
            outputs={"response": "result"},
            status=RunStatus.SUCCESS,
            metrics=Metrics(total_tokens=500),
            steps=[Step(name="step1", kind=StepKind.MODEL)],
            links=CausalLink(parent_run_id="parent-run"),
            annotations={"version": "1.0"},
        )
        assert run.run_id == "run-123"
        assert run.name == "process-request"
        assert run.inputs == {"query": "test"}
        assert run.outputs == {"response": "result"}
        assert run.status == RunStatus.SUCCESS
        assert run.metrics.total_tokens == 500
        assert len(run.steps) == 1
        assert run.links.parent_run_id == "parent-run"
        assert run.annotations == {"version": "1.0"}

    def test_run_auto_generates_id(self):
        """Run generates a unique ID if not provided."""
        run1 = Run(name="run1")
        run2 = Run(name="run2")
        assert run1.run_id != run2.run_id

    def test_run_add_step(self):
        """Steps can be added to a run."""
        run = Run(name="test")
        step = Step(name="step1", kind=StepKind.TOOL)
        run.add_step(step)
        assert len(run.steps) == 1
        assert run.steps[0] == step

    def test_run_aggregate_metrics(self):
        """Run can aggregate metrics from all steps."""
        run = Run(name="test")
        run.add_step(Step(
            name="step1",
            kind=StepKind.MODEL,
            metrics=Metrics(prompt_tokens=100, completion_tokens=50, total_tokens=150, cost=0.001)
        ))
        run.add_step(Step(
            name="step2",
            kind=StepKind.MODEL,
            metrics=Metrics(prompt_tokens=200, completion_tokens=100, total_tokens=300, cost=0.002)
        ))
        aggregated = run.aggregate_metrics()
        assert aggregated.prompt_tokens == 300
        assert aggregated.completion_tokens == 150
        assert aggregated.total_tokens == 450
        assert aggregated.cost == 0.003

    def test_run_to_dict(self):
        """Run can be serialized to dict."""
        run = Run(
            run_id="run-123",
            name="test",
            inputs={"x": 1},
            outputs={"y": 2},
            status=RunStatus.SUCCESS,
        )
        d = run.to_dict()
        assert d["run_id"] == "run-123"
        assert d["name"] == "test"
        assert d["inputs"] == {"x": 1}
        assert d["outputs"] == {"y": 2}
        assert d["status"] == "success"

    def test_run_from_dict(self):
        """Run can be deserialized from dict."""
        d = {
            "run_id": "run-123",
            "name": "test",
            "inputs": {"x": 1},
            "outputs": {"y": 2},
            "status": "success",
            "steps": [],
        }
        run = Run.from_dict(d)
        assert run.run_id == "run-123"
        assert run.name == "test"
        assert run.inputs == {"x": 1}
        assert run.outputs == {"y": 2}
        assert run.status == RunStatus.SUCCESS

    def test_run_to_dict_with_steps(self):
        """Run serializes steps correctly."""
        run = Run(name="test")
        run.add_step(Step(step_id="step-1", name="s1", kind=StepKind.TOOL))
        run.add_step(Step(step_id="step-2", name="s2", kind=StepKind.MODEL))
        d = run.to_dict()
        assert len(d["steps"]) == 2
        assert d["steps"][0]["step_id"] == "step-1"
        assert d["steps"][1]["step_id"] == "step-2"

    def test_run_from_dict_with_steps(self):
        """Run deserializes steps correctly."""
        d = {
            "run_id": "run-123",
            "name": "test",
            "status": "success",
            "steps": [
                {"step_id": "step-1", "name": "s1", "kind": "tool"},
                {"step_id": "step-2", "name": "s2", "kind": "model"},
            ],
        }
        run = Run.from_dict(d)
        assert len(run.steps) == 2
        assert run.steps[0].step_id == "step-1"
        assert run.steps[0].kind == StepKind.TOOL
        assert run.steps[1].step_id == "step-2"
        assert run.steps[1].kind == StepKind.MODEL

    def test_run_duration(self):
        """Run can calculate its duration."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 5, tzinfo=timezone.utc)
        run = Run(name="test", started_at=start, ended_at=end)
        assert run.duration_ms == 5000.0

    def test_run_duration_none_when_not_ended(self):
        """Run duration is None when not ended."""
        run = Run(name="test", started_at=datetime.now(timezone.utc))
        assert run.duration_ms is None

    def test_run_get_step_by_id(self):
        """Run can retrieve a step by its ID."""
        run = Run(name="test")
        step1 = Step(step_id="step-1", name="s1", kind=StepKind.TOOL)
        step2 = Step(step_id="step-2", name="s2", kind=StepKind.MODEL)
        run.add_step(step1)
        run.add_step(step2)
        
        found = run.get_step("step-2")
        assert found == step2
        
        not_found = run.get_step("step-999")
        assert not_found is None

    def test_run_get_steps_by_kind(self):
        """Run can filter steps by kind."""
        run = Run(name="test")
        run.add_step(Step(name="t1", kind=StepKind.TOOL))
        run.add_step(Step(name="m1", kind=StepKind.MODEL))
        run.add_step(Step(name="t2", kind=StepKind.TOOL))
        run.add_step(Step(name="r1", kind=StepKind.RETRIEVAL))
        
        tools = run.get_steps_by_kind(StepKind.TOOL)
        assert len(tools) == 2
        
        models = run.get_steps_by_kind(StepKind.MODEL)
        assert len(models) == 1

    def test_run_causal_chain(self):
        """Run can trace causal chain from a step."""
        run = Run(name="test")
        step1 = Step(step_id="step-1", name="trigger", kind=StepKind.CUSTOM)
        step2 = Step(step_id="step-2", name="process", kind=StepKind.MODEL, caused_by="step-1")
        step3 = Step(step_id="step-3", name="action", kind=StepKind.TOOL, caused_by="step-2")
        run.add_step(step1)
        run.add_step(step2)
        run.add_step(step3)
        
        chain = run.get_causal_chain("step-3")
        assert [s.step_id for s in chain] == ["step-1", "step-2", "step-3"]
