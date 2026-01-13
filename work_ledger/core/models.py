"""Core data models for Work Ledger.

This module defines the fundamental data structures:
- Run: A complete execution of an agent system
- Step: A single operation within a run (model call, tool call, etc.)
- Metrics: Vendor-agnostic performance and cost metrics
- CausalLink: Explicit causality relationships between steps/runs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class StepKind(Enum):
    """Types of steps that can occur in a run.
    
    Attributes:
        MODEL: An LLM model invocation
        TOOL: A tool/function call
        RETRIEVAL: A retrieval/search operation
        CUSTOM: User-defined step type
    """
    MODEL = "model"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    CUSTOM = "custom"


class RunStatus(Enum):
    """Status of a run.
    
    Attributes:
        PENDING: Run created but not started
        RUNNING: Run is currently executing
        SUCCESS: Run completed successfully
        FAILED: Run failed with an error
        CANCELLED: Run was cancelled
    """
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Metrics:
    """Vendor-agnostic metrics for runs and steps.
    
    Captures token usage, timing, cost, and retry information
    without being tied to any specific LLM provider.
    
    Attributes:
        prompt_tokens: Number of tokens in the prompt/input
        completion_tokens: Number of tokens in the completion/output
        total_tokens: Total tokens used (prompt + completion)
        latency_ms: Time taken in milliseconds
        cost: Estimated cost in USD
        retries: Number of retry attempts
        
    Example:
        >>> metrics = Metrics(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        >>> metrics.to_dict()
        {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150, ...}
    """
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float | None = None
    cost: float | None = None
    retries: int = 0

    def __add__(self, other: Metrics) -> Metrics:
        """Add two Metrics together, combining token counts and costs.
        
        Args:
            other: Another Metrics instance to add
            
        Returns:
            A new Metrics instance with combined values
        """
        def add_optional(a: float | None, b: float | None) -> float | None:
            if a is None and b is None:
                return None
            return (a or 0) + (b or 0)
        
        return Metrics(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            latency_ms=add_optional(self.latency_ms, other.latency_ms),
            cost=add_optional(self.cost, other.cost),
            retries=self.retries + other.retries,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a dictionary.
        
        Returns:
            Dictionary representation of the metrics
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "retries": self.retries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metrics:
        """Deserialize metrics from a dictionary.
        
        Args:
            data: Dictionary containing metrics fields
            
        Returns:
            A new Metrics instance
        """
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            latency_ms=data.get("latency_ms"),
            cost=data.get("cost"),
            retries=data.get("retries", 0),
        )


@dataclass
class CausalLink:
    """Explicit causality relationships between steps and runs.
    
    Models the "what caused what" relationships, not just temporal sequence.
    
    Attributes:
        caused_by: ID of the step/event that directly caused this
        correlation_id: ID for grouping related operations
        parent_run_id: ID of the parent run (for hierarchical runs)
        parent_step_id: ID of the parent step
        
    Example:
        >>> link = CausalLink(caused_by="step-001", correlation_id="request-123")
        >>> link.to_dict()
        {'caused_by': 'step-001', 'correlation_id': 'request-123', ...}
    """
    caused_by: str | None = None
    correlation_id: str | None = None
    parent_run_id: str | None = None
    parent_step_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize causal link to a dictionary.
        
        Returns:
            Dictionary representation of the causal link
        """
        return {
            "caused_by": self.caused_by,
            "correlation_id": self.correlation_id,
            "parent_run_id": self.parent_run_id,
            "parent_step_id": self.parent_step_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalLink:
        """Deserialize causal link from a dictionary.
        
        Args:
            data: Dictionary containing causal link fields
            
        Returns:
            A new CausalLink instance
        """
        return cls(
            caused_by=data.get("caused_by"),
            correlation_id=data.get("correlation_id"),
            parent_run_id=data.get("parent_run_id"),
            parent_step_id=data.get("parent_step_id"),
        )


def _generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid4())


@dataclass
class Step:
    """A single operation within a run.
    
    Steps represent discrete units of work: model calls, tool invocations,
    retrieval operations, or custom actions. Each step captures its inputs,
    outputs, metrics, and causal relationships.
    
    Attributes:
        name: Human-readable name for the step
        kind: Type of step (model, tool, retrieval, custom)
        step_id: Unique identifier (auto-generated if not provided)
        inputs: Input data for this step
        outputs: Output data from this step
        metrics: Performance and cost metrics
        started_at: When the step started
        ended_at: When the step ended
        caused_by: ID of the step that caused this one
        annotations: Optional extensible metadata (e.g., fixtures for replay)
        
    Example:
        >>> step = Step(name="call-api", kind=StepKind.TOOL)
        >>> step.inputs = {"url": "https://api.example.com"}
        >>> step.outputs = {"status": 200}
    """
    name: str
    kind: StepKind
    step_id: str = field(default_factory=_generate_id)
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: Metrics = field(default_factory=Metrics)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    caused_by: str | None = None
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize step to a dictionary.
        
        Returns:
            Dictionary representation of the step
        """
        return {
            "step_id": self.step_id,
            "name": self.name,
            "kind": self.kind.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "metrics": self.metrics.to_dict(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "caused_by": self.caused_by,
            "annotations": self.annotations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        """Deserialize step from a dictionary.
        
        Args:
            data: Dictionary containing step fields
            
        Returns:
            A new Step instance
        """
        started_at = None
        if data.get("started_at"):
            started_at = datetime.fromisoformat(data["started_at"])
        
        ended_at = None
        if data.get("ended_at"):
            ended_at = datetime.fromisoformat(data["ended_at"])
        
        metrics = Metrics()
        if data.get("metrics"):
            metrics = Metrics.from_dict(data["metrics"])
        
        return cls(
            step_id=data.get("step_id", _generate_id()),
            name=data["name"],
            kind=StepKind(data["kind"]),
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            metrics=metrics,
            started_at=started_at,
            ended_at=ended_at,
            caused_by=data.get("caused_by"),
            annotations=data.get("annotations", {}),
        )


@dataclass
class Run:
    """A complete execution of an agent system.
    
    A Run represents the system's reaction to an activation — not just a
    function call, but the complete work performed in response to a trigger.
    Runs have clear temporal boundaries, structured steps, and explicit
    causal relationships.
    
    Attributes:
        name: Human-readable name for the run
        run_id: Unique identifier (auto-generated if not provided)
        started_at: When the run started
        ended_at: When the run ended
        inputs: Input data that triggered the run
        outputs: Final output data from the run
        status: Current status of the run
        metrics: Aggregated metrics for the entire run
        steps: Ordered list of steps in this run
        links: Causal relationships to other runs/steps
        annotations: Optional extensible metadata
        
    Example:
        >>> with ledger.run(name="process-request") as run:
        ...     run.record_input({"query": "test"})
        ...     # perform work
        ...     run.record_output({"result": "done"})
    """
    name: str
    run_id: str = field(default_factory=_generate_id)
    started_at: datetime | None = None
    ended_at: datetime | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    status: RunStatus = RunStatus.PENDING
    metrics: Metrics = field(default_factory=Metrics)
    steps: list[Step] = field(default_factory=list)
    links: CausalLink = field(default_factory=CausalLink)
    annotations: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: Step) -> None:
        """Add a step to this run.
        
        Args:
            step: The step to add
        """
        self.steps.append(step)

    def get_step(self, step_id: str) -> Step | None:
        """Get a step by its ID.
        
        Args:
            step_id: The unique identifier of the step
            
        Returns:
            The step if found, None otherwise
        """
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_steps_by_kind(self, kind: StepKind) -> list[Step]:
        """Get all steps of a specific kind.
        
        Args:
            kind: The step kind to filter by
            
        Returns:
            List of steps matching the kind
        """
        return [step for step in self.steps if step.kind == kind]

    def aggregate_metrics(self) -> Metrics:
        """Aggregate metrics from all steps.
        
        Returns:
            Combined metrics from all steps in the run
        """
        result = Metrics()
        for step in self.steps:
            result = result + step.metrics
        return result

    @property
    def duration_ms(self) -> float | None:
        """Calculate the duration of the run in milliseconds.
        
        Returns:
            Duration in milliseconds, or None if run hasn't ended
        """
        if self.started_at is None or self.ended_at is None:
            return None
        delta = self.ended_at - self.started_at
        return delta.total_seconds() * 1000

    def get_causal_chain(self, step_id: str) -> list[Step]:
        """Trace the causal chain leading to a step.
        
        Follows the caused_by links backwards to reconstruct
        the chain of steps that led to the specified step.
        
        Args:
            step_id: The step to trace back from
            
        Returns:
            List of steps in causal order (root cause first)
        """
        chain: list[Step] = []
        current_id: str | None = step_id
        
        # Build chain backwards
        while current_id:
            step = self.get_step(current_id)
            if step is None:
                break
            chain.append(step)
            current_id = step.caused_by
        
        # Return in causal order (root first)
        chain.reverse()
        return chain

    def to_dict(self) -> dict[str, Any]:
        """Serialize run to a dictionary.
        
        Returns:
            Dictionary representation of the run
        """
        return {
            "run_id": self.run_id,
            "name": self.name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "status": self.status.value,
            "metrics": self.metrics.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "links": self.links.to_dict(),
            "annotations": self.annotations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Run:
        """Deserialize run from a dictionary.
        
        Args:
            data: Dictionary containing run fields
            
        Returns:
            A new Run instance
        """
        started_at = None
        if data.get("started_at"):
            started_at = datetime.fromisoformat(data["started_at"])
        
        ended_at = None
        if data.get("ended_at"):
            ended_at = datetime.fromisoformat(data["ended_at"])
        
        metrics = Metrics()
        if data.get("metrics"):
            metrics = Metrics.from_dict(data["metrics"])
        
        links = CausalLink()
        if data.get("links"):
            links = CausalLink.from_dict(data["links"])
        
        steps = []
        for step_data in data.get("steps", []):
            steps.append(Step.from_dict(step_data))
        
        status = RunStatus.PENDING
        if data.get("status"):
            status = RunStatus(data["status"])
        
        return cls(
            run_id=data.get("run_id", _generate_id()),
            name=data["name"],
            started_at=started_at,
            ended_at=ended_at,
            inputs=data.get("inputs", {}),
            outputs=data.get("outputs", {}),
            status=status,
            metrics=metrics,
            steps=steps,
            links=links,
            annotations=data.get("annotations", {}),
        )
