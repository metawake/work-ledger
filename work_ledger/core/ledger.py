"""Main WorkLedger class and context managers.

This module provides the primary API for recording agent runs:
- WorkLedger: The main entry point for recording runs
- RunContext: Context manager for recording a run
- StepContext: Context manager for recording steps within a run
"""

from __future__ import annotations

import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from work_ledger.core.models import (
    CausalLink,
    Metrics,
    Run,
    RunStatus,
    Step,
    StepKind,
)
from work_ledger.core.store import RunStore


class StepContext:
    """Context manager for recording a step within a run.
    
    Automatically tracks timing and adds the step to the parent run
    when the context exits.
    
    Example:
        >>> with run.step(name="call-api", kind="tool") as step:
        ...     result = api.call()
        ...     step.record_output({"result": result})
    """

    def __init__(
        self,
        run: RunContext,
        name: str,
        kind: StepKind | str,
        caused_by: str | None = None,
    ) -> None:
        """Initialize the step context.
        
        Args:
            run: The parent run context
            name: Human-readable step name
            kind: Step type (model, tool, retrieval, custom)
            caused_by: ID of the step that caused this one
        """
        self._run = run
        
        if isinstance(kind, str):
            kind = StepKind(kind)
        
        self._step = Step(
            name=name,
            kind=kind,
            caused_by=caused_by,
        )

    @property
    def step_id(self) -> str:
        """Get the step's unique identifier."""
        return self._step.step_id

    @property
    def name(self) -> str:
        """Get the step's name."""
        return self._step.name

    @property
    def kind(self) -> StepKind:
        """Get the step's kind."""
        return self._step.kind

    @property
    def inputs(self) -> dict[str, Any]:
        """Get the step's inputs."""
        return self._step.inputs

    @property
    def outputs(self) -> dict[str, Any]:
        """Get the step's outputs."""
        return self._step.outputs

    @property
    def metrics(self) -> Metrics:
        """Get the step's metrics."""
        return self._step.metrics

    @property
    def started_at(self) -> datetime | None:
        """Get when the step started."""
        return self._step.started_at

    @property
    def ended_at(self) -> datetime | None:
        """Get when the step ended."""
        return self._step.ended_at

    @property
    def caused_by(self) -> str | None:
        """Get the ID of the step that caused this one."""
        return self._step.caused_by

    def record_input(self, inputs: dict[str, Any]) -> None:
        """Record input data for this step.
        
        Args:
            inputs: The input data to record
        """
        self._step.inputs.update(inputs)

    def record_output(self, outputs: dict[str, Any]) -> None:
        """Record output data for this step.
        
        Args:
            outputs: The output data to record
        """
        self._step.outputs.update(outputs)

    def record_metrics(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: float | None = None,
        cost: float | None = None,
        retries: int = 0,
    ) -> None:
        """Record metrics for this step.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens used
            latency_ms: Latency in milliseconds
            cost: Cost in USD
            retries: Number of retries
        """
        self._step.metrics = Metrics(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost=cost,
            retries=retries,
        )

    def __enter__(self) -> StepContext:
        """Enter the step context."""
        self._step.started_at = datetime.now(timezone.utc)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the step context and add step to run."""
        self._step.ended_at = datetime.now(timezone.utc)
        self._run._add_step(self._step)


class RunContext:
    """Context manager for recording a run.
    
    Automatically tracks timing, status, and persists the run
    when the context exits.
    
    Example:
        >>> with ledger.run(name="process-request") as run:
        ...     run.record_input({"query": "test"})
        ...     # perform work
        ...     run.record_output({"result": "done"})
    """

    def __init__(
        self,
        ledger: WorkLedger,
        name: str,
        parent_run_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Initialize the run context.
        
        Args:
            ledger: The parent WorkLedger
            name: Human-readable run name
            parent_run_id: ID of the parent run (for hierarchical runs)
            correlation_id: ID for grouping related operations
        """
        self._ledger = ledger
        self._run = Run(
            name=name,
            links=CausalLink(
                parent_run_id=parent_run_id,
                correlation_id=correlation_id,
            ),
        )

    @property
    def run_id(self) -> str:
        """Get the run's unique identifier."""
        return self._run.run_id

    @property
    def name(self) -> str:
        """Get the run's name."""
        return self._run.name

    @property
    def status(self) -> RunStatus:
        """Get the run's status."""
        return self._run.status

    @property
    def inputs(self) -> dict[str, Any]:
        """Get the run's inputs."""
        return self._run.inputs

    @property
    def outputs(self) -> dict[str, Any]:
        """Get the run's outputs."""
        return self._run.outputs

    @property
    def steps(self) -> list[Step]:
        """Get the run's steps."""
        return self._run.steps

    @property
    def started_at(self) -> datetime | None:
        """Get when the run started."""
        return self._run.started_at

    @property
    def ended_at(self) -> datetime | None:
        """Get when the run ended."""
        return self._run.ended_at

    @property
    def annotations(self) -> dict[str, Any]:
        """Get the run's annotations."""
        return self._run.annotations

    @property
    def links(self) -> CausalLink:
        """Get the run's causal links."""
        return self._run.links

    def record_input(self, inputs: dict[str, Any]) -> None:
        """Record input data for this run.
        
        Args:
            inputs: The input data to record
        """
        self._run.inputs.update(inputs)

    def record_output(self, outputs: dict[str, Any]) -> None:
        """Record output data for this run.
        
        Args:
            outputs: The output data to record
        """
        self._run.outputs.update(outputs)

    def annotate(self, annotations: dict[str, Any]) -> None:
        """Add annotations to this run.
        
        Annotations are merged with existing annotations.
        
        Args:
            annotations: Key-value pairs to add
        """
        self._run.annotations.update(annotations)

    def step(
        self,
        name: str,
        kind: StepKind | str,
        caused_by: str | None = None,
    ) -> StepContext:
        """Create a step context within this run.
        
        Args:
            name: Human-readable step name
            kind: Step type (model, tool, retrieval, custom)
            caused_by: ID of the step that caused this one
            
        Returns:
            A StepContext for recording the step
        """
        return StepContext(self, name, kind, caused_by)

    def _add_step(self, step: Step) -> None:
        """Add a completed step to this run.
        
        Args:
            step: The step to add
        """
        self._run.add_step(step)

    def __enter__(self) -> RunContext:
        """Enter the run context."""
        self._run.started_at = datetime.now(timezone.utc)
        self._run.status = RunStatus.RUNNING
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the run context and persist the run."""
        self._run.ended_at = datetime.now(timezone.utc)
        
        if exc_type is None:
            self._run.status = RunStatus.SUCCESS
        else:
            self._run.status = RunStatus.FAILED
            self._run.annotations["error"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_val) if exc_val else None,
                "traceback": traceback.format_exc() if exc_tb else None,
            }
        
        # Aggregate metrics from steps
        self._run.metrics = self._run.aggregate_metrics()
        
        # Persist the run
        self._ledger._save_run(self._run)


class WorkLedger:
    """Main entry point for recording agent runs.
    
    WorkLedger provides the API for creating, storing, and retrieving
    run artifacts. It manages the storage backend and provides context
    managers for recording runs and steps.
    
    Example:
        >>> ledger = WorkLedger(store="./runs")
        >>> with ledger.run(name="process-request") as run:
        ...     run.record_input({"query": "test"})
        ...     with run.step(name="llm-call", kind="model") as step:
        ...         # perform LLM call
        ...         step.record_output({"response": "result"})
        ...     run.record_output({"result": "done"})
    """

    def __init__(self, store: str | Path = ":memory:") -> None:
        """Initialize the WorkLedger.
        
        Args:
            store: Storage specification:
                   - ":memory:" for in-memory storage (default)
                   - Path string or Path object for JSONL file storage
        """
        self._store = RunStore.create(store)

    def run(
        self,
        name: str,
        parent_run_id: str | None = None,
        correlation_id: str | None = None,
    ) -> RunContext:
        """Create a run context.
        
        Args:
            name: Human-readable run name
            parent_run_id: ID of the parent run (for hierarchical runs)
            correlation_id: ID for grouping related operations
            
        Returns:
            A RunContext for recording the run
            
        Example:
            >>> with ledger.run(name="process-request") as run:
            ...     # perform work
            ...     pass
        """
        return RunContext(
            self,
            name,
            parent_run_id=parent_run_id,
            correlation_id=correlation_id,
        )

    def _save_run(self, run: Run) -> None:
        """Save a run to storage.
        
        Args:
            run: The run to save
        """
        self._store.save_run(run)

    def get_run(self, run_id: str) -> Run | None:
        """Retrieve a run by ID.
        
        Args:
            run_id: The unique identifier of the run
            
        Returns:
            The run if found, None otherwise
        """
        return self._store.get_run(run_id)

    def list_runs(
        self,
        name: str | None = None,
        status: RunStatus | None = None,
    ) -> list[Run]:
        """List runs with optional filtering.
        
        Args:
            name: Filter by run name (optional)
            status: Filter by run status (optional)
            
        Returns:
            List of matching runs
        """
        return self._store.list_runs(name=name, status=status)

    def delete_run(self, run_id: str) -> None:
        """Delete a run from storage.
        
        Args:
            run_id: The unique identifier of the run to delete
        """
        self._store.delete_run(run_id)

    def wrap(
        self,
        obj: Any,
        run_name: str | None = None,
        replay_from: str | None = None,
    ) -> Any:
        """Wrap an agent, graph, or crew to record/replay runs.
        
        Auto-detects the type of object and returns the appropriate wrapper.
        This is a convenience method that provides a unified interface for
        all integrations.
        
        Args:
            obj: The agent, graph, or crew to wrap. Supported types:
                 - PydanticAI Agent
                 - LangGraph CompiledGraph
                 - CrewAI Crew
                 - LangChain Chain/Runnable
                 - LlamaIndex QueryEngine
                 - OpenAI client
                 - Anthropic client
            run_name: Custom name for runs (defaults to object name)
            replay_from: Run ID to replay from (no API calls made)
            
        Returns:
            Wrapped object that records or replays runs
            
        Raises:
            TypeError: If the object type is not supported
            
        Example:
            >>> ledger = WorkLedger(store="./runs")
            >>> 
            >>> # Works with any supported framework
            >>> wrapped = ledger.wrap(my_agent)
            >>> wrapped = ledger.wrap(my_graph)
            >>> wrapped = ledger.wrap(my_crew)
            >>> 
            >>> # With options
            >>> wrapped = ledger.wrap(my_agent, run_name="my-agent")
            >>> wrapped = ledger.wrap(my_agent, replay_from="run_abc123")
        """
        obj_type = type(obj)
        module = obj_type.__module__
        class_name = obj_type.__name__
        
        # PydanticAI Agent
        if "pydantic_ai" in module or class_name == "Agent":
            if hasattr(obj, "run_sync") or hasattr(obj, "run"):
                from work_ledger.integrations.pydantic_ai import wrap_agent
                return wrap_agent(obj, self, run_name=run_name, replay_from=replay_from)
        
        # LangGraph CompiledGraph
        if "langgraph" in module or class_name in ("CompiledGraph", "CompiledStateGraph"):
            from work_ledger.integrations.langgraph import wrap_graph
            return wrap_graph(obj, self, run_name=run_name)
        
        # CrewAI Crew
        if "crewai" in module or class_name == "Crew":
            if hasattr(obj, "kickoff"):
                from work_ledger.integrations.crewai import wrap_crew
                return wrap_crew(obj, self, run_name=run_name)
        
        # LangChain Chain/Runnable
        if "langchain" in module:
            if hasattr(obj, "invoke") or hasattr(obj, "__call__"):
                from work_ledger.integrations.langchain import wrap_chain
                return wrap_chain(obj, self, run_name=run_name)
        
        # LlamaIndex QueryEngine
        if "llama_index" in module or "llamaindex" in module:
            if hasattr(obj, "query"):
                from work_ledger.integrations.llamaindex import wrap_query_engine
                return wrap_query_engine(obj, self, run_name=run_name)
        
        # OpenAI client
        if "openai" in module and class_name in ("OpenAI", "AsyncOpenAI"):
            from work_ledger.integrations.openai import wrap_openai
            return wrap_openai(obj, self, replay_from=replay_from)
        
        # Anthropic client
        if "anthropic" in module and class_name in ("Anthropic", "AsyncAnthropic"):
            from work_ledger.integrations.anthropic import wrap_anthropic
            return wrap_anthropic(obj, self, replay_from=replay_from)
        
        raise TypeError(
            f"Unsupported object type: {module}.{class_name}. "
            f"Supported types: PydanticAI Agent, LangGraph CompiledGraph, "
            f"CrewAI Crew, LangChain Chain, LlamaIndex QueryEngine, "
            f"OpenAI client, Anthropic client."
        )
