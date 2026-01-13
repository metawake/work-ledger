"""CrewAI integration for Work Ledger.

Thin wrapper that records CrewAI crew executions without modifying
the crew's behavior.

Example:
    >>> from crewai import Crew, Agent, Task
    >>> from work_ledger import WorkLedger
    >>> from work_ledger.integrations.crewai import wrap_crew
    >>> 
    >>> # Build your crew
    >>> researcher = Agent(role="Researcher", goal="Research topics")
    >>> writer = Agent(role="Writer", goal="Write articles")
    >>> task1 = Task(description="Research AI trends", agent=researcher)
    >>> task2 = Task(description="Write summary", agent=writer)
    >>> crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
    >>> 
    >>> # Wrap it - runs are now automatically recorded
    >>> ledger = WorkLedger(store="./runs")
    >>> wrapped = wrap_crew(crew, ledger)
    >>> 
    >>> result = wrapped.kickoff(inputs={"topic": "AI"})
    >>> 
    >>> # Check recorded runs
    >>> runs = ledger.list_runs()
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from work_ledger.core.ledger import WorkLedger
from work_ledger.core.models import Run, Step, StepKind, Metrics, RunStatus

if TYPE_CHECKING:
    pass  # Would import crewai types here


class WrappedCrew:
    """Wrapper around a CrewAI Crew that records runs.
    
    This is a thin wrapper that:
    - Records each kickoff as a run with inputs/outputs
    - Records each task execution as a step
    - Captures agent information for each step
    - Records token usage metrics
    - Handles errors gracefully
    
    The wrapper preserves the original crew's interface.
    
    Supports fluent interface for configuration:
        >>> wrapped = wrap_crew(crew, ledger).with_name("my-crew")
    """

    def __init__(
        self,
        crew: Any,
        ledger: WorkLedger,
        run_name: str | None = None,
    ) -> None:
        """Initialize the wrapper.
        
        Args:
            crew: The CrewAI Crew to wrap
            ledger: WorkLedger instance for recording
            run_name: Custom name for runs (defaults to crew name)
        """
        self._crew = crew
        self._ledger = ledger
        self._run_name = run_name or getattr(crew, "name", "crewai")

    def with_name(self, name: str) -> "WrappedCrew":
        """Set a custom name for recorded runs.
        
        Args:
            name: Human-readable name for runs
            
        Returns:
            Self for method chaining
            
        Example:
            >>> wrapped = wrap_crew(crew, ledger).with_name("research-team")
        """
        self._run_name = name
        return self

    def kickoff(self, inputs: dict = None) -> Any:
        """Execute the crew and record the run.
        
        Args:
            inputs: Input dictionary for the crew
            
        Returns:
            The crew's output
        """
        run = self._create_run(inputs or {})
        
        result = None
        error = None
        
        try:
            result = self._crew.kickoff(inputs)
            run.status = RunStatus.SUCCESS
            
            # Record output
            run.outputs = {"result": str(result)}
            
            # Record task steps from result
            self._record_task_steps(run, result)
            
            # Record token usage
            self._record_metrics(run, result)
            
        except Exception as e:
            run.status = RunStatus.FAILED
            run.annotations["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            error = e
        
        finally:
            run.ended_at = datetime.now(timezone.utc)
            # Only aggregate if we didn't set metrics directly
            if run.metrics.total_tokens == 0:
                run.metrics = run.aggregate_metrics()
            self._ledger._save_run(run)
        
        if error:
            raise error
        
        return result

    async def kickoff_async(self, inputs: dict = None) -> Any:
        """Execute the crew asynchronously and record the run.
        
        Args:
            inputs: Input dictionary for the crew
            
        Returns:
            The crew's output
        """
        run = self._create_run(inputs or {})
        
        result = None
        error = None
        
        try:
            result = await self._crew.kickoff_async(inputs)
            run.status = RunStatus.SUCCESS
            
            # Record output
            run.outputs = {"result": str(result)}
            
            # Record task steps from result
            self._record_task_steps(run, result)
            
            # Record token usage
            self._record_metrics(run, result)
            
        except Exception as e:
            run.status = RunStatus.FAILED
            run.annotations["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            error = e
        
        finally:
            run.ended_at = datetime.now(timezone.utc)
            # Only aggregate if we didn't set metrics directly
            if run.metrics.total_tokens == 0:
                run.metrics = run.aggregate_metrics()
            self._ledger._save_run(run)
        
        if error:
            raise error
        
        return result

    def _create_run(self, inputs: dict) -> Run:
        """Create a new run for this kickoff."""
        run = Run(name=self._run_name)
        run.started_at = datetime.now(timezone.utc)
        run.status = RunStatus.RUNNING
        run.inputs = self._serialize_inputs(inputs)
        
        # Record crew structure
        run.annotations["agents"] = [
            {"role": getattr(a, "role", str(a))}
            for a in getattr(self._crew, "agents", [])
        ]
        run.annotations["tasks_count"] = len(getattr(self._crew, "tasks", []))
        
        return run

    def _serialize_inputs(self, inputs: Any) -> dict:
        """Serialize inputs to a storable dict."""
        if inputs is None:
            return {}
        if isinstance(inputs, dict):
            return {k: self._serialize_value(v) for k, v in inputs.items()}
        return {"inputs": str(inputs)}

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        return str(value)

    def _record_task_steps(self, run: Run, result: Any) -> None:
        """Extract and record task executions as steps.
        
        Args:
            run: The run to add steps to
            result: The crew's result object
        """
        tasks_output = getattr(result, "tasks_output", [])
        
        for task_output in tasks_output:
            step = Step(
                name=getattr(task_output, "description", "task"),
                kind=StepKind.CUSTOM,
            )
            step.started_at = datetime.now(timezone.utc)
            step.ended_at = datetime.now(timezone.utc)
            
            # Record task output and agent info
            outputs = {}
            raw_output = getattr(task_output, "raw", None)
            if raw_output:
                outputs["result"] = str(raw_output)
            
            # Record agent info in outputs
            agent = getattr(task_output, "agent", None)
            if agent:
                agent_role = agent if isinstance(agent, str) else getattr(agent, "role", str(agent))
                outputs["agent_role"] = agent_role
            
            step.outputs = outputs
            run.add_step(step)

    def _record_metrics(self, run: Run, result: Any) -> None:
        """Record token usage from result.
        
        Args:
            run: The run to update metrics
            result: The crew's result object
        """
        token_usage = getattr(result, "token_usage", {})
        
        if token_usage:
            total = token_usage.get("total_tokens", 0)
            prompt = token_usage.get("prompt_tokens", 0)
            completion = token_usage.get("completion_tokens", 0)
            
            run.metrics = Metrics(
                total_tokens=total,
                prompt_tokens=prompt,
                completion_tokens=completion,
            )

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped crew."""
        return getattr(self._crew, name)


def wrap_crew(
    crew: Any,
    ledger: WorkLedger,
    run_name: str | None = None,
) -> WrappedCrew:
    """Wrap a CrewAI Crew to record runs.
    
    This is the main entry point for the integration. The wrapped
    crew behaves identically to the original, but all executions
    are recorded to the ledger.
    
    Args:
        crew: The CrewAI Crew to wrap
        ledger: WorkLedger instance for recording
        run_name: Custom name for runs (defaults to crew name)
            
    Returns:
        Wrapped crew that records runs
        
    Example:
        >>> from crewai import Crew, Agent, Task
        >>> from work_ledger import WorkLedger
        >>> from work_ledger.integrations.crewai import wrap_crew
        >>> 
        >>> researcher = Agent(role="Researcher", goal="Research topics")
        >>> task = Task(description="Research AI", agent=researcher)
        >>> crew = Crew(agents=[researcher], tasks=[task])
        >>> 
        >>> ledger = WorkLedger(store="./runs")
        >>> wrapped = wrap_crew(crew, ledger)
        >>> 
        >>> result = wrapped.kickoff(inputs={"topic": "AI"})
        >>> print(ledger.list_runs())  # Shows the recorded run
    """
    return WrappedCrew(crew, ledger, run_name)
