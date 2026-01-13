"""LangGraph integration for Work Ledger.

Thin wrapper that records LangGraph executions without modifying
the graph's behavior.

Example:
    >>> from langgraph.graph import StateGraph
    >>> from work_ledger import WorkLedger
    >>> from work_ledger.integrations.langgraph import wrap_graph
    >>> 
    >>> # Build your graph
    >>> graph = StateGraph(...)
    >>> graph.add_node("agent", agent_node)
    >>> graph.add_edge("agent", "tools")
    >>> compiled = graph.compile()
    >>> 
    >>> # Wrap it - runs are now automatically recorded
    >>> ledger = WorkLedger(store="./runs")
    >>> wrapped = wrap_graph(compiled, ledger)
    >>> 
    >>> result = wrapped.invoke({"messages": [...]})
    >>> 
    >>> # Check recorded runs
    >>> runs = ledger.list_runs()
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, Iterator, AsyncIterator, TYPE_CHECKING

from work_ledger.core.ledger import WorkLedger
from work_ledger.core.models import Run, Step, StepKind, RunStatus

if TYPE_CHECKING:
    pass  # Would import langgraph types here


class WrappedGraph:
    """Wrapper around a compiled LangGraph that records runs.
    
    This is a thin wrapper that:
    - Records each invoke/stream as a run with inputs/outputs
    - Records node executions as steps (when using stream)
    - Captures errors gracefully
    - Preserves the original graph's interface
    
    Supports fluent interface for configuration:
        >>> wrapped = wrap_graph(graph, ledger).with_name("my-graph")
        >>> wrapped = wrap_graph(graph, ledger).with_stream_recording()
    """

    def __init__(
        self,
        graph: Any,
        ledger: WorkLedger,
        run_name: str | None = None,
        record_stream: bool = False,
    ) -> None:
        """Initialize the wrapper.
        
        Args:
            graph: The compiled LangGraph to wrap
            ledger: WorkLedger instance for recording
            run_name: Custom name for runs (defaults to graph name)
            record_stream: Whether to record individual stream events as steps
        """
        self._graph = graph
        self._ledger = ledger
        self._run_name = run_name or getattr(graph, "name", "langgraph")
        self._record_stream = record_stream

    def with_name(self, name: str) -> "WrappedGraph":
        """Set a custom name for recorded runs.
        
        Args:
            name: Human-readable name for runs
            
        Returns:
            Self for method chaining
            
        Example:
            >>> wrapped = wrap_graph(graph, ledger).with_name("my-workflow")
        """
        self._run_name = name
        return self

    def with_stream_recording(self, enabled: bool = True) -> "WrappedGraph":
        """Enable or disable recording of stream events as steps.
        
        Args:
            enabled: Whether to record stream events (default True)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> wrapped = wrap_graph(graph, ledger).with_stream_recording()
        """
        self._record_stream = enabled
        return self

    def invoke(self, state: dict, config: dict = None) -> dict:
        """Invoke the graph and record the run.
        
        Args:
            state: Input state for the graph
            config: Optional LangGraph config
            
        Returns:
            The graph's output state
        """
        run = self._create_run(state)
        
        result = None
        error = None
        
        try:
            result = self._graph.invoke(state, config)
            run.status = RunStatus.SUCCESS
            run.outputs = self._extract_outputs(result)
            
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
            self._ledger._save_run(run)
        
        if error:
            raise error
        
        return result

    async def ainvoke(self, state: dict, config: dict = None) -> dict:
        """Invoke the graph asynchronously and record the run.
        
        Args:
            state: Input state for the graph
            config: Optional LangGraph config
            
        Returns:
            The graph's output state
        """
        run = self._create_run(state)
        
        result = None
        error = None
        
        try:
            result = await self._graph.ainvoke(state, config)
            run.status = RunStatus.SUCCESS
            run.outputs = self._extract_outputs(result)
            
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
            self._ledger._save_run(run)
        
        if error:
            raise error
        
        return result

    def stream(self, state: dict, config: dict = None) -> Iterator[dict]:
        """Stream graph execution and record the run.
        
        Args:
            state: Input state for the graph
            config: Optional LangGraph config
            
        Yields:
            Stream events from the graph
        """
        run = self._create_run(state)
        
        error = None
        last_output = None
        
        try:
            for event in self._graph.stream(state, config):
                # Record node steps if enabled
                if self._record_stream:
                    self._record_stream_event(run, event)
                
                last_output = event
                yield event
            
            run.status = RunStatus.SUCCESS
            if last_output:
                run.outputs = self._extract_outputs(last_output)
            
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
            self._ledger._save_run(run)
        
        if error:
            raise error

    async def astream_events(self, state: dict, config: dict = None) -> AsyncIterator:
        """Stream graph events asynchronously and record the run.
        
        Args:
            state: Input state for the graph
            config: Optional LangGraph config
            
        Yields:
            Stream events from the graph
        """
        run = self._create_run(state)
        
        error = None
        
        try:
            async for event in self._graph.astream_events(state, config):
                # Record events if enabled
                if self._record_stream:
                    self._record_astream_event(run, event)
                
                yield event
            
            run.status = RunStatus.SUCCESS
            
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
            self._ledger._save_run(run)
        
        if error:
            raise error

    def _create_run(self, state: dict) -> Run:
        """Create a new run for this invocation."""
        run = Run(name=self._run_name)
        run.started_at = datetime.now(timezone.utc)
        run.status = RunStatus.RUNNING
        run.inputs = self._serialize_state(state)
        return run

    def _serialize_state(self, state: Any) -> dict:
        """Serialize state to a storable dict."""
        if isinstance(state, dict):
            return {k: self._serialize_value(v) for k, v in state.items()}
        return {"state": str(state)}

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        # For complex objects (messages, etc.), convert to string
        return str(value)

    def _extract_outputs(self, result: Any) -> dict:
        """Extract outputs from result."""
        if isinstance(result, dict):
            return self._serialize_state(result)
        return {"result": str(result)}

    def _record_stream_event(self, run: Run, event: dict) -> None:
        """Record a stream event as a step."""
        if isinstance(event, dict):
            for node_name, node_output in event.items():
                step = Step(
                    name=node_name,
                    kind=StepKind.CUSTOM,
                )
                step.started_at = datetime.now(timezone.utc)
                step.ended_at = datetime.now(timezone.utc)
                step.outputs = self._serialize_state(node_output) if isinstance(node_output, dict) else {"output": str(node_output)}
                run.add_step(step)

    def _record_astream_event(self, run: Run, event: Any) -> None:
        """Record an async stream event as a step."""
        # Handle LangGraph's astream_events format
        event_type = getattr(event, "event", None)
        event_name = getattr(event, "name", "unknown")
        
        if event_type == "on_chain_end":
            step = Step(
                name=event_name,
                kind=StepKind.CUSTOM,
            )
            step.started_at = datetime.now(timezone.utc)
            step.ended_at = datetime.now(timezone.utc)
            
            data = getattr(event, "data", {})
            if isinstance(data, dict):
                step.outputs = self._serialize_state(data)
            
            run.add_step(step)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped graph."""
        return getattr(self._graph, name)


def wrap_graph(
    graph: Any,
    ledger: WorkLedger,
    run_name: str | None = None,
    record_stream: bool = False,
) -> WrappedGraph:
    """Wrap a compiled LangGraph to record runs.
    
    This is the main entry point for the integration. The wrapped
    graph behaves identically to the original, but all executions
    are recorded to the ledger.
    
    Args:
        graph: The compiled LangGraph to wrap
        ledger: WorkLedger instance for recording
        run_name: Custom name for runs (defaults to graph name)
        record_stream: Whether to record stream events as individual steps
            
    Returns:
        Wrapped graph that records runs
        
    Example:
        >>> from langgraph.graph import StateGraph
        >>> from work_ledger import WorkLedger
        >>> from work_ledger.integrations.langgraph import wrap_graph
        >>> 
        >>> builder = StateGraph(State)
        >>> builder.add_node("agent", agent_node)
        >>> builder.add_edge(START, "agent")
        >>> graph = builder.compile()
        >>> 
        >>> ledger = WorkLedger(store="./runs")
        >>> wrapped = wrap_graph(graph, ledger)
        >>> 
        >>> result = wrapped.invoke({"messages": [HumanMessage("Hello!")]})
        >>> print(ledger.list_runs())  # Shows the recorded run
    """
    return WrappedGraph(graph, ledger, run_name, record_stream)
