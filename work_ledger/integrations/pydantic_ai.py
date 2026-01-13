"""PydanticAI integration for Work Ledger.

Thin wrapper that records PydanticAI agent runs and supports replay.

Example:
    >>> from pydantic_ai import Agent
    >>> from work_ledger import WorkLedger
    >>> from work_ledger.integrations.pydantic_ai import wrap_agent
    >>>
    >>> ledger = WorkLedger(store="./runs")
    >>> agent = Agent("openai:gpt-4")
    >>>
    >>> # Record mode (default)
    >>> wrapped = wrap_agent(agent, ledger)
    >>> result = wrapped.run_sync("What's the weather?")
    >>>
    >>> # Replay mode (no API calls)
    >>> wrapped = wrap_agent(agent, ledger, replay_from="run_abc123")
    >>> result = wrapped.run_sync("What's the weather?")  # Returns saved
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from work_ledger.core.ledger import WorkLedger
from work_ledger.core.models import Run, Step, StepKind, Metrics, RunStatus

if TYPE_CHECKING:
    pass


class ReplayError(Exception):
    """Raised when replay fails due to divergence from recording."""
    pass


class MockAgentResult:
    """Mock PydanticAI result for replay."""
    
    def __init__(self, data: dict):
        self._data = data
        self.output = data.get("output")
        self.data = data.get("output")  # Backwards compatibility
        self._messages = data.get("messages", [])
        self._usage = data.get("usage", {})
    
    def usage(self) -> "MockUsage":
        return MockUsage(self._usage)
    
    def all_messages_json(self) -> str:
        return json.dumps(self._messages)


class MockUsage:
    """Mock usage for replay."""
    
    def __init__(self, data: dict):
        self.input_tokens = data.get("input_tokens", 0)
        self.output_tokens = data.get("output_tokens", 0)
        self.requests = data.get("requests", 1)


def _serialize_result(result: Any) -> dict:
    """Serialize PydanticAI result to dict for fixture storage."""
    if result is None:
        return {}
    
    data = {}
    
    # Output
    if hasattr(result, "output"):
        output = result.output
        data["output"] = str(output) if output is not None else None
    elif hasattr(result, "data"):
        data["output"] = str(result.data) if result.data is not None else None
    
    # Messages
    if hasattr(result, "all_messages_json"):
        try:
            data["messages"] = json.loads(result.all_messages_json())
        except Exception:
            pass
    
    # Usage
    usage = None
    if callable(getattr(result, "usage", None)):
        try:
            usage = result.usage()
        except Exception:
            pass
    elif hasattr(result, "usage"):
        usage = result.usage
    
    if usage:
        data["usage"] = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "requests": getattr(usage, "requests", 1),
        }
    
    return data


class WrappedAgent:
    """Wrapper around a PydanticAI agent that records/replays runs.
    
    Supports fluent interface for configuration:
        >>> wrapped = wrap_agent(agent, ledger).with_name("my-agent")
        >>> wrapped = wrap_agent(agent, ledger).with_replay("run_abc123")
    """

    def __init__(
        self,
        agent: Any,
        ledger: WorkLedger,
        run_name: str | None = None,
        replay_from: str | None = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            agent: The PydanticAI agent to wrap
            ledger: WorkLedger instance for recording
            run_name: Custom name for runs (defaults to agent name)
            replay_from: Run ID to replay from (no API calls made)
        """
        self._agent = agent
        self._ledger = ledger
        self._run_name = run_name or getattr(agent, "name", "pydantic-agent")
        self._replay_from = replay_from
        self._replay_run: Run | None = None
        
        if replay_from:
            self._replay_run = ledger.get_run(replay_from)
            if self._replay_run is None:
                raise ReplayError(f"Run '{replay_from}' not found for replay")

    def with_name(self, name: str) -> "WrappedAgent":
        """Set a custom name for recorded runs.
        
        Args:
            name: Human-readable name for runs
            
        Returns:
            Self for method chaining
            
        Example:
            >>> wrapped = wrap_agent(agent, ledger).with_name("my-custom-agent")
        """
        self._run_name = name
        return self

    def with_replay(self, run_id: str) -> "WrappedAgent":
        """Enable replay mode from a previously recorded run.
        
        Args:
            run_id: ID of the run to replay from
            
        Returns:
            Self for method chaining
            
        Raises:
            ReplayError: If the run is not found
            
        Example:
            >>> wrapped = wrap_agent(agent, ledger).with_replay("run_abc123")
        """
        self._replay_from = run_id
        self._replay_run = self._ledger.get_run(run_id)
        if self._replay_run is None:
            raise ReplayError(f"Run '{run_id}' not found for replay")
        return self

    def run_sync(self, prompt: str, **kwargs: Any) -> Any:
        """Run the agent synchronously and record/replay the run."""
        if self._replay_run:
            return self._replay_call(prompt, kwargs)
        return self._record_call_sync(prompt, kwargs)

    async def run(self, prompt: str, **kwargs: Any) -> Any:
        """Run the agent asynchronously and record/replay the run."""
        if self._replay_run:
            return self._replay_call(prompt, kwargs)
        return await self._record_call_async(prompt, kwargs)

    def _replay_call(self, prompt: str, kwargs: dict) -> Any:
        """Return saved result instead of running agent."""
        assert self._replay_run is not None
        
        # Find step with fixture
        steps_with_fixtures = [
            s for s in self._replay_run.steps
            if s.annotations.get("fixture")
        ]
        
        if not steps_with_fixtures:
            # Fallback to run outputs
            return MockAgentResult({
                "output": self._replay_run.outputs.get("result"),
            })
        
        step = steps_with_fixtures[0]
        fixture = step.annotations["fixture"]
        
        return MockAgentResult(fixture.get("response", {}))

    def _record_call_sync(self, prompt: str, kwargs: dict) -> Any:
        """Execute the agent and record the run."""
        run = Run(name=self._run_name)
        run.started_at = datetime.now(timezone.utc)
        run.status = RunStatus.RUNNING
        run.inputs = {"prompt": prompt, **{k: str(v) for k, v in kwargs.items()}}

        result = None
        error = None

        try:
            result = self._agent.run_sync(prompt, **kwargs)
            run.status = RunStatus.SUCCESS

            # Record output
            if hasattr(result, "output"):
                run.outputs = {"result": str(result.output)}
            elif hasattr(result, "data"):
                run.outputs = {"result": str(result.data)}

            # Record steps and fixture
            self._record_steps_from_result(run, result, prompt, kwargs)

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
            run.metrics = run.aggregate_metrics()
            self._ledger._save_run(run)

        if error:
            raise error

        return result

    async def _record_call_async(self, prompt: str, kwargs: dict) -> Any:
        """Execute the agent asynchronously and record the run."""
        run = Run(name=self._run_name)
        run.started_at = datetime.now(timezone.utc)
        run.status = RunStatus.RUNNING
        run.inputs = {"prompt": prompt, **{k: str(v) for k, v in kwargs.items()}}

        result = None
        error = None

        try:
            result = await self._agent.run(prompt, **kwargs)
            run.status = RunStatus.SUCCESS

            if hasattr(result, "output"):
                run.outputs = {"result": str(result.output)}
            elif hasattr(result, "data"):
                run.outputs = {"result": str(result.data)}

            self._record_steps_from_result(run, result, prompt, kwargs)

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
            run.metrics = run.aggregate_metrics()
            self._ledger._save_run(run)

        if error:
            raise error

        return result

    def _record_steps_from_result(
        self, run: Run, result: Any, prompt: str, kwargs: dict
    ) -> None:
        """Extract and record steps from agent result."""
        # Record tool calls if available
        tool_calls = getattr(result, "_tool_calls", [])
        for tool_call in tool_calls:
            step = Step(
                name=getattr(tool_call, "tool_name", "tool"),
                kind=StepKind.TOOL,
            )
            step.started_at = datetime.now(timezone.utc)
            step.ended_at = datetime.now(timezone.utc)

            if hasattr(tool_call, "args"):
                args = tool_call.args
                step.inputs = args if isinstance(args, dict) else {"args": args}

            if hasattr(tool_call, "result"):
                step.outputs = {"result": tool_call.result}

            run.add_step(step)

        # Record model call with usage and fixture
        usage = None
        if callable(getattr(result, "usage", None)):
            try:
                usage = result.usage()
            except Exception:
                pass
        elif hasattr(result, "usage"):
            usage = result.usage

        step = Step(
            name="pydantic-ai.agent.run",
            kind=StepKind.MODEL,
        )
        step.started_at = run.started_at
        step.ended_at = datetime.now(timezone.utc)

        # Save fixture for replay
        step.annotations["fixture"] = {
            "type": "pydantic-ai.agent.run",
            "request": {
                "prompt": prompt,
                **{k: str(v) for k, v in kwargs.items()},
            },
            "response": _serialize_result(result),
        }

        if usage:
            prompt_tokens = (
                getattr(usage, "input_tokens", 0)
                or getattr(usage, "request_tokens", 0)
                or getattr(usage, "prompt_tokens", 0)
            )
            completion_tokens = (
                getattr(usage, "output_tokens", 0)
                or getattr(usage, "response_tokens", 0)
                or getattr(usage, "completion_tokens", 0)
            )

            step.metrics = Metrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        # Record messages if available
        if hasattr(result, "all_messages_json"):
            try:
                step.outputs = {"messages": result.all_messages_json()}
            except Exception:
                pass

        run.add_step(step)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the wrapped agent."""
        return getattr(self._agent, name)


def wrap_agent(
    agent: Any,
    ledger: WorkLedger,
    run_name: str | None = None,
    replay_from: str | None = None,
) -> WrappedAgent:
    """Wrap a PydanticAI agent to record/replay runs.

    Args:
        agent: The PydanticAI agent to wrap
        ledger: WorkLedger instance for recording
        run_name: Custom name for runs (defaults to agent name)
        replay_from: Run ID to replay from (no API calls made)

    Returns:
        Wrapped agent that records or replays runs

    Example:
        >>> # Record mode (default)
        >>> wrapped = wrap_agent(agent, ledger)
        >>> result = wrapped.run_sync("Hello!")
        >>>
        >>> # Replay mode
        >>> wrapped = wrap_agent(agent, ledger, replay_from="run_abc123")
        >>> result = wrapped.run_sync("Hello!")  # No API call
    """
    return WrappedAgent(agent, ledger, run_name, replay_from)
