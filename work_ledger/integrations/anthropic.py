"""Anthropic SDK integration for Work Ledger.

Thin wrapper that records Anthropic API calls and supports replay.

Example:
    >>> from anthropic import Anthropic
    >>> from work_ledger import WorkLedger
    >>> from work_ledger.integrations.anthropic import wrap_anthropic
    >>>
    >>> client = Anthropic()
    >>> ledger = WorkLedger(store="./runs")
    >>>
    >>> # Record mode (default)
    >>> wrapped = wrap_anthropic(client, ledger)
    >>> response = wrapped.messages.create(...)
    >>>
    >>> # Replay mode (no API calls)
    >>> wrapped = wrap_anthropic(client, ledger, replay_from="run_abc123")
    >>> response = wrapped.messages.create(...)  # Returns saved
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


def _to_str(val: Any, default: str = "") -> str:
    """Convert value to string, handling MagicMock."""
    if val is None:
        return default
    try:
        s = str(val)
        # MagicMock str() returns something like "<MagicMock ...>"
        if s.startswith("<") and "Mock" in s:
            return default
        return s
    except Exception:
        return default


def _to_int(val: Any, default: int = 0) -> int:
    """Convert value to int, handling MagicMock."""
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _serialize_response(result: Any) -> dict:
    """Serialize Anthropic response to dict for fixture storage."""
    if result is None:
        return {}

    data = {}

    # Core fields
    try:
        if hasattr(result, "id"):
            data["id"] = _to_str(result.id)
        if hasattr(result, "model"):
            data["model"] = _to_str(result.model)
        if hasattr(result, "type"):
            data["type"] = _to_str(result.type, "message")
        if hasattr(result, "role"):
            data["role"] = _to_str(result.role, "assistant")
        if hasattr(result, "stop_reason"):
            val = result.stop_reason
            data["stop_reason"] = _to_str(val) if val is not None else None
    except Exception:
        pass

    # Content blocks
    if hasattr(result, "content") and result.content:
        try:
            data["content"] = []
            for block in result.content:
                block_type = _to_str(getattr(block, "type", "text"), "text")
                block_data = {"type": block_type}

                if block_type == "text":
                    block_data["text"] = _to_str(getattr(block, "text", ""))
                elif block_type == "tool_use":
                    block_data["id"] = _to_str(getattr(block, "id", ""))
                    block_data["name"] = _to_str(getattr(block, "name", ""))
                    inp = getattr(block, "input", {})
                    block_data["input"] = inp if isinstance(inp, dict) else {}

                data["content"].append(block_data)
        except Exception:
            pass

    # Usage
    if hasattr(result, "usage") and result.usage:
        try:
            usage = result.usage
            data["usage"] = {
                "input_tokens": _to_int(getattr(usage, "input_tokens", 0)),
                "output_tokens": _to_int(getattr(usage, "output_tokens", 0)),
            }
        except Exception:
            pass

    return data


class MockContentBlock:
    """Mock content block for replay."""
    def __init__(self, data: dict):
        self.type = data.get("type", "text")
        if self.type == "text":
            self.text = data.get("text", "")
        elif self.type == "tool_use":
            self.id = data.get("id", "")
            self.name = data.get("name", "")
            self.input = data.get("input", {})


class MockUsage:
    """Mock usage for replay."""
    def __init__(self, data: dict):
        self.input_tokens = data.get("input_tokens", 0)
        self.output_tokens = data.get("output_tokens", 0)


class MockResponse:
    """Mock Anthropic response for replay."""
    def __init__(self, data: dict):
        self.id = data.get("id", "mock")
        self.model = data.get("model", "mock")
        self.type = data.get("type", "message")
        self.role = data.get("role", "assistant")
        self.stop_reason = data.get("stop_reason")
        self.content = [MockContentBlock(b) for b in data.get("content", [])]
        self.usage = MockUsage(data.get("usage", {})) if "usage" in data else None


class WrappedMessages:
    """Wrapper around messages that records/replays runs."""

    def __init__(
        self,
        messages: Any,
        ledger: WorkLedger,
        run_name: str | None = None,
        replay_from: str | None = None,
    ) -> None:
        self._messages = messages
        self._ledger = ledger
        self._run_name = run_name or "anthropic-chat"
        self._replay_from = replay_from
        self._replay_run: Run | None = None
        self._replay_index = 0
        
        if replay_from:
            self._replay_run = ledger.get_run(replay_from)
            if self._replay_run is None:
                raise ReplayError(f"Run '{replay_from}' not found for replay")

    def create(
        self,
        messages: list,
        model: str,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> Any:
        """Create message and record/replay the run."""
        if self._replay_run:
            return self._replay_call(messages, model, max_tokens, kwargs)
        return self._record_call(messages, model, max_tokens, kwargs)

    def _record_call(
        self,
        messages: list,
        model: str,
        max_tokens: int,
        kwargs: dict
    ) -> Any:
        """Make real API call and record fixture."""
        run = self._create_run(messages, model, max_tokens, kwargs)
        result = None
        error = None
        
        try:
            result = self._messages.create(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                **kwargs
            )
            run.status = RunStatus.SUCCESS
            self._record_response(run, result, messages, model, max_tokens, kwargs)
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

    def _replay_call(
        self,
        messages: list,
        model: str,
        max_tokens: int,
        kwargs: dict
    ) -> Any:
        """Return saved fixture instead of making API call."""
        assert self._replay_run is not None
        
        steps_with_fixtures = [
            s for s in self._replay_run.steps
            if s.annotations.get("fixture")
        ]
        
        if self._replay_index >= len(steps_with_fixtures):
            raise ReplayError(
                f"Replay diverged: expected {len(steps_with_fixtures)} API calls, "
                f"got call #{self._replay_index + 1}"
            )
        
        step = steps_with_fixtures[self._replay_index]
        fixture = step.annotations["fixture"]
        self._replay_index += 1
        
        return MockResponse(fixture.get("response", {}))

    async def acreate(
        self,
        messages: list,
        model: str,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> Any:
        """Async create message and record the run."""
        if self._replay_run:
            return self._replay_call(messages, model, max_tokens, kwargs)
        
        run = self._create_run(messages, model, max_tokens, kwargs)
        result = None
        error = None
        
        try:
            result = await self._messages.acreate(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                **kwargs
            )
            run.status = RunStatus.SUCCESS
            self._record_response(run, result, messages, model, max_tokens, kwargs)
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

    def _create_run(
        self, messages: list, model: str, max_tokens: int, kwargs: dict
    ) -> Run:
        run = Run(name=self._run_name)
        run.started_at = datetime.now(timezone.utc)
        run.status = RunStatus.RUNNING
        run.inputs = {"messages": self._serialize_messages(messages)}
        run.annotations["model"] = model
        run.annotations["max_tokens"] = max_tokens
        return run

    def _serialize_messages(self, messages: list) -> list:
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                result.append(msg)
            else:
                result.append({
                    "role": getattr(msg, "role", "user"),
                    "content": str(msg)
                })
        return result

    def _record_response(
        self, run: Run, result: Any,
        messages: list, model: str, max_tokens: int, kwargs: dict
    ) -> None:
        """Record response content, usage, tool use, and fixture."""
        # Create step for this API call
        step = Step(
            name="anthropic.messages.create",
            kind=StepKind.MODEL,
        )
        step.started_at = run.started_at
        step.ended_at = datetime.now(timezone.utc)
        
        # Save fixture for replay
        step.annotations["fixture"] = {
            "type": "anthropic.messages.create",
            "request": {
                "model": model,
                "max_tokens": max_tokens,
                "messages": self._serialize_messages(messages),
                **{k: v for k, v in kwargs.items() if self._is_serializable(v)},
            },
            "response": _serialize_response(result),
        }
        
        content_blocks = getattr(result, "content", [])
        
        # Extract text content
        text_parts = []
        for block in content_blocks:
            block_type = getattr(block, "type", "text")
            
            if block_type == "text":
                text_parts.append(getattr(block, "text", str(block)))
            elif block_type == "tool_use":
                tool_step = Step(
                    name=getattr(block, "name", "tool"),
                    kind=StepKind.TOOL,
                )
                tool_step.started_at = datetime.now(timezone.utc)
                tool_step.ended_at = datetime.now(timezone.utc)
                tool_step.inputs = getattr(block, "input", {})
                run.add_step(tool_step)
        
        content = " ".join(text_parts)
        step.outputs = {"content": content}
        run.outputs["content"] = content
        
        # Record usage
        usage = getattr(result, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            step.metrics = Metrics(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )
            run.metrics = step.metrics
        
        run.add_step(step)

    def _is_serializable(self, value: Any) -> bool:
        """Check if value is JSON serializable."""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._messages, name)


class WrappedAnthropic:
    """Wrapper around Anthropic client that records/replays runs.
    
    Supports fluent interface for configuration:
        >>> wrapped = wrap_anthropic(client, ledger).with_name("my-claude")
        >>> wrapped = wrap_anthropic(client, ledger).with_replay("run_abc123")
    """

    def __init__(
        self,
        client: Any,
        ledger: WorkLedger,
        run_name: str | None = None,
        replay_from: str | None = None,
    ) -> None:
        self._client = client
        self._ledger = ledger
        self._run_name = run_name
        self._replay_from = replay_from
        self._messages: WrappedMessages | None = None

    def with_name(self, name: str) -> "WrappedAnthropic":
        """Set a custom name for recorded runs.
        
        Args:
            name: Human-readable name for runs
            
        Returns:
            Self for method chaining
            
        Example:
            >>> wrapped = wrap_anthropic(client, ledger).with_name("claude-chat")
        """
        self._run_name = name
        # Reset messages wrapper so it picks up the new name
        self._messages = None
        return self

    def with_replay(self, run_id: str) -> "WrappedAnthropic":
        """Enable replay mode from a previously recorded run.
        
        Args:
            run_id: ID of the run to replay from
            
        Returns:
            Self for method chaining
            
        Raises:
            ReplayError: If the run is not found
            
        Example:
            >>> wrapped = wrap_anthropic(client, ledger).with_replay("run_abc123")
        """
        replay_run = self._ledger.get_run(run_id)
        if replay_run is None:
            raise ReplayError(f"Run '{run_id}' not found for replay")
        self._replay_from = run_id
        # Reset messages wrapper so it picks up replay mode
        self._messages = None
        return self

    @property
    def messages(self) -> WrappedMessages:
        if self._messages is None:
            self._messages = WrappedMessages(
                self._client.messages,
                self._ledger,
                self._run_name,
                self._replay_from,
            )
        return self._messages

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_anthropic(
    client: Any,
    ledger: WorkLedger,
    run_name: str | None = None,
    replay_from: str | None = None,
) -> WrappedAnthropic:
    """Wrap an Anthropic client to record/replay runs.

    Args:
        client: Anthropic client instance
        ledger: WorkLedger instance
        run_name: Custom name for runs
        replay_from: Run ID to replay from (no API calls made)

    Returns:
        Wrapped client that records or replays runs

    Example:
        >>> # Record mode (default)
        >>> wrapped = wrap_anthropic(client, ledger)
        >>> response = wrapped.messages.create(...)
        >>>
        >>> # Replay mode
        >>> wrapped = wrap_anthropic(client, ledger, replay_from="run_abc123")
        >>> response = wrapped.messages.create(...)  # No API call
    """
    return WrappedAnthropic(client, ledger, run_name, replay_from)
