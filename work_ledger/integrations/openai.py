"""OpenAI SDK integration for Work Ledger.

Thin wrapper that records OpenAI API calls and supports replay.

Example:
    >>> from openai import OpenAI
    >>> from work_ledger import WorkLedger
    >>> from work_ledger.integrations.openai import wrap_openai
    >>>
    >>> client = OpenAI()
    >>> ledger = WorkLedger(store="./runs")
    >>>
    >>> # Record mode (default)
    >>> wrapped = wrap_openai(client, ledger)
    >>> response = wrapped.chat.completions.create(
    ...     messages=[{"role": "user", "content": "Hello!"}],
    ...     model="gpt-4"
    ... )
    >>>
    >>> # Replay mode (no API calls)
    >>> wrapped = wrap_openai(client, ledger, replay_from="run_abc123")
    >>> response = wrapped.chat.completions.create(...)  # Returns saved
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from work_ledger.core.ledger import WorkLedger
from work_ledger.core.models import Metrics, Run, RunStatus, Step, StepKind

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
    """Serialize OpenAI response to dict for fixture storage."""
    if result is None:
        return {}

    data = {}

    # Core fields
    try:
        if hasattr(result, "id"):
            data["id"] = _to_str(result.id, "")
        if hasattr(result, "model"):
            data["model"] = _to_str(result.model, "")
        if hasattr(result, "created"):
            data["created"] = _to_int(result.created, 0)
    except Exception:
        pass

    # Choices
    if hasattr(result, "choices") and result.choices:
        try:
            data["choices"] = []
            for choice in result.choices:
                choice_data = {
                    "index": _to_int(getattr(choice, "index", 0), 0),
                    "finish_reason": _to_str(getattr(choice, "finish_reason", None)),
                }

                if hasattr(choice, "message"):
                    msg = choice.message
                    content = getattr(msg, "content", None)

                    choice_data["message"] = {
                        "role": _to_str(getattr(msg, "role", "assistant"), "assistant"),
                        "content": _to_str(content) if content is not None else None,
                    }

                    # Tool calls - only if it's a real list
                    tool_calls = getattr(msg, "tool_calls", None)
                    if tool_calls and isinstance(tool_calls, list):
                        choice_data["message"]["tool_calls"] = [
                            {
                                "id": _to_str(getattr(tc, "id", "")),
                                "type": _to_str(getattr(tc, "type", "function")),
                                "function": {
                                    "name": _to_str(getattr(tc.function, "name", "")),
                                    "arguments": _to_str(
                                        getattr(tc.function, "arguments", "{}")
                                    ),
                                },
                            }
                            for tc in tool_calls
                        ]
                data["choices"].append(choice_data)
        except Exception:
            pass

    # Usage
    if hasattr(result, "usage") and result.usage:
        try:
            usage = result.usage
            data["usage"] = {
                "prompt_tokens": _to_int(getattr(usage, "prompt_tokens", 0)),
                "completion_tokens": _to_int(getattr(usage, "completion_tokens", 0)),
                "total_tokens": _to_int(getattr(usage, "total_tokens", 0)),
            }
        except Exception:
            pass

    return data


class MockMessage:
    """Mock message object for replay."""

    def __init__(self, data: dict):
        self.role = data.get("role", "assistant")
        self.content = data.get("content")
        self.tool_calls = None
        if "tool_calls" in data:
            self.tool_calls = [MockToolCall(tc) for tc in data["tool_calls"]]


class MockToolCall:
    """Mock tool call for replay."""

    def __init__(self, data: dict):
        self.id = data.get("id", "")
        self.type = data.get("type", "function")
        self.function = MockFunction(data.get("function", {}))


class MockFunction:
    """Mock function for replay."""

    def __init__(self, data: dict):
        self.name = data.get("name", "")
        self.arguments = data.get("arguments", "{}")


class MockChoice:
    """Mock choice for replay."""

    def __init__(self, data: dict):
        self.index = data.get("index", 0)
        self.finish_reason = data.get("finish_reason")
        self.message = MockMessage(data.get("message", {}))


class MockUsage:
    """Mock usage for replay."""

    def __init__(self, data: dict):
        self.prompt_tokens = data.get("prompt_tokens", 0)
        self.completion_tokens = data.get("completion_tokens", 0)
        self.total_tokens = data.get("total_tokens", 0)


class MockResponse:
    """Mock OpenAI response for replay."""

    def __init__(self, data: dict):
        self.id = data.get("id", "mock")
        self.model = data.get("model", "mock")
        self.created = data.get("created", 0)
        self.choices = [MockChoice(c) for c in data.get("choices", [])]
        self.usage = MockUsage(data.get("usage", {})) if "usage" in data else None


class WrappedCompletions:
    """Wrapper around chat.completions that records/replays runs."""

    def __init__(
        self,
        completions: Any,
        ledger: WorkLedger,
        run_name: str | None = None,
        replay_from: str | None = None,
    ) -> None:
        self._completions = completions
        self._ledger = ledger
        self._run_name = run_name or "openai-chat"
        self._replay_from = replay_from
        self._replay_run: Run | None = None
        self._replay_index = 0

        if replay_from:
            self._replay_run = ledger.get_run(replay_from)
            if self._replay_run is None:
                raise ReplayError(f"Run '{replay_from}' not found for replay")

    def create(self, messages: list, model: str = "gpt-4", **kwargs: Any) -> Any:
        """Create completion and record/replay the run."""
        # Replay mode
        if self._replay_run:
            return self._replay_call(messages, model, kwargs)

        # Record mode
        return self._record_call(messages, model, kwargs)

    def _record_call(self, messages: list, model: str, kwargs: dict) -> Any:
        """Make real API call and record fixture."""
        run = self._create_run(messages, model, kwargs)
        result = None
        error = None

        try:
            result = self._completions.create(messages=messages, model=model, **kwargs)
            run.status = RunStatus.SUCCESS
            self._record_response(run, result, messages, model, kwargs)
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

    def _replay_call(self, messages: list, model: str, kwargs: dict) -> Any:
        """Return saved fixture instead of making API call."""
        assert self._replay_run is not None

        # Find step with fixture at current index
        steps_with_fixtures = [
            s for s in self._replay_run.steps if s.annotations.get("fixture")
        ]

        if self._replay_index >= len(steps_with_fixtures):
            raise ReplayError(
                f"Replay diverged: expected {len(steps_with_fixtures)} API calls, "
                f"got call #{self._replay_index + 1}"
            )

        step = steps_with_fixtures[self._replay_index]
        fixture = step.annotations["fixture"]
        self._replay_index += 1

        # Return mock response
        return MockResponse(fixture.get("response", {}))

    async def acreate(
        self, messages: list, model: str = "gpt-4", **kwargs: Any
    ) -> Any:
        """Async create completion and record the run."""
        if self._replay_run:
            return self._replay_call(messages, model, kwargs)

        run = self._create_run(messages, model, kwargs)
        result = None
        error = None

        try:
            result = await self._completions.acreate(
                messages=messages, model=model, **kwargs
            )
            run.status = RunStatus.SUCCESS
            self._record_response(run, result, messages, model, kwargs)
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

    def _create_run(self, messages: list, model: str, kwargs: dict) -> Run:
        run = Run(name=self._run_name)
        run.started_at = datetime.now(timezone.utc)
        run.status = RunStatus.RUNNING
        run.inputs = {"messages": self._serialize_messages(messages)}
        run.annotations["model"] = model
        if kwargs:
            run.annotations["params"] = {
                k: v for k, v in kwargs.items() if k not in ("messages", "model")
            }
        return run

    def _serialize_messages(self, messages: list) -> list:
        result = []
        for msg in messages:
            if isinstance(msg, dict):
                result.append(msg)
            else:
                result.append(
                    {"role": getattr(msg, "role", "user"), "content": str(msg)}
                )
        return result

    def _record_response(
        self,
        run: Run,
        result: Any,
        messages: list,
        model: str,
        kwargs: dict,
    ) -> None:
        """Record response content, usage, tool calls, and fixture."""
        # Create step for this API call
        step = Step(
            name="openai.chat.completions.create",
            kind=StepKind.MODEL,
        )
        step.started_at = run.started_at
        step.ended_at = datetime.now(timezone.utc)

        # Save fixture for replay
        step.annotations["fixture"] = {
            "type": "openai.chat.completions.create",
            "request": {
                "model": model,
                "messages": self._serialize_messages(messages),
                **{k: v for k, v in kwargs.items() if self._is_serializable(v)},
            },
            "response": _serialize_response(result),
        }

        # Get the message from first choice
        if hasattr(result, "choices") and result.choices:
            message = result.choices[0].message
            content = getattr(message, "content", "") or ""
            step.outputs = {"content": content}
            run.outputs["content"] = content

            # Record tool calls as nested steps
            tool_calls = getattr(message, "tool_calls", []) or []
            for tool_call in tool_calls:
                func = getattr(tool_call, "function", tool_call)
                tool_step = Step(
                    name=getattr(func, "name", "tool"),
                    kind=StepKind.TOOL,
                )
                tool_step.started_at = datetime.now(timezone.utc)
                tool_step.ended_at = datetime.now(timezone.utc)

                args = getattr(func, "arguments", "{}")
                try:
                    tool_step.inputs = (
                        json.loads(args) if isinstance(args, str) else args
                    )
                except json.JSONDecodeError:
                    tool_step.inputs = {"raw": args}

                run.add_step(tool_step)

        # Record usage
        usage = getattr(result, "usage", None)
        if usage:
            step.metrics = Metrics(
                prompt_tokens=getattr(usage, "prompt_tokens", 0),
                completion_tokens=getattr(usage, "completion_tokens", 0),
                total_tokens=getattr(usage, "total_tokens", 0),
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
        return getattr(self._completions, name)


class WrappedChat:
    """Wrapper around chat namespace."""

    def __init__(
        self,
        chat: Any,
        ledger: WorkLedger,
        run_name: str | None = None,
        replay_from: str | None = None,
    ) -> None:
        self._chat = chat
        self._ledger = ledger
        self._run_name = run_name
        self._replay_from = replay_from
        self._completions: WrappedCompletions | None = None

    @property
    def completions(self) -> WrappedCompletions:
        if self._completions is None:
            self._completions = WrappedCompletions(
                self._chat.completions,
                self._ledger,
                self._run_name,
                self._replay_from,
            )
        return self._completions

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class WrappedOpenAI:
    """Wrapper around OpenAI client that records/replays runs.
    
    Supports fluent interface for configuration:
        >>> wrapped = wrap_openai(client, ledger).with_name("my-openai")
        >>> wrapped = wrap_openai(client, ledger).with_replay("run_abc123")
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
        self._chat: WrappedChat | None = None

    def with_name(self, name: str) -> "WrappedOpenAI":
        """Set a custom name for recorded runs.
        
        Args:
            name: Human-readable name for runs
            
        Returns:
            Self for method chaining
            
        Example:
            >>> wrapped = wrap_openai(client, ledger).with_name("gpt4-chat")
        """
        self._run_name = name
        # Reset chat wrapper so it picks up the new name
        self._chat = None
        return self

    def with_replay(self, run_id: str) -> "WrappedOpenAI":
        """Enable replay mode from a previously recorded run.
        
        Args:
            run_id: ID of the run to replay from
            
        Returns:
            Self for method chaining
            
        Raises:
            ReplayError: If the run is not found
            
        Example:
            >>> wrapped = wrap_openai(client, ledger).with_replay("run_abc123")
        """
        replay_run = self._ledger.get_run(run_id)
        if replay_run is None:
            raise ReplayError(f"Run '{run_id}' not found for replay")
        self._replay_from = run_id
        # Reset chat wrapper so it picks up replay mode
        self._chat = None
        return self

    @property
    def chat(self) -> WrappedChat:
        if self._chat is None:
            self._chat = WrappedChat(
                self._client.chat,
                self._ledger,
                self._run_name,
                self._replay_from,
            )
        return self._chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_openai(
    client: Any,
    ledger: WorkLedger,
    run_name: str | None = None,
    replay_from: str | None = None,
) -> WrappedOpenAI:
    """Wrap an OpenAI client to record/replay runs.

    Args:
        client: OpenAI client instance
        ledger: WorkLedger instance
        run_name: Custom name for runs
        replay_from: Run ID to replay from (no API calls made)

    Returns:
        Wrapped client that records or replays runs

    Example:
        >>> # Record mode (default)
        >>> wrapped = wrap_openai(client, ledger)
        >>> response = wrapped.chat.completions.create(...)
        >>>
        >>> # Replay mode
        >>> wrapped = wrap_openai(client, ledger, replay_from="run_abc123")
        >>> response = wrapped.chat.completions.create(...)  # No API call
    """
    return WrappedOpenAI(client, ledger, run_name, replay_from)
