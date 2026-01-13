"""Tests for replay functionality."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timezone

from work_ledger import WorkLedger
from work_ledger.core.models import Run, Step, StepKind, RunStatus, Metrics
from work_ledger.integrations.openai import (
    wrap_openai,
    ReplayError,
    MockResponse,
)
from work_ledger.integrations.anthropic import wrap_anthropic
from work_ledger.integrations.pydantic_ai import wrap_agent


class TestOpenAIReplay:
    """Tests for OpenAI replay functionality."""

    def test_record_creates_fixture(self):
        """Test that recording creates a fixture in step annotations."""
        ledger = WorkLedger(store=":memory:")
        
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-123"
        mock_response.model = "gpt-4"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "Hello!"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_client.chat.completions.create.return_value = mock_response
        
        wrapped = wrap_openai(mock_client, ledger)
        wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4"
        )
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        
        run = runs[0]
        steps_with_fixtures = [
            s for s in run.steps
            if s.annotations.get("fixture")
        ]
        
        assert len(steps_with_fixtures) == 1
        fixture = steps_with_fixtures[0].annotations["fixture"]
        assert fixture["type"] == "openai.chat.completions.create"
        assert "request" in fixture
        assert "response" in fixture
        assert fixture["response"]["id"] == "chatcmpl-123"

    def test_replay_returns_saved_response(self):
        """Test that replay returns saved response without API call."""
        ledger = WorkLedger(store=":memory:")
        
        # Create a run with fixture
        run = Run(name="test")
        run.status = RunStatus.SUCCESS
        run.started_at = datetime.now(timezone.utc)
        run.ended_at = datetime.now(timezone.utc)
        run.inputs = {"messages": [{"role": "user", "content": "Hi"}]}
        run.outputs = {"content": "Hello!"}
        
        step = Step(name="openai.chat.completions.create", kind=StepKind.MODEL)
        step.annotations["fixture"] = {
            "type": "openai.chat.completions.create",
            "request": {"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]},
            "response": {
                "id": "chatcmpl-123",
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from fixture!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        }
        run.add_step(step)
        ledger._save_run(run)
        
        # Replay
        mock_client = MagicMock()
        wrapped = wrap_openai(mock_client, ledger, replay_from=run.run_id)
        
        response = wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4"
        )
        
        # Verify no real API call was made
        mock_client.chat.completions.create.assert_not_called()
        
        # Verify response from fixture
        assert isinstance(response, MockResponse)
        assert response.choices[0].message.content == "Hello from fixture!"

    def test_replay_raises_on_nonexistent_run(self):
        """Test that replay raises error for nonexistent run."""
        ledger = WorkLedger(store=":memory:")
        mock_client = MagicMock()

        wrapped = wrap_openai(mock_client, ledger, replay_from="nonexistent")
        # Error is raised lazily when accessing completions
        with pytest.raises(ReplayError, match="not found"):
            _ = wrapped.chat.completions

    def test_replay_divergence_error(self):
        """Test that replay raises error when calls exceed fixtures."""
        ledger = WorkLedger(store=":memory:")
        
        # Create run with one fixture
        run = Run(name="test")
        run.status = RunStatus.SUCCESS
        run.started_at = datetime.now(timezone.utc)
        run.ended_at = datetime.now(timezone.utc)
        
        step = Step(name="call1", kind=StepKind.MODEL)
        step.annotations["fixture"] = {
            "type": "openai",
            "response": {"choices": [{"message": {"content": "Hi"}}]},
        }
        run.add_step(step)
        ledger._save_run(run)
        
        mock_client = MagicMock()
        wrapped = wrap_openai(mock_client, ledger, replay_from=run.run_id)
        
        # First call succeeds
        wrapped.chat.completions.create(messages=[], model="gpt-4")
        
        # Second call should fail (no more fixtures)
        with pytest.raises(ReplayError, match="diverged"):
            wrapped.chat.completions.create(messages=[], model="gpt-4")


class TestAnthropicReplay:
    """Tests for Anthropic replay functionality."""

    def test_replay_returns_saved_response(self):
        """Test that replay returns saved response."""
        ledger = WorkLedger(store=":memory:")
        
        # Create run with fixture
        run = Run(name="test")
        run.status = RunStatus.SUCCESS
        run.started_at = datetime.now(timezone.utc)
        run.ended_at = datetime.now(timezone.utc)
        
        step = Step(name="anthropic.messages.create", kind=StepKind.MODEL)
        step.annotations["fixture"] = {
            "type": "anthropic.messages.create",
            "response": {
                "id": "msg_123",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello from Anthropic!"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        }
        run.add_step(step)
        ledger._save_run(run)
        
        mock_client = MagicMock()
        wrapped = wrap_anthropic(mock_client, ledger, replay_from=run.run_id)
        
        response = wrapped.messages.create(
            messages=[{"role": "user", "content": "Hi"}],
            model="claude-3",
            max_tokens=100
        )
        
        mock_client.messages.create.assert_not_called()
        assert response.content[0].text == "Hello from Anthropic!"


class TestPydanticAIReplay:
    """Tests for PydanticAI replay functionality."""

    def test_replay_returns_saved_result(self):
        """Test that replay returns saved result."""
        ledger = WorkLedger(store=":memory:")
        
        # Create run with fixture
        run = Run(name="test")
        run.status = RunStatus.SUCCESS
        run.started_at = datetime.now(timezone.utc)
        run.ended_at = datetime.now(timezone.utc)
        run.outputs = {"result": "Hello!"}
        
        step = Step(name="pydantic-ai.agent.run", kind=StepKind.MODEL)
        step.annotations["fixture"] = {
            "type": "pydantic-ai.agent.run",
            "response": {
                "output": "Hello from fixture!",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        }
        run.add_step(step)
        ledger._save_run(run)
        
        mock_agent = MagicMock()
        wrapped = wrap_agent(mock_agent, ledger, replay_from=run.run_id)
        
        result = wrapped.run_sync("Hi")
        
        mock_agent.run_sync.assert_not_called()
        assert result.output == "Hello from fixture!"


class TestReplayRoundtrip:
    """Test recording and replaying in sequence."""

    def test_openai_record_then_replay(self):
        """Test full record → replay cycle."""
        ledger = WorkLedger(store=":memory:")
        
        # Record phase
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.id = "chatcmpl-abc"
        mock_response.model = "gpt-4"
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "The answer is 42"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30
        mock_client.chat.completions.create.return_value = mock_response
        
        wrapped = wrap_openai(mock_client, ledger)
        response1 = wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "What is the meaning of life?"}],
            model="gpt-4"
        )
        
        assert response1.choices[0].message.content == "The answer is 42"
        
        # Get recorded run
        runs = ledger.list_runs()
        assert len(runs) == 1
        run_id = runs[0].run_id
        
        # Replay phase - reset mock to ensure no calls
        mock_client.reset_mock()
        
        wrapped_replay = wrap_openai(mock_client, ledger, replay_from=run_id)
        response2 = wrapped_replay.chat.completions.create(
            messages=[{"role": "user", "content": "What is the meaning of life?"}],
            model="gpt-4"
        )
        
        # Verify no API call made during replay
        mock_client.chat.completions.create.assert_not_called()
        
        # Verify same response
        assert response2.choices[0].message.content == "The answer is 42"
        assert response2.usage.total_tokens == 30
