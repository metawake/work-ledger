"""Tests for Anthropic SDK integration."""

import pytest
from typing import Any

from work_ledger import WorkLedger
from work_ledger.core.models import StepKind, RunStatus


# --- Mock Anthropic classes ---

class MockContentBlock:
    """Mock content block."""
    def __init__(self, text: str, type: str = "text"):
        self.text = text
        self.type = type


class MockToolUseBlock:
    """Mock tool use block."""
    def __init__(self, id: str, name: str, input: dict):
        self.id = id
        self.name = name
        self.input = input
        self.type = "tool_use"


class MockUsage:
    """Mock token usage."""
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockMessage:
    """Mock Anthropic Message response."""
    def __init__(
        self,
        content: list = None,
        usage: MockUsage = None,
        model: str = "claude-3-opus",
        stop_reason: str = "end_turn"
    ):
        self.content = content or [MockContentBlock("Default response")]
        self.usage = usage or MockUsage()
        self.model = model
        self.stop_reason = stop_reason
        self.id = "msg_123"
        self.role = "assistant"


class MockMessages:
    """Mock messages namespace."""
    
    def __init__(self):
        self._result = None
    
    def set_result(self, result: MockMessage):
        self._result = result
    
    def create(
        self,
        messages: list,
        model: str = "claude-3-opus",
        max_tokens: int = 1024,
        **kwargs
    ) -> MockMessage:
        if self._result:
            return self._result
        content = f"Response to: {messages[-1].get('content', str(messages[-1]))}"
        return MockMessage(content=[MockContentBlock(content)])
    
    async def acreate(self, **kwargs) -> MockMessage:
        return self.create(**kwargs)


class MockAnthropic:
    """Mock Anthropic client."""
    def __init__(self):
        self.messages = MockMessages()


class TestAnthropicIntegration:
    """Tests for Anthropic SDK wrapper."""

    def test_wrap_anthropic_basic(self):
        """Wrapped client records runs."""
        from work_ledger.integrations.anthropic import wrap_anthropic
        
        ledger = WorkLedger(store=":memory:")
        client = MockAnthropic()
        
        wrapped = wrap_anthropic(client, ledger)
        response = wrapped.messages.create(
            messages=[{"role": "user", "content": "Hello Claude"}],
            model="claude-3-opus",
            max_tokens=1024
        )
        
        assert len(response.content) > 0
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.SUCCESS

    def test_records_messages_and_response(self):
        """Wrapper records input messages and response."""
        from work_ledger.integrations.anthropic import wrap_anthropic
        
        ledger = WorkLedger(store=":memory:")
        client = MockAnthropic()
        client.messages.set_result(MockMessage(
            content=[MockContentBlock("Paris is the capital of France")]
        ))
        
        wrapped = wrap_anthropic(client, ledger)
        wrapped.messages.create(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            model="claude-3-opus",
            max_tokens=1024
        )
        
        run = ledger.list_runs()[0]
        assert "capital" in str(run.inputs).lower()
        assert "Paris" in run.outputs.get("content", "")

    def test_records_token_usage(self):
        """Wrapper records token usage."""
        from work_ledger.integrations.anthropic import wrap_anthropic
        
        ledger = WorkLedger(store=":memory:")
        client = MockAnthropic()
        client.messages.set_result(MockMessage(
            content=[MockContentBlock("Response")],
            usage=MockUsage(input_tokens=200, output_tokens=100)
        ))
        
        wrapped = wrap_anthropic(client, ledger)
        wrapped.messages.create(
            messages=[{"role": "user", "content": "test"}],
            model="claude-3-opus",
            max_tokens=1024
        )
        
        run = ledger.list_runs()[0]
        assert run.metrics.prompt_tokens == 200
        assert run.metrics.completion_tokens == 100
        assert run.metrics.total_tokens == 300

    def test_records_tool_use(self):
        """Wrapper records tool use blocks."""
        from work_ledger.integrations.anthropic import wrap_anthropic
        
        ledger = WorkLedger(store=":memory:")
        client = MockAnthropic()
        client.messages.set_result(MockMessage(
            content=[
                MockToolUseBlock("toolu_1", "get_weather", {"city": "Paris"}),
            ]
        ))
        
        wrapped = wrap_anthropic(client, ledger)
        wrapped.messages.create(
            messages=[{"role": "user", "content": "What's the weather?"}],
            model="claude-3-opus",
            max_tokens=1024
        )
        
        run = ledger.list_runs()[0]
        
        # Should have tool step
        tool_steps = run.get_steps_by_kind(StepKind.TOOL)
        assert len(tool_steps) == 1
        assert tool_steps[0].name == "get_weather"

    def test_records_model_info(self):
        """Wrapper records model information."""
        from work_ledger.integrations.anthropic import wrap_anthropic
        
        ledger = WorkLedger(store=":memory:")
        client = MockAnthropic()
        
        wrapped = wrap_anthropic(client, ledger)
        wrapped.messages.create(
            messages=[{"role": "user", "content": "test"}],
            model="claude-3-sonnet",
            max_tokens=1024
        )
        
        run = ledger.list_runs()[0]
        assert run.annotations.get("model") == "claude-3-sonnet"

    def test_handles_exceptions(self):
        """Wrapper handles API exceptions."""
        from work_ledger.integrations.anthropic import wrap_anthropic
        
        ledger = WorkLedger(store=":memory:")
        
        class FailingClient(MockAnthropic):
            def __init__(self):
                super().__init__()
                self.messages = FailingMessages()
        
        class FailingMessages(MockMessages):
            def create(self, **kwargs):
                raise Exception("API Error")
        
        client = FailingClient()
        wrapped = wrap_anthropic(client, ledger)
        
        with pytest.raises(Exception):
            wrapped.messages.create(
                messages=[{"role": "user", "content": "test"}],
                model="claude-3-opus",
                max_tokens=1024
            )
        
        runs = ledger.list_runs()
        assert runs[0].status == RunStatus.FAILED

    def test_custom_run_name(self):
        """Can specify custom run name."""
        from work_ledger.integrations.anthropic import wrap_anthropic
        
        ledger = WorkLedger(store=":memory:")
        client = MockAnthropic()
        
        wrapped = wrap_anthropic(client, ledger, run_name="my-claude-call")
        wrapped.messages.create(
            messages=[{"role": "user", "content": "test"}],
            model="claude-3-opus",
            max_tokens=1024
        )
        
        run = ledger.list_runs()[0]
        assert run.name == "my-claude-call"
