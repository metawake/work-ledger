"""Tests for OpenAI SDK integration."""

import pytest
from typing import Any

from work_ledger import WorkLedger
from work_ledger.core.models import StepKind, RunStatus


# --- Mock OpenAI classes ---

class MockMessage:
    """Mock chat message."""
    def __init__(self, role: str, content: str, tool_calls: list = None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []


class MockChoice:
    """Mock completion choice."""
    def __init__(self, message: MockMessage, finish_reason: str = "stop"):
        self.message = message
        self.finish_reason = finish_reason
        self.index = 0


class MockUsage:
    """Mock token usage."""
    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockToolCall:
    """Mock tool call."""
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = type("Function", (), {"name": name, "arguments": arguments})()


class MockChatCompletion:
    """Mock OpenAI ChatCompletion response."""
    def __init__(self, content: str, tool_calls: list = None, usage: MockUsage = None):
        message = MockMessage("assistant", content, tool_calls)
        self.choices = [MockChoice(message)]
        self.usage = usage or MockUsage()
        self.model = "gpt-4"
        self.id = "chatcmpl-123"


class MockChatCompletions:
    """Mock chat.completions namespace."""
    
    def __init__(self):
        self._result = None
    
    def set_result(self, result: MockChatCompletion):
        self._result = result
    
    def create(self, messages: list, model: str = "gpt-4", **kwargs) -> MockChatCompletion:
        if self._result:
            return self._result
        content = f"Response to: {messages[-1].get('content', messages[-1])}"
        return MockChatCompletion(content=content)
    
    async def acreate(self, messages: list, model: str = "gpt-4", **kwargs) -> MockChatCompletion:
        return self.create(messages, model, **kwargs)


class MockChat:
    """Mock chat namespace."""
    def __init__(self):
        self.completions = MockChatCompletions()


class MockOpenAI:
    """Mock OpenAI client."""
    def __init__(self):
        self.chat = MockChat()


class TestOpenAIIntegration:
    """Tests for OpenAI SDK wrapper."""

    def test_wrap_openai_basic(self):
        """Wrapped client records runs."""
        from work_ledger.integrations.openai import wrap_openai
        
        ledger = WorkLedger(store=":memory:")
        client = MockOpenAI()
        
        wrapped = wrap_openai(client, ledger)
        response = wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4"
        )
        
        assert response.choices[0].message.content
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.SUCCESS

    def test_records_messages_and_response(self):
        """Wrapper records input messages and response."""
        from work_ledger.integrations.openai import wrap_openai
        
        ledger = WorkLedger(store=":memory:")
        client = MockOpenAI()
        client.chat.completions.set_result(MockChatCompletion(
            content="Paris is the capital of France"
        ))
        
        wrapped = wrap_openai(client, ledger)
        wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            model="gpt-4"
        )
        
        run = ledger.list_runs()[0]
        assert "capital" in str(run.inputs).lower()
        assert "Paris" in run.outputs.get("content", "")

    def test_records_token_usage(self):
        """Wrapper records token usage."""
        from work_ledger.integrations.openai import wrap_openai
        
        ledger = WorkLedger(store=":memory:")
        client = MockOpenAI()
        client.chat.completions.set_result(MockChatCompletion(
            content="Response",
            usage=MockUsage(prompt_tokens=150, completion_tokens=75)
        ))
        
        wrapped = wrap_openai(client, ledger)
        wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4"
        )
        
        run = ledger.list_runs()[0]
        assert run.metrics.prompt_tokens == 150
        assert run.metrics.completion_tokens == 75
        assert run.metrics.total_tokens == 225

    def test_records_tool_calls(self):
        """Wrapper records tool calls."""
        from work_ledger.integrations.openai import wrap_openai
        
        ledger = WorkLedger(store=":memory:")
        client = MockOpenAI()
        client.chat.completions.set_result(MockChatCompletion(
            content="",
            tool_calls=[
                MockToolCall("call_1", "get_weather", '{"city": "Paris"}'),
            ]
        ))
        
        wrapped = wrap_openai(client, ledger)
        wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "What's the weather?"}],
            model="gpt-4"
        )
        
        run = ledger.list_runs()[0]
        
        # Should have tool step
        tool_steps = run.get_steps_by_kind(StepKind.TOOL)
        assert len(tool_steps) == 1
        assert tool_steps[0].name == "get_weather"

    def test_records_model_info(self):
        """Wrapper records model information."""
        from work_ledger.integrations.openai import wrap_openai
        
        ledger = WorkLedger(store=":memory:")
        client = MockOpenAI()
        
        wrapped = wrap_openai(client, ledger)
        wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4-turbo"
        )
        
        run = ledger.list_runs()[0]
        assert run.annotations.get("model") == "gpt-4-turbo"

    def test_handles_exceptions(self):
        """Wrapper handles API exceptions."""
        from work_ledger.integrations.openai import wrap_openai
        
        ledger = WorkLedger(store=":memory:")
        
        class FailingClient(MockOpenAI):
            def __init__(self):
                super().__init__()
                self.chat.completions = FailingCompletions()
        
        class FailingCompletions(MockChatCompletions):
            def create(self, **kwargs):
                raise Exception("API Error")
        
        client = FailingClient()
        wrapped = wrap_openai(client, ledger)
        
        with pytest.raises(Exception):
            wrapped.chat.completions.create(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4"
            )
        
        runs = ledger.list_runs()
        assert runs[0].status == RunStatus.FAILED

    def test_custom_run_name(self):
        """Can specify custom run name."""
        from work_ledger.integrations.openai import wrap_openai
        
        ledger = WorkLedger(store=":memory:")
        client = MockOpenAI()
        
        wrapped = wrap_openai(client, ledger, run_name="my-gpt-call")
        wrapped.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4"
        )
        
        run = ledger.list_runs()[0]
        assert run.name == "my-gpt-call"
