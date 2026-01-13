"""Tests for PydanticAI integration."""

import pytest

from work_ledger import WorkLedger
from work_ledger.core.models import StepKind, RunStatus


# Mock PydanticAI classes for testing without the actual dependency
class MockToolCall:
    """Mock tool call result."""
    def __init__(self, tool_name: str, args: dict, result: str):
        self.tool_name = tool_name
        self.args = args
        self.result = result


class MockMessage:
    """Mock message from model."""
    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role


class MockUsage:
    """Mock token usage."""
    def __init__(self, prompt_tokens: int = 100, completion_tokens: int = 50):
        self.request_tokens = prompt_tokens
        self.response_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens


class MockRunResult:
    """Mock result from agent.run()."""
    def __init__(
        self,
        data: str,
        messages: list = None,
        tool_calls: list = None,
        usage: MockUsage = None,
    ):
        self.data = data
        self.all_messages = messages or [MockMessage(data)]
        self._tool_calls = tool_calls or []
        self.usage = usage or MockUsage()
    
    def all_messages_json(self):
        return [{"role": m.role, "content": m.content} for m in self.all_messages]


class MockAgent:
    """Mock PydanticAI Agent for testing."""
    def __init__(self, name: str = "test-agent"):
        self.name = name
        self._tools = []
        self._run_result = None
    
    def set_result(self, result: MockRunResult):
        """Set the result that run() will return."""
        self._run_result = result
    
    def run_sync(self, prompt: str, **kwargs):
        """Synchronous run."""
        if self._run_result:
            return self._run_result
        return MockRunResult(data=f"Response to: {prompt}")
    
    async def run(self, prompt: str, **kwargs):
        """Async run."""
        return self.run_sync(prompt, **kwargs)


class TestPydanticAIIntegration:
    """Tests for PydanticAI wrapper."""

    def test_wrap_agent_basic(self):
        """Wrapped agent records runs."""
        from work_ledger.integrations.pydantic_ai import wrap_agent
        
        ledger = WorkLedger(store=":memory:")
        agent = MockAgent(name="weather-agent")
        
        wrapped = wrap_agent(agent, ledger)
        result = wrapped.run_sync("What's the weather?")
        
        # Should return the agent's result
        assert result.data == "Response to: What's the weather?"
        
        # Should have recorded a run
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].name == "weather-agent"
        assert runs[0].status == RunStatus.SUCCESS

    def test_records_input_output(self):
        """Wrapper records input prompt and output."""
        from work_ledger.integrations.pydantic_ai import wrap_agent
        
        ledger = WorkLedger(store=":memory:")
        agent = MockAgent()
        agent.set_result(MockRunResult(data="It's sunny!"))
        
        wrapped = wrap_agent(agent, ledger)
        wrapped.run_sync("What's the weather in Paris?")
        
        run = ledger.list_runs()[0]
        assert run.inputs["prompt"] == "What's the weather in Paris?"
        assert run.outputs["result"] == "It's sunny!"

    def test_records_model_step(self):
        """Wrapper records model call as a step."""
        from work_ledger.integrations.pydantic_ai import wrap_agent
        
        ledger = WorkLedger(store=":memory:")
        agent = MockAgent()
        agent.set_result(MockRunResult(
            data="The answer is 42",
            usage=MockUsage(prompt_tokens=100, completion_tokens=50),
        ))
        
        wrapped = wrap_agent(agent, ledger)
        wrapped.run_sync("What is the meaning of life?")
        
        run = ledger.list_runs()[0]
        
        # Should have at least one model step
        model_steps = run.get_steps_by_kind(StepKind.MODEL)
        assert len(model_steps) >= 1
        
        # Should record metrics
        assert run.metrics.prompt_tokens == 100
        assert run.metrics.completion_tokens == 50

    def test_records_tool_calls(self):
        """Wrapper records tool calls as steps."""
        from work_ledger.integrations.pydantic_ai import wrap_agent
        
        ledger = WorkLedger(store=":memory:")
        agent = MockAgent()
        agent.set_result(MockRunResult(
            data="Weather fetched",
            tool_calls=[
                MockToolCall("get_weather", {"city": "Paris"}, "sunny, 22C"),
            ],
        ))
        
        wrapped = wrap_agent(agent, ledger)
        wrapped.run_sync("Get Paris weather")
        
        run = ledger.list_runs()[0]
        
        # Should have tool step
        tool_steps = run.get_steps_by_kind(StepKind.TOOL)
        assert len(tool_steps) == 1
        assert tool_steps[0].name == "get_weather"
        assert tool_steps[0].inputs == {"city": "Paris"}
        assert tool_steps[0].outputs == {"result": "sunny, 22C"}

    def test_handles_exceptions(self):
        """Wrapper handles agent exceptions gracefully."""
        from work_ledger.integrations.pydantic_ai import wrap_agent
        
        ledger = WorkLedger(store=":memory:")
        
        class FailingAgent(MockAgent):
            def run_sync(self, prompt, **kwargs):
                raise ValueError("API Error")
        
        agent = FailingAgent()
        wrapped = wrap_agent(agent, ledger)
        
        with pytest.raises(ValueError):
            wrapped.run_sync("This will fail")
        
        # Should still record the failed run
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.FAILED
        assert "error" in runs[0].annotations

    def test_custom_run_name(self):
        """Can specify custom run name."""
        from work_ledger.integrations.pydantic_ai import wrap_agent
        
        ledger = WorkLedger(store=":memory:")
        agent = MockAgent(name="default-name")
        
        wrapped = wrap_agent(agent, ledger, run_name="custom-run")
        wrapped.run_sync("test")
        
        run = ledger.list_runs()[0]
        assert run.name == "custom-run"

    def test_pass_through_kwargs(self):
        """Wrapper passes through kwargs to agent."""
        from work_ledger.integrations.pydantic_ai import wrap_agent
        
        received_kwargs = {}
        
        class KwargsAgent(MockAgent):
            def run_sync(self, prompt, **kwargs):
                nonlocal received_kwargs
                received_kwargs = kwargs
                return MockRunResult(data="ok")
        
        ledger = WorkLedger(store=":memory:")
        agent = KwargsAgent()
        wrapped = wrap_agent(agent, ledger)
        
        wrapped.run_sync("test", temperature=0.5, max_tokens=100)
        
        assert received_kwargs["temperature"] == 0.5
        assert received_kwargs["max_tokens"] == 100


class TestPydanticAIAsyncIntegration:
    """Tests for async PydanticAI wrapper."""

    @pytest.mark.asyncio
    async def test_async_run(self):
        """Async run is recorded."""
        from work_ledger.integrations.pydantic_ai import wrap_agent
        
        ledger = WorkLedger(store=":memory:")
        agent = MockAgent()
        
        wrapped = wrap_agent(agent, ledger)
        result = await wrapped.run("Async test")
        
        assert result.data == "Response to: Async test"
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.SUCCESS
