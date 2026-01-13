"""Tests for LangChain integration."""

import pytest
from typing import Any

from work_ledger import WorkLedger
from work_ledger.core.models import StepKind, RunStatus


# --- Mock LangChain classes for testing ---

class MockMessage:
    """Mock LangChain message."""
    def __init__(self, content: str, type: str = "ai"):
        self.content = content
        self.type = type


class MockToolCall:
    """Mock tool call in response."""
    def __init__(self, name: str, args: dict):
        self.name = name
        self.args = args


class MockAIMessage:
    """Mock AI message with optional tool calls."""
    def __init__(self, content: str, tool_calls: list = None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.type = "ai"


class MockRunnable:
    """Mock LangChain Runnable (chain/LCEL)."""
    
    def __init__(self, name: str = "test-chain"):
        self.name = name
        self._result = None
        self._steps = []
    
    def set_result(self, result: Any):
        """Set the result that invoke() will return."""
        self._result = result
    
    def invoke(self, input: dict, config: dict = None) -> Any:
        """Invoke the chain."""
        if self._result is not None:
            return self._result
        return {"output": f"Processed: {input}"}
    
    async def ainvoke(self, input: dict, config: dict = None) -> Any:
        """Async invoke."""
        return self.invoke(input, config)


class MockAgentExecutor:
    """Mock LangChain AgentExecutor."""
    
    def __init__(self, name: str = "test-agent"):
        self.name = name
        self._result = None
        self._intermediate_steps = []
    
    def set_result(self, result: dict, intermediate_steps: list = None):
        """Set result and intermediate steps."""
        self._result = result
        self._intermediate_steps = intermediate_steps or []
    
    def invoke(self, input: dict, config: dict = None) -> dict:
        """Invoke the agent."""
        result = self._result or {"output": "Agent response"}
        result["intermediate_steps"] = self._intermediate_steps
        return result
    
    async def ainvoke(self, input: dict, config: dict = None) -> dict:
        """Async invoke."""
        return self.invoke(input, config)


class MockAgentAction:
    """Mock agent action (tool call)."""
    def __init__(self, tool: str, tool_input: dict, log: str = ""):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log


class TestLangChainChainIntegration:
    """Tests for LangChain chain wrapper."""

    def test_wrap_chain_basic(self):
        """Wrapped chain records runs."""
        from work_ledger.integrations.langchain import wrap_chain
        
        ledger = WorkLedger(store=":memory:")
        chain = MockRunnable(name="my-chain")
        
        wrapped = wrap_chain(chain, ledger)
        result = wrapped.invoke({"question": "What is AI?"})
        
        assert "output" in result or "Processed" in str(result)
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].name == "my-chain"
        assert runs[0].status == RunStatus.SUCCESS

    def test_records_input_output(self):
        """Wrapper records input and output."""
        from work_ledger.integrations.langchain import wrap_chain
        
        ledger = WorkLedger(store=":memory:")
        chain = MockRunnable()
        chain.set_result({"answer": "AI is artificial intelligence"})
        
        wrapped = wrap_chain(chain, ledger)
        wrapped.invoke({"question": "What is AI?"})
        
        run = ledger.list_runs()[0]
        assert run.inputs["question"] == "What is AI?"
        assert run.outputs["answer"] == "AI is artificial intelligence"

    def test_handles_string_output(self):
        """Wrapper handles string output."""
        from work_ledger.integrations.langchain import wrap_chain
        
        ledger = WorkLedger(store=":memory:")
        chain = MockRunnable()
        chain.set_result("Simple string response")
        
        wrapped = wrap_chain(chain, ledger)
        result = wrapped.invoke({"input": "test"})
        
        assert result == "Simple string response"
        run = ledger.list_runs()[0]
        assert run.outputs["result"] == "Simple string response"

    def test_handles_exceptions(self):
        """Wrapper handles chain exceptions."""
        from work_ledger.integrations.langchain import wrap_chain
        
        ledger = WorkLedger(store=":memory:")
        
        class FailingChain(MockRunnable):
            def invoke(self, input, config=None):
                raise ValueError("Chain failed")
        
        chain = FailingChain()
        wrapped = wrap_chain(chain, ledger)
        
        with pytest.raises(ValueError):
            wrapped.invoke({"input": "test"})
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.FAILED
        assert "error" in runs[0].annotations

    def test_custom_run_name(self):
        """Can specify custom run name."""
        from work_ledger.integrations.langchain import wrap_chain
        
        ledger = WorkLedger(store=":memory:")
        chain = MockRunnable(name="default")
        
        wrapped = wrap_chain(chain, ledger, run_name="custom-chain")
        wrapped.invoke({})
        
        run = ledger.list_runs()[0]
        assert run.name == "custom-chain"


class TestLangChainAgentIntegration:
    """Tests for LangChain agent wrapper."""

    def test_wrap_agent_basic(self):
        """Wrapped agent records runs."""
        from work_ledger.integrations.langchain import wrap_agent
        
        ledger = WorkLedger(store=":memory:")
        agent = MockAgentExecutor(name="my-agent")
        
        wrapped = wrap_agent(agent, ledger)
        result = wrapped.invoke({"input": "Hello"})
        
        assert "output" in result
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].name == "my-agent"

    def test_records_intermediate_steps(self):
        """Wrapper records tool calls as steps."""
        from work_ledger.integrations.langchain import wrap_agent
        
        ledger = WorkLedger(store=":memory:")
        agent = MockAgentExecutor()
        
        # Simulate agent with tool calls
        action1 = MockAgentAction("search", {"query": "weather"})
        action2 = MockAgentAction("calculator", {"expression": "2+2"})
        agent.set_result(
            {"output": "The weather is sunny and 2+2=4"},
            intermediate_steps=[
                (action1, "Sunny, 22C"),
                (action2, "4"),
            ]
        )
        
        wrapped = wrap_agent(agent, ledger)
        wrapped.invoke({"input": "What's the weather and what's 2+2?"})
        
        run = ledger.list_runs()[0]
        
        # Should have steps for each tool call
        assert len(run.steps) == 2
        assert run.steps[0].name == "search"
        assert run.steps[0].kind == StepKind.TOOL
        assert run.steps[1].name == "calculator"


class TestLangChainAsyncIntegration:
    """Tests for async LangChain wrapper."""

    @pytest.mark.asyncio
    async def test_async_chain(self):
        """Async chain invoke is recorded."""
        from work_ledger.integrations.langchain import wrap_chain
        
        ledger = WorkLedger(store=":memory:")
        chain = MockRunnable()
        
        wrapped = wrap_chain(chain, ledger)
        result = await wrapped.ainvoke({"input": "async test"})
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_async_agent(self):
        """Async agent invoke is recorded."""
        from work_ledger.integrations.langchain import wrap_agent
        
        ledger = WorkLedger(store=":memory:")
        agent = MockAgentExecutor()
        
        wrapped = wrap_agent(agent, ledger)
        result = await wrapped.ainvoke({"input": "async test"})
        
        runs = ledger.list_runs()
        assert len(runs) == 1
