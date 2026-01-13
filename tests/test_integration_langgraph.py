"""Tests for LangGraph integration."""

import pytest
from typing import TypedDict, Any

from work_ledger import WorkLedger
from work_ledger.core.models import StepKind, RunStatus


# --- Mock LangGraph classes for testing ---

class MockState(TypedDict):
    """Mock state for graph."""
    messages: list
    result: str


class MockStreamEvent:
    """Mock stream event from LangGraph."""
    def __init__(self, event: str, name: str, data: dict):
        self.event = event
        self.name = name
        self.data = data


class MockCompiledGraph:
    """Mock compiled LangGraph for testing."""
    
    def __init__(self, name: str = "test-graph"):
        self.name = name
        self._nodes = {}
        self._result = None
        self._stream_events = []
    
    def set_result(self, result: dict):
        """Set the result that invoke() will return."""
        self._result = result
    
    def set_stream_events(self, events: list):
        """Set events for stream()."""
        self._stream_events = events
    
    def invoke(self, state: dict, config: dict = None) -> dict:
        """Mock invoke."""
        if self._result:
            return self._result
        # Default: return state with result
        return {**state, "result": "completed"}
    
    async def ainvoke(self, state: dict, config: dict = None) -> dict:
        """Mock async invoke."""
        return self.invoke(state, config)
    
    def stream(self, state: dict, config: dict = None):
        """Mock stream - yields events."""
        for event in self._stream_events:
            yield event
        yield {"result": "streamed"}
    
    async def astream_events(self, state: dict, config: dict = None):
        """Mock async stream events."""
        for event in self._stream_events:
            yield event


class TestLangGraphIntegration:
    """Tests for LangGraph wrapper."""

    def test_wrap_graph_basic(self):
        """Wrapped graph records runs."""
        from work_ledger.integrations.langgraph import wrap_graph
        
        ledger = WorkLedger(store=":memory:")
        graph = MockCompiledGraph(name="my-graph")
        
        wrapped = wrap_graph(graph, ledger)
        result = wrapped.invoke({"messages": ["hello"]})
        
        # Should return the graph's result
        assert result["result"] == "completed"
        
        # Should have recorded a run
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].name == "my-graph"
        assert runs[0].status == RunStatus.SUCCESS

    def test_records_input_output(self):
        """Wrapper records input state and output state."""
        from work_ledger.integrations.langgraph import wrap_graph
        
        ledger = WorkLedger(store=":memory:")
        graph = MockCompiledGraph()
        graph.set_result({"messages": ["hello", "world"], "result": "done"})
        
        wrapped = wrap_graph(graph, ledger)
        wrapped.invoke({"messages": ["hello"]})
        
        run = ledger.list_runs()[0]
        assert run.inputs["messages"] == ["hello"]
        assert run.outputs["result"] == "done"

    def test_records_node_steps_from_stream(self):
        """Wrapper records node executions as steps when streaming."""
        from work_ledger.integrations.langgraph import wrap_graph
        
        ledger = WorkLedger(store=":memory:")
        graph = MockCompiledGraph()
        graph.set_stream_events([
            {"node_1": {"output": "step 1 done"}},
            {"node_2": {"output": "step 2 done"}},
        ])
        
        wrapped = wrap_graph(graph, ledger, record_stream=True)
        
        # Consume the stream
        results = list(wrapped.stream({"messages": []}))
        
        run = ledger.list_runs()[0]
        
        # Should have steps for each node
        assert len(run.steps) >= 2

    def test_handles_exceptions(self):
        """Wrapper handles graph exceptions gracefully."""
        from work_ledger.integrations.langgraph import wrap_graph
        
        ledger = WorkLedger(store=":memory:")
        
        class FailingGraph(MockCompiledGraph):
            def invoke(self, state, config=None):
                raise ValueError("Node execution failed")
        
        graph = FailingGraph()
        wrapped = wrap_graph(graph, ledger)
        
        with pytest.raises(ValueError):
            wrapped.invoke({"messages": []})
        
        # Should still record the failed run
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.FAILED
        assert "error" in runs[0].annotations

    def test_custom_run_name(self):
        """Can specify custom run name."""
        from work_ledger.integrations.langgraph import wrap_graph
        
        ledger = WorkLedger(store=":memory:")
        graph = MockCompiledGraph(name="default-name")
        
        wrapped = wrap_graph(graph, ledger, run_name="custom-workflow")
        wrapped.invoke({})
        
        run = ledger.list_runs()[0]
        assert run.name == "custom-workflow"

    def test_pass_through_config(self):
        """Wrapper passes through config to graph."""
        from work_ledger.integrations.langgraph import wrap_graph
        
        received_config = {}
        
        class ConfigGraph(MockCompiledGraph):
            def invoke(self, state, config=None):
                nonlocal received_config
                received_config = config or {}
                return {"result": "ok"}
        
        ledger = WorkLedger(store=":memory:")
        graph = ConfigGraph()
        wrapped = wrap_graph(graph, ledger)
        
        wrapped.invoke({}, config={"recursion_limit": 50})
        
        assert received_config.get("recursion_limit") == 50

    def test_multiple_invocations(self):
        """Multiple invocations create separate runs."""
        from work_ledger.integrations.langgraph import wrap_graph
        
        ledger = WorkLedger(store=":memory:")
        graph = MockCompiledGraph()
        wrapped = wrap_graph(graph, ledger)
        
        wrapped.invoke({"query": "first"})
        wrapped.invoke({"query": "second"})
        wrapped.invoke({"query": "third"})
        
        runs = ledger.list_runs()
        assert len(runs) == 3


class TestLangGraphAsyncIntegration:
    """Tests for async LangGraph wrapper."""

    @pytest.mark.asyncio
    async def test_async_invoke(self):
        """Async invoke is recorded."""
        from work_ledger.integrations.langgraph import wrap_graph
        
        ledger = WorkLedger(store=":memory:")
        graph = MockCompiledGraph()
        
        wrapped = wrap_graph(graph, ledger)
        result = await wrapped.ainvoke({"messages": ["async test"]})
        
        assert result["result"] == "completed"
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_async_stream_events(self):
        """Async stream events are recorded."""
        from work_ledger.integrations.langgraph import wrap_graph
        
        ledger = WorkLedger(store=":memory:")
        graph = MockCompiledGraph()
        graph.set_stream_events([
            MockStreamEvent("on_chain_start", "node_1", {}),
            MockStreamEvent("on_chain_end", "node_1", {"output": "done"}),
        ])
        
        wrapped = wrap_graph(graph, ledger, record_stream=True)
        
        events = []
        async for event in wrapped.astream_events({"messages": []}):
            events.append(event)
        
        runs = ledger.list_runs()
        assert len(runs) == 1
