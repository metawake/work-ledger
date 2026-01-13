"""Tests for LlamaIndex integration."""

import pytest
from typing import Any

from work_ledger import WorkLedger
from work_ledger.core.models import StepKind, RunStatus


# --- Mock LlamaIndex classes ---

class MockNodeWithScore:
    """Mock retrieved node."""
    def __init__(self, text: str, score: float, node_id: str):
        self.text = text
        self.score = score
        self.node_id = node_id
    
    def get_content(self):
        return self.text


class MockResponse:
    """Mock LlamaIndex Response."""
    def __init__(self, response: str, source_nodes: list = None, metadata: dict = None):
        self.response = response
        self.source_nodes = source_nodes or []
        self.metadata = metadata or {}
    
    def __str__(self):
        return self.response


class MockQueryEngine:
    """Mock LlamaIndex QueryEngine."""
    
    def __init__(self, name: str = "test-engine"):
        self.name = name
        self._result = None
    
    def set_result(self, result: MockResponse):
        self._result = result
    
    def query(self, query: str) -> MockResponse:
        if self._result:
            return self._result
        return MockResponse(
            response=f"Answer to: {query}",
            source_nodes=[
                MockNodeWithScore("Doc 1 content", 0.95, "doc1"),
                MockNodeWithScore("Doc 2 content", 0.87, "doc2"),
            ]
        )
    
    async def aquery(self, query: str) -> MockResponse:
        return self.query(query)


class MockChatEngine:
    """Mock LlamaIndex ChatEngine."""
    
    def __init__(self, name: str = "test-chat"):
        self.name = name
        self._result = None
    
    def set_result(self, result: MockResponse):
        self._result = result
    
    def chat(self, message: str) -> MockResponse:
        if self._result:
            return self._result
        return MockResponse(response=f"Chat response to: {message}")
    
    async def achat(self, message: str) -> MockResponse:
        return self.chat(message)


class TestLlamaIndexQueryEngineIntegration:
    """Tests for LlamaIndex QueryEngine wrapper."""

    def test_wrap_query_engine_basic(self):
        """Wrapped query engine records runs."""
        from work_ledger.integrations.llamaindex import wrap_query_engine
        
        ledger = WorkLedger(store=":memory:")
        engine = MockQueryEngine(name="my-rag")
        
        wrapped = wrap_query_engine(engine, ledger)
        result = wrapped.query("What is machine learning?")
        
        assert "machine learning" in str(result).lower() or "Answer" in str(result)
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].name == "my-rag"
        assert runs[0].status == RunStatus.SUCCESS

    def test_records_query_and_response(self):
        """Wrapper records query input and response."""
        from work_ledger.integrations.llamaindex import wrap_query_engine
        
        ledger = WorkLedger(store=":memory:")
        engine = MockQueryEngine()
        engine.set_result(MockResponse(
            response="Machine learning is a subset of AI",
            source_nodes=[MockNodeWithScore("ML is...", 0.9, "doc1")]
        ))
        
        wrapped = wrap_query_engine(engine, ledger)
        wrapped.query("What is ML?")
        
        run = ledger.list_runs()[0]
        assert run.inputs["query"] == "What is ML?"
        assert "Machine learning" in run.outputs["response"]

    def test_records_source_nodes(self):
        """Wrapper records retrieved source nodes."""
        from work_ledger.integrations.llamaindex import wrap_query_engine
        
        ledger = WorkLedger(store=":memory:")
        engine = MockQueryEngine()
        engine.set_result(MockResponse(
            response="Answer",
            source_nodes=[
                MockNodeWithScore("Document A", 0.95, "docA"),
                MockNodeWithScore("Document B", 0.88, "docB"),
            ]
        ))
        
        wrapped = wrap_query_engine(engine, ledger)
        wrapped.query("Query")
        
        run = ledger.list_runs()[0]
        
        # Should have retrieval step
        retrieval_steps = run.get_steps_by_kind(StepKind.RETRIEVAL)
        assert len(retrieval_steps) == 1
        assert len(retrieval_steps[0].outputs.get("nodes", [])) == 2

    def test_handles_exceptions(self):
        """Wrapper handles query exceptions."""
        from work_ledger.integrations.llamaindex import wrap_query_engine
        
        ledger = WorkLedger(store=":memory:")
        
        class FailingEngine(MockQueryEngine):
            def query(self, query):
                raise RuntimeError("Retrieval failed")
        
        engine = FailingEngine()
        wrapped = wrap_query_engine(engine, ledger)
        
        with pytest.raises(RuntimeError):
            wrapped.query("test")
        
        runs = ledger.list_runs()
        assert runs[0].status == RunStatus.FAILED

    def test_custom_run_name(self):
        """Can specify custom run name."""
        from work_ledger.integrations.llamaindex import wrap_query_engine
        
        ledger = WorkLedger(store=":memory:")
        engine = MockQueryEngine()
        
        wrapped = wrap_query_engine(engine, ledger, run_name="custom-rag")
        wrapped.query("test")
        
        run = ledger.list_runs()[0]
        assert run.name == "custom-rag"


class TestLlamaIndexChatEngineIntegration:
    """Tests for LlamaIndex ChatEngine wrapper."""

    def test_wrap_chat_engine_basic(self):
        """Wrapped chat engine records runs."""
        from work_ledger.integrations.llamaindex import wrap_chat_engine
        
        ledger = WorkLedger(store=":memory:")
        engine = MockChatEngine(name="my-chat")
        
        wrapped = wrap_chat_engine(engine, ledger)
        result = wrapped.chat("Hello!")
        
        assert "Hello" in str(result) or "Chat" in str(result)
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].name == "my-chat"


class TestLlamaIndexAsyncIntegration:
    """Tests for async LlamaIndex wrapper."""

    @pytest.mark.asyncio
    async def test_async_query(self):
        """Async query is recorded."""
        from work_ledger.integrations.llamaindex import wrap_query_engine
        
        ledger = WorkLedger(store=":memory:")
        engine = MockQueryEngine()
        
        wrapped = wrap_query_engine(engine, ledger)
        result = await wrapped.aquery("async test")
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.SUCCESS
