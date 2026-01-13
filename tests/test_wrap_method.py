"""Tests for WorkLedger.wrap() method and fluent interface."""

import pytest
from unittest.mock import MagicMock

from work_ledger import WorkLedger
from work_ledger.integrations.pydantic_ai import WrappedAgent
from work_ledger.integrations.langgraph import WrappedGraph
from work_ledger.integrations.crewai import WrappedCrew
from work_ledger.integrations.langchain import WrappedChain
from work_ledger.integrations.llamaindex import WrappedQueryEngine
from work_ledger.integrations.openai import WrappedOpenAI
from work_ledger.integrations.anthropic import WrappedAnthropic


class TestWorkLedgerWrapMethod:
    """Tests for the universal wrap() method."""

    def test_wrap_pydantic_ai_agent(self):
        """Test auto-detecting and wrapping a PydanticAI agent."""
        ledger = WorkLedger()
        
        # Create a mock PydanticAI agent
        mock_agent = MagicMock()
        mock_agent.__class__.__module__ = "pydantic_ai"
        mock_agent.__class__.__name__ = "Agent"
        mock_agent.run_sync = MagicMock()
        
        wrapped = ledger.wrap(mock_agent)
        
        assert isinstance(wrapped, WrappedAgent)

    def test_wrap_langgraph(self):
        """Test auto-detecting and wrapping a LangGraph."""
        ledger = WorkLedger()
        
        mock_graph = MagicMock()
        mock_graph.__class__.__module__ = "langgraph.graph.state"
        mock_graph.__class__.__name__ = "CompiledGraph"
        
        wrapped = ledger.wrap(mock_graph)
        
        assert isinstance(wrapped, WrappedGraph)

    def test_wrap_crewai(self):
        """Test auto-detecting and wrapping a CrewAI crew."""
        ledger = WorkLedger()
        
        mock_crew = MagicMock()
        mock_crew.__class__.__module__ = "crewai"
        mock_crew.__class__.__name__ = "Crew"
        mock_crew.kickoff = MagicMock()
        
        wrapped = ledger.wrap(mock_crew)
        
        assert isinstance(wrapped, WrappedCrew)

    def test_wrap_langchain_chain(self):
        """Test auto-detecting and wrapping a LangChain chain."""
        ledger = WorkLedger()
        
        mock_chain = MagicMock()
        mock_chain.__class__.__module__ = "langchain_core.runnables"
        mock_chain.__class__.__name__ = "RunnableSequence"
        mock_chain.invoke = MagicMock()
        
        wrapped = ledger.wrap(mock_chain)
        
        assert isinstance(wrapped, WrappedChain)

    def test_wrap_llamaindex_query_engine(self):
        """Test auto-detecting and wrapping a LlamaIndex query engine."""
        ledger = WorkLedger()
        
        mock_engine = MagicMock()
        mock_engine.__class__.__module__ = "llama_index.core.query_engine"
        mock_engine.__class__.__name__ = "RetrieverQueryEngine"
        mock_engine.query = MagicMock()
        
        wrapped = ledger.wrap(mock_engine)
        
        assert isinstance(wrapped, WrappedQueryEngine)

    def test_wrap_openai_client(self):
        """Test auto-detecting and wrapping an OpenAI client."""
        ledger = WorkLedger()
        
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "openai"
        mock_client.__class__.__name__ = "OpenAI"
        mock_client.chat = MagicMock()
        
        wrapped = ledger.wrap(mock_client)
        
        assert isinstance(wrapped, WrappedOpenAI)

    def test_wrap_anthropic_client(self):
        """Test auto-detecting and wrapping an Anthropic client."""
        ledger = WorkLedger()
        
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "anthropic"
        mock_client.__class__.__name__ = "Anthropic"
        mock_client.messages = MagicMock()
        
        wrapped = ledger.wrap(mock_client)
        
        assert isinstance(wrapped, WrappedAnthropic)

    def test_wrap_unsupported_type_raises(self):
        """Test that wrapping an unsupported type raises TypeError."""
        ledger = WorkLedger()
        
        mock_obj = MagicMock()
        mock_obj.__class__.__module__ = "some.unknown.module"
        mock_obj.__class__.__name__ = "UnknownClass"
        
        with pytest.raises(TypeError) as exc_info:
            ledger.wrap(mock_obj)
        
        assert "Unsupported object type" in str(exc_info.value)

    def test_wrap_with_custom_name(self):
        """Test wrap() with custom run_name parameter."""
        ledger = WorkLedger()
        
        mock_agent = MagicMock()
        mock_agent.__class__.__module__ = "pydantic_ai"
        mock_agent.__class__.__name__ = "Agent"
        mock_agent.run_sync = MagicMock()
        
        wrapped = ledger.wrap(mock_agent, run_name="custom-agent")
        
        assert wrapped._run_name == "custom-agent"


class TestFluentInterface:
    """Tests for fluent interface methods on wrappers."""

    def test_pydantic_ai_with_name(self):
        """Test with_name() on PydanticAI wrapper."""
        ledger = WorkLedger()
        
        mock_agent = MagicMock()
        mock_agent.__class__.__module__ = "pydantic_ai"
        mock_agent.__class__.__name__ = "Agent"
        mock_agent.run_sync = MagicMock()
        
        wrapped = ledger.wrap(mock_agent)
        result = wrapped.with_name("fluent-agent")
        
        # Should return self for chaining
        assert result is wrapped
        assert wrapped._run_name == "fluent-agent"

    def test_langgraph_with_name(self):
        """Test with_name() on LangGraph wrapper."""
        ledger = WorkLedger()
        
        mock_graph = MagicMock()
        mock_graph.__class__.__module__ = "langgraph.graph.state"
        mock_graph.__class__.__name__ = "CompiledGraph"
        
        wrapped = ledger.wrap(mock_graph)
        result = wrapped.with_name("fluent-graph")
        
        assert result is wrapped
        assert wrapped._run_name == "fluent-graph"

    def test_langgraph_with_stream_recording(self):
        """Test with_stream_recording() on LangGraph wrapper."""
        ledger = WorkLedger()
        
        mock_graph = MagicMock()
        mock_graph.__class__.__module__ = "langgraph.graph.state"
        mock_graph.__class__.__name__ = "CompiledGraph"
        
        wrapped = ledger.wrap(mock_graph)
        assert wrapped._record_stream is False
        
        result = wrapped.with_stream_recording()
        
        assert result is wrapped
        assert wrapped._record_stream is True

    def test_crewai_with_name(self):
        """Test with_name() on CrewAI wrapper."""
        ledger = WorkLedger()
        
        mock_crew = MagicMock()
        mock_crew.__class__.__module__ = "crewai"
        mock_crew.__class__.__name__ = "Crew"
        mock_crew.kickoff = MagicMock()
        
        wrapped = ledger.wrap(mock_crew)
        result = wrapped.with_name("fluent-crew")
        
        assert result is wrapped
        assert wrapped._run_name == "fluent-crew"

    def test_openai_with_name(self):
        """Test with_name() on OpenAI wrapper."""
        ledger = WorkLedger()
        
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "openai"
        mock_client.__class__.__name__ = "OpenAI"
        mock_client.chat = MagicMock()
        
        wrapped = ledger.wrap(mock_client)
        result = wrapped.with_name("fluent-openai")
        
        assert result is wrapped
        assert wrapped._run_name == "fluent-openai"

    def test_anthropic_with_name(self):
        """Test with_name() on Anthropic wrapper."""
        ledger = WorkLedger()
        
        mock_client = MagicMock()
        mock_client.__class__.__module__ = "anthropic"
        mock_client.__class__.__name__ = "Anthropic"
        mock_client.messages = MagicMock()
        
        wrapped = ledger.wrap(mock_client)
        result = wrapped.with_name("fluent-anthropic")
        
        assert result is wrapped
        assert wrapped._run_name == "fluent-anthropic"

    def test_method_chaining(self):
        """Test that fluent methods can be chained."""
        ledger = WorkLedger()
        
        mock_graph = MagicMock()
        mock_graph.__class__.__module__ = "langgraph.graph.state"
        mock_graph.__class__.__name__ = "CompiledGraph"
        
        wrapped = (
            ledger.wrap(mock_graph)
            .with_name("chained-graph")
            .with_stream_recording()
        )
        
        assert wrapped._run_name == "chained-graph"
        assert wrapped._record_stream is True
