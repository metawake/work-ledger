"""Framework integrations for Work Ledger.

Thin adapters (~100-150 lines each) for popular agent frameworks.
One line to add recording - everything else stays the same.

Agent Frameworks:
    >>> from work_ledger import WorkLedger, wrap_agent, wrap_graph, wrap_crew, wrap_chain
    >>> ledger = WorkLedger(store="./runs")
    >>> 
    >>> wrapped = wrap_agent(pydantic_agent, ledger)     # PydanticAI
    >>> wrapped = wrap_graph(langgraph, ledger)          # LangGraph
    >>> wrapped = wrap_crew(crew, ledger)                # CrewAI
    >>> wrapped = wrap_chain(chain, ledger)              # LangChain

RAG Pipelines:
    >>> from work_ledger import wrap_query_engine
    >>> wrapped = wrap_query_engine(engine, ledger)      # LlamaIndex

Direct SDK:
    >>> from work_ledger import wrap_openai, wrap_anthropic
    >>> wrapped = wrap_openai(client, ledger)            # OpenAI
    >>> wrapped = wrap_anthropic(client, ledger)         # Anthropic
"""

from work_ledger.integrations.pydantic_ai import wrap_agent
from work_ledger.integrations.langgraph import wrap_graph
from work_ledger.integrations.crewai import wrap_crew
from work_ledger.integrations.langchain import wrap_chain, wrap_agent as wrap_lc_agent
from work_ledger.integrations.llamaindex import wrap_query_engine, wrap_chat_engine
from work_ledger.integrations.openai import wrap_openai
from work_ledger.integrations.anthropic import wrap_anthropic

__all__ = [
    # PydanticAI
    "wrap_agent",
    # LangGraph
    "wrap_graph",
    # CrewAI
    "wrap_crew",
    # LangChain
    "wrap_chain",
    "wrap_lc_agent",
    # LlamaIndex
    "wrap_query_engine",
    "wrap_chat_engine",
    # OpenAI
    "wrap_openai",
    # Anthropic
    "wrap_anthropic",
]
