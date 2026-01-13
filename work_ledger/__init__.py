"""Work Ledger - Agent diagnostics for LLM workflows.

Record. Replay. Diff.

When your agent breaks after a change, Work Ledger shows you what's different:
- Record runs with full step/tool/model traces
- Replay without API calls using saved fixtures
- Diff to see exactly what changed

Framework Integrations:
    >>> from work_ledger import WorkLedger, wrap_agent, wrap_graph, wrap_crew
    >>>
    >>> ledger = WorkLedger(store="./runs")
    >>>
    >>> # PydanticAI
    >>> wrapped = wrap_agent(my_agent, ledger)
    >>> result = wrapped.run_sync("Hello!")
    >>>
    >>> # LangGraph
    >>> wrapped = wrap_graph(my_graph, ledger)
    >>> result = wrapped.invoke({"messages": [...]})
    >>>
    >>> # CrewAI
    >>> wrapped = wrap_crew(my_crew, ledger)
    >>> result = wrapped.kickoff(inputs={"topic": "AI"})

Manual Recording:
    >>> with ledger.run(name="process-request") as run:
    ...     run.record_input({"query": "test"})
    ...     with run.step(name="llm-call", kind="model") as step:
    ...         step.record_output({"response": "result"})
    ...     run.record_output({"result": "done"})

Testing:
    >>> from work_ledger.testing import replay, assert_output_matches
    >>>
    >>> @replay("fixtures/my_agent.json")
    >>> def test_my_agent():
    ...     result = agent.run("test")
    ...     assert_output_matches(result, expected_keys=["response"])
"""

from work_ledger.core.ledger import WorkLedger
from work_ledger.core.models import (
    CausalLink,
    Metrics,
    Run,
    RunStatus,
    Step,
    StepKind,
)
from work_ledger.core.store import (
    GCSStore,
    JSONLStore,
    MemoryStore,
    MongoDBStore,
    PostgresStore,
    RedisStore,
    RunStore,
    S3Store,
    SQLiteStore,
)
from work_ledger.integrations.anthropic import wrap_anthropic
from work_ledger.integrations.crewai import wrap_crew
from work_ledger.integrations.langchain import wrap_chain
from work_ledger.integrations.langgraph import wrap_graph
from work_ledger.integrations.llamaindex import wrap_query_engine
from work_ledger.integrations.openai import ReplayError, wrap_openai
from work_ledger.integrations.pydantic_ai import wrap_agent

__version__ = "0.1.0"

__all__ = [
    # Main API
    "WorkLedger",
    # Models
    "Run",
    "Step",
    "Metrics",
    "CausalLink",
    "RunStatus",
    "StepKind",
    # Storage
    "RunStore",
    "MemoryStore",
    "JSONLStore",
    "SQLiteStore",
    "PostgresStore",
    "RedisStore",
    "S3Store",
    "MongoDBStore",
    "GCSStore",
    # Integrations
    "wrap_agent",
    "wrap_graph",
    "wrap_crew",
    "wrap_chain",
    "wrap_query_engine",
    "wrap_openai",
    "wrap_anthropic",
    # Exceptions
    "ReplayError",
]
