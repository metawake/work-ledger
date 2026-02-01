# Integrations

Work Ledger provides thin wrappers for popular agent frameworks and LLM SDKs.

## Quick Start

```python
from work_ledger import WorkLedger

ledger = WorkLedger(store="./runs")
```

## Agent Frameworks

### PydanticAI

```python
from work_ledger import wrap_agent

wrapped = wrap_agent(my_agent, ledger)
result = wrapped.run_sync("Hello!")
```

### LangGraph

```python
from work_ledger import wrap_graph

wrapped = wrap_graph(my_graph, ledger)
result = wrapped.invoke({"messages": [...]})
```

### CrewAI

```python
from work_ledger import wrap_crew

wrapped = wrap_crew(my_crew, ledger)
result = wrapped.kickoff(inputs={"topic": "AI"})
```

### LangChain

```python
from work_ledger import wrap_chain

wrapped = wrap_chain(my_chain, ledger)
result = wrapped.invoke({"question": "..."})
```

## RAG Pipelines

### LlamaIndex

```python
from work_ledger import wrap_query_engine

wrapped = wrap_query_engine(my_engine, ledger)
response = wrapped.query("What is machine learning?")
```

## Direct LLM SDKs

### OpenAI

```python
from work_ledger import wrap_openai

wrapped = wrap_openai(openai_client, ledger)
response = wrapped.chat.completions.create(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4"
)
```

### Anthropic

```python
from work_ledger import wrap_anthropic

wrapped = wrap_anthropic(anthropic_client, ledger)
response = wrapped.messages.create(
    messages=[{"role": "user", "content": "Hello"}],
    model="claude-3-opus",
    max_tokens=1024
)
```

## Replay Support

The following integrations support replay (no API calls):

```python
# Record once
wrapped = wrap_openai(client, ledger)
response = wrapped.chat.completions.create(...)
run_id = ledger.list_runs()[0].run_id

# Replay many times (free, instant)
wrapped = wrap_openai(client, ledger, replay_from=run_id)
response = wrapped.chat.completions.create(...)  # No API call
```

| Integration | Replay Support |
|-------------|----------------|
| OpenAI SDK | ✅ Full |
| Anthropic SDK | ✅ Full |
| PydanticAI | ✅ Full |
| LangGraph | 🔜 Planned |
| LangChain | 🔜 Planned |
| CrewAI | 🔜 Planned |
| LlamaIndex | 🔜 Planned |

## What Gets Recorded

| Feature | PydanticAI | LangGraph | CrewAI | LangChain | LlamaIndex | OpenAI | Anthropic |
|---------|-----------|-----------|--------|-----------|------------|--------|-----------|
| Inputs/outputs | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tool calls | ✅ | stream | — | ✅ | — | ✅ | ✅ |
| Token metrics | ✅ | — | ✅ | — | — | ✅ | ✅ |
| Retrieved docs | — | — | — | — | ✅ | — | — |
| Task/steps | — | ✅ | ✅ | ✅ | ✅ | — | — |
| Errors | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Async | ✅ | ✅ | ✅ | ✅ | ✅ | — | — |
| Replay | ✅ | — | — | — | — | ✅ | ✅ |

## Custom Integration

For frameworks not listed, use the manual recording API:

```python
with ledger.run(name="custom-agent") as run:
    run.record_input({"query": query})
    
    with run.step("llm-call", kind="model") as step:
        response = my_custom_llm(query)
        step.record_output({"response": response})
    
    run.record_output({"result": response})
```

## Installation

Integrations are optional dependencies:

```bash
# Individual
pip install work-ledger[pydantic-ai]
pip install work-ledger[langchain]
pip install work-ledger[openai]

# All integrations
pip install work-ledger[integrations]
```

## Version Compatibility

| Integration | Minimum | Tested | Notes |
|-------------|---------|--------|-------|
| OpenAI SDK | 1.0.0 | 1.50+ | v1 API, stable |
| Anthropic SDK | 0.25.0 | 0.40+ | Pre-1.0 but stable |
| PydanticAI | 0.0.14 | 0.0.20+ | Rapidly evolving |
| LangChain | 0.2.0 | 0.2+ | Requires langchain-core |
| LangGraph | 0.2.0 | 0.2+ | Graph-based agents |
| CrewAI | 0.50.0 | 0.70+ | Multi-agent orchestration |
| LlamaIndex | 0.10.0 | 0.10+ | RAG pipelines |

### Compatibility Approach

Work Ledger uses **duck typing** and defensive attribute access:

```python
# We handle API changes gracefully
if hasattr(result, "output"):
    data = result.output
elif hasattr(result, "data"):
    data = result.data  # Fallback for older versions
```

If you're using an older version that breaks, please [open an issue](https://github.com/metawake/work-ledger/issues).

## Complementary Tools

### Retrieval Debugging

Work Ledger records retrieval steps (query, retrieved docs). For deep retrieval analysis:

**[Ragtune](https://github.com/metawake/ragtune)** — "EXPLAIN ANALYZE for RAG retrieval"
- Debug why specific queries fail
- Benchmark with Recall@k, MRR, NDCG
- CI/CD quality gates
- Compare embedders and chunk sizes

```
Work Ledger          Ragtune
────────────         ───────
Records WHAT         Measures HOW GOOD
happened             retrieval is
```

Workflow:
1. Record RAG runs with Work Ledger
2. Export retrieval queries
3. Benchmark with `ragtune simulate`
