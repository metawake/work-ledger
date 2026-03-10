# Regression testing for LangGraph apps — Work Ledger

**Category:** Feature Request / Show and Tell

## Problem

LangGraph has great built-in state persistence and time-travel via checkpoints, but there's a gap when it comes to **cross-run regression testing**: comparing two executions of the same graph (e.g. before/after a prompt change) to see exactly what changed — steps, outputs, token usage, cost.

Currently, developers either eyeball outputs manually or build custom diffing scripts. There's no standard way to:

1. Record a graph execution as a structured artifact (inputs, outputs, steps, metrics)
2. Replay it deterministically against a new version
3. Get a structured diff: "step X produced different output, tokens went from 70 → 120, new tool call added"

## Proposal

[Work Ledger](https://github.com/metawake/work-ledger) is an open-source library that does exactly this. It already has a LangGraph integration:

```python
from langgraph.graph import StateGraph
from work_ledger import WorkLedger
from work_ledger.integrations.langgraph import wrap_graph

graph = builder.compile()
ledger = WorkLedger(store="./runs")
wrapped = wrap_graph(graph, ledger)

result = wrapped.invoke({"messages": [HumanMessage("Hello!")]})
```

After recording two runs, you can diff them:

```python
from work_ledger.testing.diff import RunDiff

diff = RunDiff(run_v1, run_v2)
print(f"Similarity: {diff.similarity:.0%}")
print(f"Steps added: {diff.steps_added}, removed: {diff.steps_removed}")
print(f"Token delta: {diff.token_diff:+d}")
```

Or via CLI:

```bash
work-ledger diff <run-id-1> <run-id-2>
```

It also provides pytest decorators for golden-file testing:

```python
@golden(store="./golden", name="qa-graph")
def test_qa_graph():
    return graph.invoke({"messages": [HumanMessage("What is AI?")]})
```

## How it complements LangGraph checkpoints

| Feature | LangGraph Checkpoints | Work Ledger |
|---|---|---|
| State persistence | Yes (built-in) | Captures full run artifacts |
| Time travel | Yes (replay from checkpoint) | Cross-run comparison & diff |
| Regression testing | — | Structured diff, golden tests |
| Multi-framework | LangGraph only | LangGraph + LangChain + others |
| CLI inspection | — | `list`, `show`, `diff`, `replay` |

Work Ledger doesn't replace checkpoints — it sits on top as a testing/debugging layer.

## Questions for the community

1. Would a tighter integration be useful — e.g. a LangGraph checkpoint-backed store, or automatic step recording via LangGraph's event system?
2. Is there interest in a `langgraph-contrib` package or docs recipe for regression testing workflows?
3. What does your current "did my graph break after this change?" workflow look like?

Happy to take feedback and adapt. The library is MIT-licensed and the goal is to be a lightweight, composable tool — not another observability platform.
