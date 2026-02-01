# Data Model

Work Ledger uses a structured data model to capture agent runs.

## Run

A **Run** is the system's reaction to an activation — not just a function call.

```
┌─────────────────────────────────────────────────────────────┐
│                          RUN                                 │
├─────────────────────────────────────────────────────────────┤
│  inputs        → what came in                               │
│  steps[]       → what was done (model/tool/retrieval calls) │
│  outputs       → what came out                              │
│  metrics       → tokens, latency, cost                      │
│  causal_links  → what caused what                           │
│  annotations   → optional extensible metadata               │
└─────────────────────────────────────────────────────────────┘
```

### Run Fields

| Field | Description |
|-------|-------------|
| `run_id` | Unique identifier |
| `name` | Human-readable name |
| `started_at` / `ended_at` | Temporal boundaries |
| `inputs` | What triggered the run |
| `outputs` | What the run produced |
| `status` | success / failed / cancelled |
| `metrics` | Aggregated metrics |
| `steps[]` | Ordered list of steps |
| `links` | Causal / parent / correlation links |
| `annotations` | Optional extensible metadata |

## Step

A **Step** is a single operation within a run.

| Field | Description |
|-------|-------------|
| `step_id` | Unique identifier |
| `name` | Human-readable name |
| `kind` | `model` / `tool` / `retrieval` / `custom` |
| `inputs` | Step inputs |
| `outputs` | Step outputs |
| `metrics` | Step-level metrics |
| `started_at` / `ended_at` | Temporal boundaries |
| `caused_by` | What triggered this step |

## Metrics (Vendor-Agnostic)

- Token usage (prompt / completion / total)
- Latency (milliseconds)
- Cost (estimated USD)
- Retries

## Causal Links

Work Ledger explicitly models **causality**, not just temporal sequence.

Each step can have:
- `caused_by` — the event/step/message that triggered it
- `correlation_id` — for grouping related operations
- `parent_run_id` / `parent_step_id` — for hierarchical structures

This enables:
- Reconstructing decision chains
- Understanding why an agent started doing something
- Investigating loops and cascade failures

## Serialization

Runs serialize to JSON:

```python
run = ledger.get_run("abc123")
print(run.to_dict())
```

```json
{
  "run_id": "abc123",
  "name": "process-request",
  "status": "success",
  "started_at": "2024-01-15T10:30:00Z",
  "ended_at": "2024-01-15T10:30:02Z",
  "inputs": {"query": "What's the weather?"},
  "outputs": {"response": "It's sunny, 72°F"},
  "metrics": {
    "total_tokens": 150,
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "latency_ms": 1200
  },
  "steps": [
    {
      "step_id": "step_001",
      "name": "retrieve-context",
      "kind": "retrieval",
      "inputs": {"query": "weather"},
      "outputs": {"docs": ["..."]}
    }
  ]
}
```
