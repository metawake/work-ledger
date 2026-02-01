# Testing Module

Work Ledger includes a testing framework that **complements** (not replaces) existing tools like pytest, PydanticAI's `TestModel`, and LangSmith evaluations.

## How It Fits With Existing Tools

```
┌─────────────────────────────────────────────────────────────────┐
│                    Testing Pyramid                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  UNIT TESTS (existing tools)                                     │
│  ├─ pytest + mocks                                               │
│  ├─ PydanticAI TestModel ──▶ "Does my tool format correctly?"   │
│  └─ LangChain mock chains                                        │
│                                                                  │
│  WORKFLOW TESTS (Work Ledger)                                    │
│  ├─ @recorded / @replay ───▶ "Is the execution path the same?"  │
│  ├─ RunDiff ───────────────▶ "What changed between runs?"       │
│  └─ Causal chain ──────────▶ "Why did step B follow step A?"    │
│                                                                  │
│  PRODUCTION DEBUGGING (Work Ledger)                              │
│  ├─ Same recording format ─▶ Replay prod issues locally         │
│  └─ Forensic analysis ─────▶ "What happened at 3am?"            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Different Questions, Same Workflow

| Tool | Question It Answers |
|------|---------------------|
| **pytest + mocks** | "Does this function return the right thing?" |
| **PydanticAI TestModel** | "Does my agent handle this input correctly?" |
| **LangSmith evals** | "How good is my agent on this dataset?" |
| **Work Ledger** | "What did my agent actually do, and did it change?" |

## Using Them Together

```python
# FAST UNIT TESTS: PydanticAI mocks (no API calls)
from pydantic_ai.models.test import TestModel

def test_unit_fast():
    agent = Agent(TestModel())  # Mock LLM
    result = agent.run_sync("test")
    assert result.valid

# WORKFLOW RECORDING: Work Ledger captures execution
from work_ledger.testing import recorded, replay

@recorded("fixtures/weather.json")
def test_record_integration(real_agent):
    """First run: captures full execution trace."""
    result = real_agent.run("What's the weather?")
    assert "temperature" in result

# REGRESSION TESTING: Replay without API calls
@replay("fixtures/weather.json", diff=True)
def test_no_regression(real_agent):
    """Subsequent runs: replays and diffs."""
    result = real_agent.run("What's the weather?")
    # Automatically fails if execution path changes
```

## Testing API

```python
from work_ledger.testing import (
    # Decorators
    recorded,      # Capture fixtures on first run
    replay,        # Run against fixtures (no API calls)
    golden,        # Record once, compare on subsequent runs
    compare,       # Live run vs baseline with threshold
    
    # Assertions
    assert_run_matches,     # Compare two runs
    assert_steps_match,     # Compare step sequences
    assert_output_matches,  # Check output structure
    assert_no_regression,   # Detect regressions
    
    # Diff
    RunDiff,       # Detailed diff computation
    format_diff,   # Human-readable diff output
)
```

## Decorators

### @recorded

Captures a run on first execution, skips if fixture exists:

```python
@recorded("fixtures/my_test.json")
def test_capture():
    result = agent.run("test")
    return result
```

### @replay

Loads fixture and injects saved responses:

```python
@replay("fixtures/my_test.json")
def test_replay():
    result = agent.run("test")  # Uses saved responses
    assert result == expected
```

### @golden

Records on first run, replays and compares on subsequent runs:

```python
@golden("fixtures/baseline.json")
def test_golden():
    result = agent.run("test")
    # First run: records
    # Later runs: compares to baseline
```

### @compare

Runs live and compares to baseline with threshold:

```python
@compare("fixtures/baseline.json", threshold=0.9)
def test_compare():
    result = agent.run("test")
    # Fails if similarity < 90%
```

## Assertions

```python
from work_ledger.testing import (
    assert_run_matches,
    assert_no_regression,
    assert_output_matches,
)

# Compare runs
assert_run_matches(run1, run2, ignore_timing=True)

# Check for regressions
assert_no_regression(baseline, current, threshold=0.95)

# Verify output structure
assert_output_matches(run.outputs, expected_keys=["response", "confidence"])
```

## RunDiff

Detailed comparison between runs:

```python
from work_ledger.testing import RunDiff

diff = RunDiff(expected_run, actual_run)

print(f"Similarity: {diff.similarity:.1%}")
print(f"Status changed: {diff.status_changed}")
print(f"Steps added: {len(diff.steps_added)}")
print(f"Steps removed: {len(diff.steps_removed)}")
print(f"Output changes: {diff.output_diff}")
```

## Replay via Wrappers

The wrappers support direct replay via the `replay_from` parameter:

```python
from work_ledger import WorkLedger, wrap_openai

ledger = WorkLedger(store="./runs")
client = OpenAI()

# Step 1: Record (makes real API call)
wrapped = wrap_openai(client, ledger)
response = wrapped.chat.completions.create(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4"
)
run_id = ledger.list_runs()[0].run_id

# Step 2: Replay (returns saved fixture, no API call)
wrapped = wrap_openai(client, ledger, replay_from=run_id)
response = wrapped.chat.completions.create(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4"
)
# response.choices[0].message.content → same as original
```

### Supported Wrappers

| Wrapper | Replay Support |
|---------|----------------|
| `wrap_openai` | ✅ Full |
| `wrap_anthropic` | ✅ Full |
| `wrap_agent` (PydanticAI) | ✅ Full |
| `wrap_graph` (LangGraph) | Planned |
| `wrap_chain` (LangChain) | Planned |

### Replay Errors

When replay diverges from recording:

```python
from work_ledger.integrations.openai import ReplayError

try:
    # More calls than fixtures
    wrapped.chat.completions.create(...)  # 3rd call
except ReplayError as e:
    print(e)  # "Replay diverged: expected 2 API calls, got call #3"
```

### CLI Replay Info

```bash
work-ledger replay ./runs <run_id>
```

```
Run: abc123...
Name: my-agent
Status: SUCCESS

Fixtures: 2 API call(s) captured

Steps:
  1. [model] openai.chat.completions.create
     Fixture: ✓ openai.chat.completions.create
     Tokens: 150
  2. [tool] get_weather
  3. [model] openai.chat.completions.create
     Fixture: ✓ openai.chat.completions.create
     Tokens: 89

To replay in Python:

  from work_ledger import WorkLedger, wrap_openai
  ledger = WorkLedger(store="./runs")
  wrapped = wrap_openai(client, ledger, replay_from="abc123...")
  response = wrapped.chat.completions.create(...)  # No API call
```

## Why Not Just Use Mocks?

| Approach | Pros | Cons |
|----------|------|------|
| **Mocks (TestModel)** | Fast, isolated, deterministic | No execution trace, can't debug workflows |
| **Work Ledger Recording** | Full trace, replay, diff, same format dev→prod | Initial recording requires real calls |

**Best practice:** Use mocks for fast unit tests, Work Ledger for integration tests and debugging.
