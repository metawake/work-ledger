# Work Ledger Samples

**Record. Replay. Diff.** See what your agent actually did.

## Pick Your Problem

| Your Problem | Sample | Time |
|--------------|--------|------|
| "I want to see what my agent actually did" | [01_basic_recording.py](01_basic_recording.py) | 2 min |
| "I need to test without burning API credits" | [02_testing_basics.py](02_testing_basics.py) | 3 min |
| "Something changed and I don't know what" | [03_detect_regression.py](03_detect_regression.py) | 5 min |
| "Why is my agent stuck / slow / wrong?" | [04_debug_workflow.py](04_debug_workflow.py) | 5 min |
| "I use PydanticAI — how do I integrate?" | [05_pydantic_ai_integration.py](05_pydantic_ai_integration.py) | 3 min |
| "I use LangGraph — how do I integrate?" | [06_langgraph_integration.py](06_langgraph_integration.py) | 3 min |
| "I use CrewAI — how do I integrate?" | [07_crewai_integration.py](07_crewai_integration.py) | 3 min |

## Real API Tests

See [`real/`](real/) for integration tests against live APIs (requires API keys).

## Run Right Now (No API Key)

```bash
cd samples
python 03_detect_regression.py  # See a diff in action
```

Mock samples use no external dependencies.

## What You'll See

```
DIFF: What Changed?

Run Diff:
  Similarity: 58.3%

  Outputs changed:
    ~ summary: "AI is transforming..." → "AI and ML are reshaping..."
  
  Steps:
    Added: 1 (retrieve-context)

✗ Regression detected: unexpected new steps
```

**Now you know what broke.**
