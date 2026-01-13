"""Testing utilities for Work Ledger.

This module provides decorators and assertions for testing agent
workflows against recorded fixtures.

**Complements, not replaces** existing testing tools:

- Use **pytest + mocks** for fast unit tests
- Use **PydanticAI TestModel** for mocking LLM calls
- Use **Work Ledger testing** for workflow recording, replay, and diff

Work Ledger testing adds what mocks can't provide:
- Full execution trace (steps, I/O, causal links)
- Deterministic replay of recorded runs
- Diff-based regression detection
- Same recording format works dev → prod

Example:
    >>> from work_ledger.testing import recorded, replay, assert_no_regression
    >>> 
    >>> # First run: record the execution
    >>> @recorded("fixtures/my_agent.json")
    >>> def test_record(agent):
    ...     result = agent.run("test query")
    ...     assert "response" in result
    >>> 
    >>> # Subsequent runs: replay without API calls
    >>> @replay("fixtures/my_agent.json", diff=True)
    >>> def test_replay(agent):
    ...     result = agent.run("test query")
    ...     # Automatically fails if execution differs

Recording workflow:
    1. Use @recorded to capture a golden run
    2. Use @replay to run tests against captured fixtures
    3. Use assertions to verify behavior

Available decorators:
    - @recorded: Capture fixtures during test execution
    - @replay: Run against captured fixtures (no API calls)
    - @golden: Record once, compare on subsequent runs
    - @compare: Live run vs baseline comparison

Available assertions:
    - assert_run_matches: Compare two runs
    - assert_steps_match: Compare step sequences
    - assert_output_matches: Check output structure
    - assert_no_regression: Check for regressions
"""

from work_ledger.testing.assertions import (
    assert_no_regression,
    assert_output_matches,
    assert_run_matches,
    assert_steps_match,
)
from work_ledger.testing.decorators import (
    compare,
    golden,
    recorded,
    replay,
)
from work_ledger.testing.diff import RunDiff, StepDiff, format_diff
from work_ledger.testing.fixtures import (
    Fixture,
    FixtureInjector,
    FixtureRecorder,
    Recording,
    load_recording,
    save_recording,
)

__all__ = [
    # Decorators
    "recorded",
    "replay",
    "golden",
    "compare",
    # Fixtures
    "Fixture",
    "Recording",
    "FixtureRecorder",
    "FixtureInjector",
    "save_recording",
    "load_recording",
    # Diff
    "RunDiff",
    "StepDiff",
    "format_diff",
    # Assertions
    "assert_run_matches",
    "assert_steps_match",
    "assert_output_matches",
    "assert_no_regression",
]
