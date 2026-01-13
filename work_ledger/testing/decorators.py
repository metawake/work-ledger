"""Test decorators for recording and replaying agent runs.

This module provides pytest-compatible decorators for:
- Recording test fixtures
- Replaying tests against fixtures
- Golden master testing
- Comparing runs against baselines
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Callable, TypeVar

from work_ledger.testing.fixtures import (
    FixtureInjector,
    FixtureRecorder,
    Recording,
    load_recording,
    save_recording,
)
from work_ledger.testing.diff import RunDiff, format_diff

F = TypeVar("F", bound=Callable[..., Any])


def recorded(
    fixture_path: str | Path,
    overwrite: bool = False,
) -> Callable[[F], F]:
    """Decorator that records a test run and saves fixtures.
    
    Use this to create golden recordings that can be replayed later.
    On first run, captures fixtures; on subsequent runs, skips if
    fixture file exists (unless overwrite=True).
    
    Args:
        fixture_path: Where to save the recording
        overwrite: If True, always overwrite existing recording
        
    Returns:
        Decorated function
        
    Example:
        >>> @recorded("fixtures/weather.json")
        >>> def test_record_weather(agent):
        ...     result = agent.run("What's the weather?")
        ...     assert "temperature" in result
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            path = Path(fixture_path)
            
            if path.exists() and not overwrite:
                # Skip recording, run normally
                return func(*args, **kwargs)
            
            # Record the run
            with FixtureRecorder() as recorder:
                result = func(*args, **kwargs)
                
                # Only save if we captured a run
                if recorder._run is not None:
                    save_recording(path, recorder.recording)
            
            return result
        return wrapper  # type: ignore
    return decorator


def replay(
    fixture_path: str | Path,
    strict: bool = True,
    diff: bool = False,
) -> Callable[[F], F]:
    """Decorator that replays a test against recorded fixtures.
    
    External calls (LLM, tools, HTTP) are replaced with recorded results.
    No actual API calls are made — tests run fast and deterministically.
    
    Args:
        fixture_path: Path to the recording
        strict: If True, fail when a call isn't in fixtures
        diff: If True, compare actual run against recording
        
    Returns:
        Decorated function
        
    Example:
        >>> @replay("fixtures/weather.json")
        >>> def test_weather_replay(agent):
        ...     result = agent.run("What's the weather?")
        ...     assert "temperature" in result
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            path = Path(fixture_path)
            
            if not path.exists():
                raise FileNotFoundError(
                    f"Fixture file not found: {path}\n"
                    f"Run with @recorded first to capture fixtures."
                )
            
            recording = load_recording(path)
            
            with FixtureInjector(recording.fixtures, strict=strict) as injector:
                result = func(*args, **kwargs)
                
                if diff and injector.captured_run is not None:
                    # Compare actual run to recorded run
                    run_diff = RunDiff(recording.run, injector.captured_run)
                    if run_diff.has_changes:
                        raise AssertionError(
                            f"Run differs from recording:\n{format_diff(run_diff)}"
                        )
            
            return result
        return wrapper  # type: ignore
    return decorator


def golden(fixture_path: str | Path) -> Callable[[F], F]:
    """Decorator that treats a recording as the expected "golden" output.
    
    Combines recording and replay:
    - First run: Records the run as golden
    - Subsequent runs: Replays and asserts output matches
    
    Args:
        fixture_path: Path to store/load the golden recording
        
    Returns:
        Decorated function
        
    Example:
        >>> @golden("fixtures/weather_golden.json")
        >>> def test_weather_golden(agent):
        ...     return agent.run("What's the weather?")
        ...     # First run: records result
        ...     # Later runs: replays and asserts same output
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            path = Path(fixture_path)
            
            if not path.exists():
                # First run — record as golden
                with FixtureRecorder() as recorder:
                    result = func(*args, **kwargs)
                    if recorder._run is not None:
                        save_recording(path, recorder.recording)
                return result
            
            # Subsequent runs — replay and compare
            recording = load_recording(path)
            
            with FixtureInjector(recording.fixtures) as injector:
                result = func(*args, **kwargs)
                
                if injector.captured_run is not None:
                    # Assert outputs match
                    actual_outputs = injector.captured_run.outputs
                    expected_outputs = recording.run.outputs
                    
                    if actual_outputs != expected_outputs:
                        raise AssertionError(
                            f"Output changed from golden:\n"
                            f"Expected: {expected_outputs}\n"
                            f"Actual: {actual_outputs}"
                        )
            
            return result
        return wrapper  # type: ignore
    return decorator


def compare(
    baseline_path: str | Path,
    threshold: float = 0.0,
) -> Callable[[F], F]:
    """Decorator that compares a live run against a baseline.
    
    Unlike @replay, this makes real calls and compares the results.
    Useful for detecting model drift or behavior changes.
    
    Args:
        baseline_path: Path to the baseline recording
        threshold: Allowed difference threshold (0.0 = exact match)
        
    Returns:
        Decorated function
        
    Example:
        >>> @compare("fixtures/baseline.json", threshold=0.1)
        >>> def test_model_drift(agent):
        ...     return agent.run("Summarize this document")
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            path = Path(baseline_path)
            
            if not path.exists():
                raise FileNotFoundError(
                    f"Baseline file not found: {path}\n"
                    f"Create a baseline first with @recorded."
                )
            
            baseline = load_recording(path)
            
            with FixtureRecorder() as recorder:
                result = func(*args, **kwargs)
                
                if recorder._run is not None:
                    diff = RunDiff(baseline.run, recorder._run)
                    
                    if diff.difference > threshold:
                        raise AssertionError(
                            f"Run differs from baseline by {diff.difference:.1%} "
                            f"(threshold: {threshold:.1%}):\n"
                            f"{format_diff(diff)}"
                        )
            
            return result
        return wrapper  # type: ignore
    return decorator
