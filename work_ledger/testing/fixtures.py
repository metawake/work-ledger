"""Fixture recording and injection for testing.

This module provides data structures and utilities for capturing
and replaying external call results (fixtures) during tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from work_ledger.core.models import Run


@dataclass
class Fixture:
    """A captured external call result.
    
    Fixtures capture the inputs and outputs of external calls
    (LLM, tools, HTTP, retrieval) so they can be replayed later.
    
    Attributes:
        step_id: ID of the step that made this call
        kind: Type of call (model, tool, http, retrieval)
        call: The call parameters/inputs
        result: The captured result
        error: Error message if the call failed
        
    Example:
        >>> fixture = Fixture(
        ...     step_id="step-123",
        ...     kind="model",
        ...     call={"model": "gpt-4", "prompt": "Hello"},
        ...     result={"response": "Hi there"},
        ... )
    """
    step_id: str
    kind: str
    call: dict[str, Any]
    result: Any
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize fixture to a dictionary."""
        return {
            "step_id": self.step_id,
            "kind": self.kind,
            "call": self.call,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fixture:
        """Deserialize fixture from a dictionary."""
        return cls(
            step_id=data["step_id"],
            kind=data["kind"],
            call=data.get("call", {}),
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass
class Recording:
    """A complete recording of a run with fixtures.
    
    Recordings contain both the run data (inputs, outputs, steps)
    and the fixtures (captured external call results) needed to
    replay the run deterministically.
    
    Attributes:
        run: The recorded run
        fixtures: List of captured fixtures
        metadata: Optional metadata about the recording
        
    Example:
        >>> recording = Recording(
        ...     run=my_run,
        ...     fixtures=[fixture1, fixture2],
        ...     metadata={"recorded_at": "2024-01-15"},
        ... )
    """
    run: Run
    fixtures: list[Fixture] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize recording to a dictionary."""
        return {
            "run": self.run.to_dict(),
            "fixtures": [f.to_dict() for f in self.fixtures],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recording:
        """Deserialize recording from a dictionary."""
        return cls(
            run=Run.from_dict(data["run"]),
            fixtures=[
                Fixture.from_dict(f) for f in data.get("fixtures", [])
            ],
            metadata=data.get("metadata", {}),
        )


def save_recording(path: str | Path, recording: Recording) -> None:
    """Save a recording to a JSON file.
    
    Creates parent directories if they don't exist.
    
    Args:
        path: File path to save to
        recording: The recording to save
        
    Example:
        >>> save_recording("fixtures/my_test.json", recording)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as f:
        json.dump(recording.to_dict(), f, indent=2)


def load_recording(path: str | Path) -> Recording:
    """Load a recording from a JSON file.
    
    Args:
        path: File path to load from
        
    Returns:
        The loaded recording
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        
    Example:
        >>> recording = load_recording("fixtures/my_test.json")
    """
    with open(path) as f:
        data = json.load(f)
    return Recording.from_dict(data)


class FixtureRecorder:
    """Context manager that records external calls.
    
    Used during test recording to capture fixtures.
    
    Example:
        >>> with FixtureRecorder() as recorder:
        ...     result = agent.run("test")
        ...     recording = recorder.recording
    """
    
    def __init__(self) -> None:
        self._fixtures: list[Fixture] = []
        self._run: Run | None = None
    
    @property
    def recording(self) -> Recording:
        """Get the recorded data."""
        if self._run is None:
            raise RuntimeError("No run captured")
        return Recording(run=self._run, fixtures=self._fixtures)
    
    def capture(
        self,
        step_id: str,
        kind: str,
        call: dict[str, Any],
        result: Any,
        error: str | None = None,
    ) -> None:
        """Capture an external call result.
        
        Args:
            step_id: ID of the step making the call
            kind: Type of call
            call: Call parameters
            result: Call result
            error: Error message if failed
        """
        self._fixtures.append(Fixture(
            step_id=step_id,
            kind=kind,
            call=call,
            result=result,
            error=error,
        ))
    
    def set_run(self, run: Run) -> None:
        """Set the run being recorded."""
        self._run = run
    
    def __enter__(self) -> FixtureRecorder:
        """Enter recording context."""
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Exit recording context."""
        pass


class FixtureInjector:
    """Context manager that injects recorded fixtures.
    
    Used during test replay to substitute real calls
    with recorded results.
    
    Example:
        >>> with FixtureInjector(recording.fixtures) as injector:
        ...     result = agent.run("test")  # Uses fixtures, not real calls
    """
    
    def __init__(
        self,
        fixtures: list[Fixture],
        strict: bool = True,
    ) -> None:
        """Initialize the injector.
        
        Args:
            fixtures: Fixtures to inject
            strict: If True, fail when call not in fixtures
        """
        self._fixtures = {self._key(f): f for f in fixtures}
        self._strict = strict
        self._captured_run: Run | None = None
        self._used_fixtures: set[tuple] = set()
    
    def _key(self, fixture: Fixture) -> tuple:
        """Create a lookup key for a fixture."""
        return (fixture.kind, json.dumps(fixture.call, sort_keys=True))
    
    @property
    def captured_run(self) -> Run | None:
        """Get the run captured during replay."""
        return self._captured_run
    
    def set_run(self, run: Run) -> None:
        """Set the run being replayed."""
        self._captured_run = run
    
    def get_fixture(self, kind: str, call: dict[str, Any]) -> Any:
        """Get a fixture result for a call.
        
        Args:
            kind: Type of call
            call: Call parameters
            
        Returns:
            The recorded result
            
        Raises:
            RuntimeError: If strict and no fixture found
        """
        key = (kind, json.dumps(call, sort_keys=True))
        
        if key in self._fixtures:
            fixture = self._fixtures[key]
            self._used_fixtures.add(key)
            if fixture.error:
                raise RuntimeError(f"Recorded error: {fixture.error}")
            return fixture.result
        
        if self._strict:
            raise RuntimeError(
                f"No fixture found for {kind} call: {call}\n"
                f"Run with @recorded first to capture fixtures."
            )
        
        return None
    
    def __enter__(self) -> FixtureInjector:
        """Enter injection context."""
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Exit injection context."""
        pass
