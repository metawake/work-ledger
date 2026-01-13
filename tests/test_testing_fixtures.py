"""Tests for testing fixtures and recordings."""

import json
import tempfile
from pathlib import Path

import pytest

from work_ledger.core.models import Run, Step, StepKind, Metrics, RunStatus
from work_ledger.testing.fixtures import (
    Fixture,
    Recording,
    save_recording,
    load_recording,
)


class TestFixture:
    """Tests for Fixture dataclass."""

    def test_create_fixture(self):
        """Fixture can be created with all fields."""
        fixture = Fixture(
            step_id="step-123",
            kind="model",
            call={"model": "gpt-4", "prompt": "Hello"},
            result={"response": "Hi there", "tokens": 10},
        )
        assert fixture.step_id == "step-123"
        assert fixture.kind == "model"
        assert fixture.call["model"] == "gpt-4"
        assert fixture.result["response"] == "Hi there"
        assert fixture.error is None

    def test_create_fixture_with_error(self):
        """Fixture can capture errors."""
        fixture = Fixture(
            step_id="step-123",
            kind="tool",
            call={"tool": "search", "query": "test"},
            result=None,
            error="Connection timeout",
        )
        assert fixture.error == "Connection timeout"
        assert fixture.result is None

    def test_fixture_to_dict(self):
        """Fixture can be serialized."""
        fixture = Fixture(
            step_id="step-123",
            kind="model",
            call={"prompt": "test"},
            result={"response": "result"},
        )
        d = fixture.to_dict()
        assert d["step_id"] == "step-123"
        assert d["kind"] == "model"
        assert d["call"] == {"prompt": "test"}
        assert d["result"] == {"response": "result"}

    def test_fixture_from_dict(self):
        """Fixture can be deserialized."""
        d = {
            "step_id": "step-123",
            "kind": "tool",
            "call": {"tool": "api"},
            "result": {"status": 200},
        }
        fixture = Fixture.from_dict(d)
        assert fixture.step_id == "step-123"
        assert fixture.kind == "tool"


class TestRecording:
    """Tests for Recording dataclass."""

    def test_create_recording(self):
        """Recording can be created with run and fixtures."""
        run = Run(run_id="run-123", name="test-run", status=RunStatus.SUCCESS)
        fixtures = [
            Fixture(step_id="s1", kind="model", call={}, result={"x": 1}),
            Fixture(step_id="s2", kind="tool", call={}, result={"y": 2}),
        ]
        recording = Recording(run=run, fixtures=fixtures)
        
        assert recording.run.run_id == "run-123"
        assert len(recording.fixtures) == 2

    def test_recording_with_metadata(self):
        """Recording can have metadata."""
        run = Run(name="test")
        recording = Recording(
            run=run,
            fixtures=[],
            metadata={"version": "1.0", "recorded_at": "2024-01-15"},
        )
        assert recording.metadata["version"] == "1.0"

    def test_recording_to_dict(self):
        """Recording can be serialized."""
        run = Run(run_id="run-123", name="test", status=RunStatus.SUCCESS)
        run.add_step(Step(step_id="s1", name="step1", kind=StepKind.MODEL))
        fixtures = [
            Fixture(step_id="s1", kind="model", call={"p": 1}, result={"r": 2}),
        ]
        recording = Recording(run=run, fixtures=fixtures)
        
        d = recording.to_dict()
        assert d["run"]["run_id"] == "run-123"
        assert len(d["fixtures"]) == 1
        assert d["fixtures"][0]["step_id"] == "s1"

    def test_recording_from_dict(self):
        """Recording can be deserialized."""
        d = {
            "run": {
                "run_id": "run-123",
                "name": "test",
                "status": "success",
                "steps": [{"step_id": "s1", "name": "step1", "kind": "model"}],
            },
            "fixtures": [
                {"step_id": "s1", "kind": "model", "call": {}, "result": {"x": 1}},
            ],
            "metadata": {"version": "1.0"},
        }
        recording = Recording.from_dict(d)
        
        assert recording.run.run_id == "run-123"
        assert len(recording.run.steps) == 1
        assert len(recording.fixtures) == 1
        assert recording.metadata["version"] == "1.0"


class TestSaveLoadRecording:
    """Tests for saving and loading recordings."""

    def test_save_and_load_recording(self):
        """Recording can be saved and loaded."""
        run = Run(run_id="run-123", name="test", status=RunStatus.SUCCESS)
        run.add_step(Step(step_id="s1", name="llm", kind=StepKind.MODEL))
        fixtures = [
            Fixture(step_id="s1", kind="model", call={"prompt": "hi"}, result={"text": "hello"}),
        ]
        recording = Recording(run=run, fixtures=fixtures)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_recording(path, recording)
            
            loaded = load_recording(path)
            
            assert loaded.run.run_id == "run-123"
            assert loaded.run.name == "test"
            assert len(loaded.fixtures) == 1
            assert loaded.fixtures[0].result["text"] == "hello"

    def test_save_creates_directory(self):
        """save_recording creates parent directories."""
        run = Run(name="test")
        recording = Recording(run=run, fixtures=[])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "test.json"
            save_recording(path, recording)
            
            assert path.exists()

    def test_load_nonexistent_raises(self):
        """Loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_recording("/nonexistent/path.json")

    def test_recording_preserves_complex_data(self):
        """Recording preserves complex nested data."""
        run = Run(run_id="run-123", name="test", status=RunStatus.SUCCESS)
        run.inputs = {"query": "test", "options": {"limit": 10, "filters": ["a", "b"]}}
        run.outputs = {"results": [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}]}
        
        fixtures = [
            Fixture(
                step_id="s1",
                kind="retrieval",
                call={"query": "test", "k": 5},
                result={"documents": [{"text": "doc1"}, {"text": "doc2"}]},
            ),
        ]
        recording = Recording(run=run, fixtures=fixtures)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_recording(path, recording)
            loaded = load_recording(path)
            
            assert loaded.run.inputs["options"]["filters"] == ["a", "b"]
            assert loaded.fixtures[0].result["documents"][0]["text"] == "doc1"
