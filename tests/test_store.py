"""Tests for storage backends."""

import json
import tempfile
from pathlib import Path

import pytest

from work_ledger.core.models import Run, RunStatus, Step, StepKind, Metrics
from work_ledger.core.store import MemoryStore, JSONLStore, RunStore


class TestMemoryStore:
    """Tests for in-memory storage backend."""

    def test_save_and_get_run(self):
        """Runs can be saved and retrieved."""
        store = MemoryStore()
        run = Run(run_id="run-123", name="test-run")
        
        store.save_run(run)
        retrieved = store.get_run("run-123")
        
        assert retrieved is not None
        assert retrieved.run_id == "run-123"
        assert retrieved.name == "test-run"

    def test_get_nonexistent_run(self):
        """Getting non-existent run returns None."""
        store = MemoryStore()
        result = store.get_run("nonexistent")
        assert result is None

    def test_list_runs(self):
        """All runs can be listed."""
        store = MemoryStore()
        store.save_run(Run(run_id="run-1", name="test1"))
        store.save_run(Run(run_id="run-2", name="test2"))
        store.save_run(Run(run_id="run-3", name="test3"))
        
        runs = store.list_runs()
        assert len(runs) == 3

    def test_list_runs_filter_by_name(self):
        """Runs can be filtered by name."""
        store = MemoryStore()
        store.save_run(Run(run_id="run-1", name="process"))
        store.save_run(Run(run_id="run-2", name="process"))
        store.save_run(Run(run_id="run-3", name="other"))
        
        runs = store.list_runs(name="process")
        assert len(runs) == 2

    def test_list_runs_filter_by_status(self):
        """Runs can be filtered by status."""
        store = MemoryStore()
        store.save_run(Run(run_id="run-1", name="t1", status=RunStatus.SUCCESS))
        store.save_run(Run(run_id="run-2", name="t2", status=RunStatus.SUCCESS))
        store.save_run(Run(run_id="run-3", name="t3", status=RunStatus.FAILED))
        
        runs = store.list_runs(status=RunStatus.SUCCESS)
        assert len(runs) == 2

    def test_update_run(self):
        """Runs can be updated."""
        store = MemoryStore()
        run = Run(run_id="run-123", name="test", status=RunStatus.PENDING)
        store.save_run(run)
        
        run.status = RunStatus.SUCCESS
        store.save_run(run)
        
        retrieved = store.get_run("run-123")
        assert retrieved.status == RunStatus.SUCCESS

    def test_delete_run(self):
        """Runs can be deleted."""
        store = MemoryStore()
        store.save_run(Run(run_id="run-123", name="test"))
        
        store.delete_run("run-123")
        
        result = store.get_run("run-123")
        assert result is None

    def test_run_with_steps(self):
        """Runs with steps are saved correctly."""
        store = MemoryStore()
        run = Run(run_id="run-123", name="test")
        run.add_step(Step(step_id="step-1", name="s1", kind=StepKind.TOOL))
        run.add_step(Step(step_id="step-2", name="s2", kind=StepKind.MODEL))
        
        store.save_run(run)
        retrieved = store.get_run("run-123")
        
        assert len(retrieved.steps) == 2
        assert retrieved.steps[0].step_id == "step-1"
        assert retrieved.steps[1].step_id == "step-2"


class TestJSONLStore:
    """Tests for JSONL file storage backend."""

    def test_create_store_creates_directory(self):
        """Store creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "runs"
            store = JSONLStore(store_path)
            
            assert store_path.exists()

    def test_save_and_get_run(self):
        """Runs can be saved and retrieved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONLStore(tmpdir)
            run = Run(run_id="run-123", name="test-run", status=RunStatus.SUCCESS)
            
            store.save_run(run)
            retrieved = store.get_run("run-123")
            
            assert retrieved is not None
            assert retrieved.run_id == "run-123"
            assert retrieved.name == "test-run"
            assert retrieved.status == RunStatus.SUCCESS

    def test_run_persisted_to_file(self):
        """Run is persisted to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONLStore(tmpdir)
            run = Run(run_id="run-123", name="test")
            store.save_run(run)
            
            # Check file exists
            run_file = Path(tmpdir) / "run-123.jsonl"
            assert run_file.exists()
            
            # Check content is valid JSON
            with open(run_file) as f:
                data = json.loads(f.readline())
            assert data["run_id"] == "run-123"

    def test_list_runs(self):
        """All runs can be listed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONLStore(tmpdir)
            store.save_run(Run(run_id="run-1", name="test1"))
            store.save_run(Run(run_id="run-2", name="test2"))
            store.save_run(Run(run_id="run-3", name="test3"))
            
            runs = store.list_runs()
            assert len(runs) == 3

    def test_list_runs_filter_by_name(self):
        """Runs can be filtered by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONLStore(tmpdir)
            store.save_run(Run(run_id="run-1", name="process"))
            store.save_run(Run(run_id="run-2", name="process"))
            store.save_run(Run(run_id="run-3", name="other"))
            
            runs = store.list_runs(name="process")
            assert len(runs) == 2

    def test_delete_run(self):
        """Runs can be deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONLStore(tmpdir)
            store.save_run(Run(run_id="run-123", name="test"))
            
            store.delete_run("run-123")
            
            result = store.get_run("run-123")
            assert result is None
            
            # File should be deleted
            run_file = Path(tmpdir) / "run-123.jsonl"
            assert not run_file.exists()

    def test_run_with_steps_and_metrics(self):
        """Complex runs with steps and metrics are saved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JSONLStore(tmpdir)
            run = Run(run_id="run-123", name="test", status=RunStatus.SUCCESS)
            run.add_step(Step(
                step_id="step-1",
                name="llm-call",
                kind=StepKind.MODEL,
                inputs={"prompt": "Hello"},
                outputs={"response": "Hi there"},
                metrics=Metrics(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.001),
            ))
            
            store.save_run(run)
            retrieved = store.get_run("run-123")
            
            assert len(retrieved.steps) == 1
            assert retrieved.steps[0].inputs == {"prompt": "Hello"}
            assert retrieved.steps[0].outputs == {"response": "Hi there"}
            assert retrieved.steps[0].metrics.total_tokens == 15
            assert retrieved.steps[0].metrics.cost == 0.001


class TestRunStoreFactory:
    """Tests for RunStore factory function."""

    def test_memory_store_from_string(self):
        """':memory:' creates MemoryStore."""
        store = RunStore.create(":memory:")
        assert isinstance(store, MemoryStore)

    def test_jsonl_store_from_path(self):
        """Path string creates JSONLStore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(tmpdir)
            assert isinstance(store, JSONLStore)

    def test_jsonl_store_from_pathlib(self):
        """Path object creates JSONLStore."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = RunStore.create(Path(tmpdir))
            assert isinstance(store, JSONLStore)
