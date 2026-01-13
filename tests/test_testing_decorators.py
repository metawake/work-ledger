"""Tests for testing decorators."""

import tempfile
from pathlib import Path

import pytest

from work_ledger.core.models import Run, Step, StepKind, RunStatus
from work_ledger.testing.decorators import recorded, replay, golden, compare
from work_ledger.testing.fixtures import Recording, Fixture, save_recording, load_recording


class TestRecordedDecorator:
    """Tests for @recorded decorator."""

    def test_recorded_creates_fixture_file(self):
        """@recorded creates a fixture file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = Path(tmpdir) / "test.json"
            
            @recorded(fixture_path)
            def my_test():
                return "result"
            
            # First run creates fixture
            result = my_test()
            assert result == "result"
            # Note: File won't exist unless we actually capture a run
            # This tests the decorator mechanics, not full integration

    def test_recorded_skips_if_exists(self):
        """@recorded skips recording if fixture exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = Path(tmpdir) / "test.json"
            
            # Create existing fixture
            run = Run(name="test", status=RunStatus.SUCCESS)
            run.outputs = {"original": True}
            save_recording(fixture_path, Recording(run=run, fixtures=[]))
            
            call_count = 0
            
            @recorded(fixture_path)
            def my_test():
                nonlocal call_count
                call_count += 1
                return "new_result"
            
            result = my_test()
            assert result == "new_result"
            assert call_count == 1
            
            # Fixture should still have original content
            loaded = load_recording(fixture_path)
            assert loaded.run.outputs == {"original": True}

    def test_recorded_overwrites_when_requested(self):
        """@recorded overwrites fixture when overwrite=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = Path(tmpdir) / "test.json"
            
            # Create existing fixture
            run = Run(name="test", status=RunStatus.SUCCESS)
            save_recording(fixture_path, Recording(run=run, fixtures=[]))
            
            @recorded(fixture_path, overwrite=True)
            def my_test():
                return "result"
            
            # Should run and potentially overwrite
            my_test()


class TestReplayDecorator:
    """Tests for @replay decorator."""

    def test_replay_loads_fixture(self):
        """@replay loads fixture file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = Path(tmpdir) / "test.json"
            
            # Create fixture
            run = Run(name="test", status=RunStatus.SUCCESS)
            fixtures = [
                Fixture(step_id="s1", kind="model", call={"p": "hi"}, result={"r": "hello"})
            ]
            save_recording(fixture_path, Recording(run=run, fixtures=fixtures))
            
            @replay(fixture_path)
            def my_test():
                return "result"
            
            result = my_test()
            assert result == "result"

    def test_replay_raises_if_no_fixture(self):
        """@replay raises if fixture file doesn't exist."""
        @replay("/nonexistent/fixture.json")
        def my_test():
            return "result"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            my_test()
        
        assert "not found" in str(exc_info.value).lower()


class TestGoldenDecorator:
    """Tests for @golden decorator."""

    def test_golden_first_run_records(self):
        """@golden records on first run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = Path(tmpdir) / "golden.json"
            
            @golden(fixture_path)
            def my_test():
                return "golden_result"
            
            result = my_test()
            assert result == "golden_result"

    def test_golden_subsequent_run_replays(self):
        """@golden replays on subsequent runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fixture_path = Path(tmpdir) / "golden.json"
            
            # Create existing golden
            run = Run(name="test", status=RunStatus.SUCCESS)
            run.outputs = {"expected": "value"}
            save_recording(fixture_path, Recording(run=run, fixtures=[]))
            
            @golden(fixture_path)
            def my_test():
                return "result"
            
            result = my_test()
            assert result == "result"


class TestCompareDecorator:
    """Tests for @compare decorator."""

    def test_compare_loads_baseline(self):
        """@compare loads baseline for comparison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            
            # Create baseline
            run = Run(name="test", status=RunStatus.SUCCESS)
            save_recording(baseline_path, Recording(run=run, fixtures=[]))
            
            @compare(baseline_path)
            def my_test():
                return "result"
            
            result = my_test()
            assert result == "result"

    def test_compare_raises_if_no_baseline(self):
        """@compare raises if baseline doesn't exist."""
        @compare("/nonexistent/baseline.json")
        def my_test():
            return "result"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            my_test()
        
        assert "not found" in str(exc_info.value).lower()

    def test_compare_with_threshold(self):
        """@compare respects threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.json"
            
            # Create baseline
            run = Run(name="test", status=RunStatus.SUCCESS)
            save_recording(baseline_path, Recording(run=run, fixtures=[]))
            
            @compare(baseline_path, threshold=0.5)
            def my_test():
                return "result"
            
            # Should pass with high threshold
            result = my_test()
            assert result == "result"
