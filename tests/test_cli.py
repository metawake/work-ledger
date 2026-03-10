"""Tests for the CLI commands."""

import argparse
import json
from datetime import datetime, timezone, timedelta
from io import StringIO
from unittest.mock import patch

import pytest

from work_ledger.cli.main import cmd_list, cmd_show, cmd_diff, cmd_replay
from work_ledger.core.models import Run, Step, StepKind, Metrics, RunStatus
from work_ledger.core.store import MemoryStore, RunStore


def _store_with_runs() -> tuple[MemoryStore, Run, Run]:
    """Create a memory store with two runs for testing."""
    store = MemoryStore()

    t1 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t2 = t1 + timedelta(seconds=2)

    run1 = Run(run_id="run-aaa-111", name="agent-v1", status=RunStatus.SUCCESS)
    run1.started_at = t1
    run1.ended_at = t2
    run1.inputs = {"query": "hello"}
    run1.outputs = {"answer": "world"}
    run1.metrics = Metrics(prompt_tokens=50, completion_tokens=20, total_tokens=70, cost=0.001)
    step1 = Step(name="llm-call", kind=StepKind.MODEL)
    step1.metrics = Metrics(total_tokens=70)
    run1.add_step(step1)
    store.save_run(run1)

    run2 = Run(run_id="run-bbb-222", name="agent-v2", status=RunStatus.SUCCESS)
    run2.started_at = t1 + timedelta(minutes=5)
    run2.ended_at = t1 + timedelta(minutes=5, seconds=3)
    run2.inputs = {"query": "hello"}
    run2.outputs = {"answer": "changed"}
    run2.metrics = Metrics(prompt_tokens=80, completion_tokens=40, total_tokens=120, cost=0.003)
    step2a = Step(name="llm-call", kind=StepKind.MODEL)
    step2a.metrics = Metrics(total_tokens=90)
    step2b = Step(name="search", kind=StepKind.RETRIEVAL)
    run2.add_step(step2a)
    run2.add_step(step2b)
    store.save_run(run2)

    return store, run1, run2


class TestCmdList:

    def test_list_empty(self, capsys):
        args = argparse.Namespace(store=":memory:", json=False)
        with patch.object(RunStore, "create", return_value=MemoryStore()):
            ret = cmd_list(args)
        assert ret == 0
        assert "No runs found" in capsys.readouterr().out

    def test_list_runs(self, capsys):
        store, _, _ = _store_with_runs()
        args = argparse.Namespace(store="./fake", json=False)
        with patch.object(RunStore, "create", return_value=store):
            ret = cmd_list(args)
        out = capsys.readouterr().out
        assert ret == 0
        assert "agent-v1" in out
        assert "agent-v2" in out
        assert "Total: 2" in out

    def test_list_json(self, capsys):
        store, _, _ = _store_with_runs()
        args = argparse.Namespace(store="./fake", json=True)
        with patch.object(RunStore, "create", return_value=store):
            ret = cmd_list(args)
        data = json.loads(capsys.readouterr().out)
        assert ret == 0
        assert len(data) == 2


class TestCmdShow:

    def test_show_run(self, capsys):
        store, run1, _ = _store_with_runs()
        args = argparse.Namespace(store="./fake", run_id=run1.run_id, json=False)
        with patch.object(RunStore, "create", return_value=store):
            ret = cmd_show(args)
        out = capsys.readouterr().out
        assert ret == 0
        assert "agent-v1" in out
        assert "Cost: $0.0010" in out
        assert "llm-call" in out

    def test_show_partial_id(self, capsys):
        store, _, _ = _store_with_runs()
        args = argparse.Namespace(store="./fake", run_id="run-aaa", json=False)
        with patch.object(RunStore, "create", return_value=store):
            ret = cmd_show(args)
        assert ret == 0
        assert "agent-v1" in capsys.readouterr().out

    def test_show_not_found(self, capsys):
        store, _, _ = _store_with_runs()
        args = argparse.Namespace(store="./fake", run_id="nonexistent", json=False)
        with patch.object(RunStore, "create", return_value=store):
            ret = cmd_show(args)
        assert ret == 1
        assert "not found" in capsys.readouterr().out

    def test_show_json(self, capsys):
        store, run1, _ = _store_with_runs()
        args = argparse.Namespace(store="./fake", run_id=run1.run_id, json=True)
        with patch.object(RunStore, "create", return_value=store):
            ret = cmd_show(args)
        data = json.loads(capsys.readouterr().out)
        assert ret == 0
        assert data["name"] == "agent-v1"


class TestCmdDiff:

    def test_diff_identical(self, capsys):
        store, run1, _ = _store_with_runs()
        args = argparse.Namespace(store="./fake", id1=run1.run_id, id2=run1.run_id, json=False)
        with patch.object(RunStore, "create", return_value=store):
            ret = cmd_diff(args)
        out = capsys.readouterr().out
        assert ret == 0
        assert "100.0%" in out or "No significant changes" in out

    def test_diff_changes(self, capsys):
        store, run1, run2 = _store_with_runs()
        args = argparse.Namespace(store="./fake", id1=run1.run_id, id2=run2.run_id, json=False)
        with patch.object(RunStore, "create", return_value=store):
            ret = cmd_diff(args)
        out = capsys.readouterr().out
        assert ret == 0
        assert "Similarity" in out
        assert "Output changes" in out
        assert "Step changes" in out
        assert "search" in out
        assert "Metric changes" in out

    def test_diff_json(self, capsys):
        store, run1, run2 = _store_with_runs()
        args = argparse.Namespace(store="./fake", id1=run1.run_id, id2=run2.run_id, json=True)
        with patch.object(RunStore, "create", return_value=store):
            ret = cmd_diff(args)
        data = json.loads(capsys.readouterr().out)
        assert ret == 0
        assert data["has_changes"] is True
        assert isinstance(data["steps_added"], int)
        assert isinstance(data["metrics_diff"], dict)

    def test_diff_not_found(self, capsys):
        store, run1, _ = _store_with_runs()
        args = argparse.Namespace(store="./fake", id1=run1.run_id, id2="missing", json=False)
        with patch.object(RunStore, "create", return_value=store):
            ret = cmd_diff(args)
        assert ret == 1
        assert "not found" in capsys.readouterr().out


class TestCmdReplay:

    def test_replay_no_fixtures(self, capsys):
        store, run1, _ = _store_with_runs()
        args = argparse.Namespace(store="./fake", run_id=run1.run_id, json=False)
        with patch.object(RunStore, "create", return_value=store):
            ret = cmd_replay(args)
        out = capsys.readouterr().out
        assert ret == 0
        assert "No fixtures found" in out

    def test_replay_json(self, capsys):
        store, run1, _ = _store_with_runs()
        args = argparse.Namespace(store="./fake", run_id=run1.run_id, json=True)
        with patch.object(RunStore, "create", return_value=store):
            ret = cmd_replay(args)
        data = json.loads(capsys.readouterr().out)
        assert ret == 0
        assert data["replayable"] is False
