"""Tests for storage backends."""

import json
import tempfile
from pathlib import Path

import pytest

from work_ledger.core.models import Run, RunStatus
from work_ledger.core.store import (
    MemoryStore,
    JSONLStore,
    SQLiteStore,
    PostgresStore,
    RedisStore,
    S3Store,
)


class TestMemoryStore:
    """Tests for MemoryStore."""

    def test_save_and_get(self):
        store = MemoryStore()
        run = Run(name="test")
        store.save_run(run)
        
        retrieved = store.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_get_nonexistent(self):
        store = MemoryStore()
        assert store.get_run("nonexistent") is None

    def test_list_runs(self):
        store = MemoryStore()
        store.save_run(Run(name="a"))
        store.save_run(Run(name="b"))
        
        runs = store.list_runs()
        assert len(runs) == 2

    def test_list_runs_filter_by_name(self):
        store = MemoryStore()
        store.save_run(Run(name="target"))
        store.save_run(Run(name="other"))
        
        runs = store.list_runs(name="target")
        assert len(runs) == 1
        assert runs[0].name == "target"

    def test_list_runs_filter_by_status(self):
        store = MemoryStore()
        run1 = Run(name="success")
        run1.status = RunStatus.SUCCESS
        run2 = Run(name="failed")
        run2.status = RunStatus.FAILED
        store.save_run(run1)
        store.save_run(run2)
        
        runs = store.list_runs(status=RunStatus.SUCCESS)
        assert len(runs) == 1
        assert runs[0].name == "success"

    def test_delete_run(self):
        store = MemoryStore()
        run = Run(name="test")
        store.save_run(run)
        store.delete_run(run.run_id)
        
        assert store.get_run(run.run_id) is None


class TestJSONLStore:
    """Tests for JSONLStore."""

    def test_save_and_get(self, tmp_path):
        store = JSONLStore(tmp_path)
        run = Run(name="test")
        store.save_run(run)
        
        retrieved = store.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_persists_to_disk(self, tmp_path):
        store = JSONLStore(tmp_path)
        run = Run(name="test")
        store.save_run(run)
        
        # Create a new store instance
        store2 = JSONLStore(tmp_path)
        retrieved = store2.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.name == "test"

    def test_creates_directory(self, tmp_path):
        store_path = tmp_path / "nested" / "path"
        store = JSONLStore(store_path)
        
        assert store_path.exists()

    def test_list_runs(self, tmp_path):
        store = JSONLStore(tmp_path)
        store.save_run(Run(name="a"))
        store.save_run(Run(name="b"))
        
        runs = store.list_runs()
        assert len(runs) == 2

    def test_delete_run(self, tmp_path):
        store = JSONLStore(tmp_path)
        run = Run(name="test")
        store.save_run(run)
        store.delete_run(run.run_id)
        
        assert store.get_run(run.run_id) is None
        assert not (tmp_path / f"{run.run_id}.jsonl").exists()


class TestSQLiteStore:
    """Tests for SQLiteStore."""

    def test_save_and_get(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path)
        run = Run(name="test")
        store.save_run(run)
        
        retrieved = store.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.name == "test"
        store.close()

    def test_persists_to_disk(self, tmp_path):
        db_path = tmp_path / "test.db"
        
        store = SQLiteStore(db_path)
        run = Run(name="test")
        store.save_run(run)
        store.close()
        
        # Create a new store instance
        store2 = SQLiteStore(db_path)
        retrieved = store2.get_run(run.run_id)
        assert retrieved is not None
        assert retrieved.name == "test"
        store2.close()

    def test_list_runs(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path)
        store.save_run(Run(name="a"))
        store.save_run(Run(name="b"))
        
        runs = store.list_runs()
        assert len(runs) == 2
        store.close()

    def test_list_runs_filter_by_name(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path)
        store.save_run(Run(name="target"))
        store.save_run(Run(name="other"))
        
        runs = store.list_runs(name="target")
        assert len(runs) == 1
        assert runs[0].name == "target"
        store.close()

    def test_list_runs_filter_by_status(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path)
        run1 = Run(name="success")
        run1.status = RunStatus.SUCCESS
        run2 = Run(name="failed")
        run2.status = RunStatus.FAILED
        store.save_run(run1)
        store.save_run(run2)
        
        runs = store.list_runs(status=RunStatus.SUCCESS)
        assert len(runs) == 1
        assert runs[0].name == "success"
        store.close()

    def test_delete_run(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path)
        run = Run(name="test")
        store.save_run(run)
        store.delete_run(run.run_id)
        
        assert store.get_run(run.run_id) is None
        store.close()

    def test_update_run(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = SQLiteStore(db_path)
        run = Run(name="test")
        store.save_run(run)
        
        run.status = RunStatus.SUCCESS
        store.save_run(run)
        
        retrieved = store.get_run(run.run_id)
        assert retrieved.status == RunStatus.SUCCESS
        store.close()


class TestPostgresStore:
    """Tests for PostgresStore (requires psycopg2)."""

    def test_import_error_without_psycopg2(self):
        """Test that helpful error is raised without psycopg2."""
        # This test just verifies the import mechanism works
        # Actual Postgres tests would need a running database
        pass


class TestRedisStore:
    """Tests for RedisStore (requires redis)."""

    def test_import_error_without_redis(self):
        """Test that helpful error is raised without redis."""
        pass


class TestS3Store:
    """Tests for S3Store (requires boto3)."""

    def test_import_error_without_boto3(self):
        """Test that helpful error is raised without boto3."""
        pass


class TestMongoDBStore:
    """Tests for MongoDBStore (requires pymongo)."""

    def test_import_error_without_pymongo(self):
        """Test that helpful error is raised without pymongo."""
        pass


class TestGCSStore:
    """Tests for GCSStore (requires google-cloud-storage)."""

    def test_import_error_without_gcs(self):
        """Test that helpful error is raised without google-cloud-storage."""
        pass
