"""Storage backends for Work Ledger.

This module provides storage interfaces and implementations:
- RunStore: Abstract base for storage backends
- MemoryStore: In-memory storage for testing
- JSONLStore: File-based JSONL storage for persistence
- SQLiteStore: SQLite database storage (zero dependencies)
- PostgresStore: PostgreSQL storage (requires psycopg2)
- RedisStore: Redis storage (requires redis)
- S3Store: AWS S3 storage (requires boto3)
- MongoDBStore: MongoDB storage (requires pymongo)
- GCSStore: Google Cloud Storage (requires google-cloud-storage)
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from work_ledger.core.models import Run, RunStatus


class RunStore(ABC):
    """Abstract base class for run storage backends.
    
    Defines the interface that all storage backends must implement.
    Storage backends are responsible for persisting and retrieving runs.
    """

    @abstractmethod
    def save_run(self, run: Run) -> None:
        """Save a run to storage.
        
        Args:
            run: The run to save
        """
        pass

    @abstractmethod
    def get_run(self, run_id: str) -> Run | None:
        """Retrieve a run by ID.
        
        Args:
            run_id: The unique identifier of the run
            
        Returns:
            The run if found, None otherwise
        """
        pass

    @abstractmethod
    def list_runs(
        self,
        name: str | None = None,
        status: RunStatus | None = None,
    ) -> list[Run]:
        """List runs with optional filtering.
        
        Args:
            name: Filter by run name (optional)
            status: Filter by run status (optional)
            
        Returns:
            List of matching runs
        """
        pass

    @abstractmethod
    def delete_run(self, run_id: str) -> None:
        """Delete a run from storage.
        
        Args:
            run_id: The unique identifier of the run to delete
        """
        pass

    @classmethod
    def create(cls, store: str | Path) -> RunStore:
        """Factory method to create appropriate storage backend.
        
        Args:
            store: Storage specification:
                   - ":memory:" for in-memory storage
                   - Path string or Path object for JSONL file storage
                   
        Returns:
            An appropriate RunStore implementation
            
        Example:
            >>> store = RunStore.create(":memory:")
            >>> store = RunStore.create("./runs")
        """
        if store == ":memory:":
            return MemoryStore()
        return JSONLStore(store)


class MemoryStore(RunStore):
    """In-memory storage backend.
    
    Stores runs in a dictionary. Useful for testing and short-lived
    processes where persistence is not required.
    
    Example:
        >>> store = MemoryStore()
        >>> store.save_run(Run(name="test"))
        >>> runs = store.list_runs()
    """

    def __init__(self) -> None:
        """Initialize the in-memory store."""
        self._runs: dict[str, Run] = {}

    def save_run(self, run: Run) -> None:
        """Save a run to memory.
        
        Args:
            run: The run to save
        """
        self._runs[run.run_id] = run

    def get_run(self, run_id: str) -> Run | None:
        """Retrieve a run by ID.
        
        Args:
            run_id: The unique identifier of the run
            
        Returns:
            The run if found, None otherwise
        """
        return self._runs.get(run_id)

    def list_runs(
        self,
        name: str | None = None,
        status: RunStatus | None = None,
    ) -> list[Run]:
        """List runs with optional filtering.
        
        Args:
            name: Filter by run name (optional)
            status: Filter by run status (optional)
            
        Returns:
            List of matching runs
        """
        runs = list(self._runs.values())
        
        if name is not None:
            runs = [r for r in runs if r.name == name]
        
        if status is not None:
            runs = [r for r in runs if r.status == status]
        
        return runs

    def delete_run(self, run_id: str) -> None:
        """Delete a run from memory.
        
        Args:
            run_id: The unique identifier of the run to delete
        """
        self._runs.pop(run_id, None)


class JSONLStore(RunStore):
    """JSONL file-based storage backend.
    
    Stores each run as a JSONL file in a directory. This provides
    human-readable persistence with easy inspection and debugging.
    
    File structure:
        {store_path}/
            {run_id}.jsonl
            {run_id}.jsonl
            ...
    
    Example:
        >>> store = JSONLStore("./runs")
        >>> store.save_run(Run(name="test"))
        >>> runs = store.list_runs()
    """

    def __init__(self, store_path: str | Path) -> None:
        """Initialize the JSONL store.
        
        Args:
            store_path: Directory path for storing run files
        """
        self._path = Path(store_path)
        self._path.mkdir(parents=True, exist_ok=True)

    def _run_file(self, run_id: str) -> Path:
        """Get the file path for a run.
        
        Args:
            run_id: The run identifier
            
        Returns:
            Path to the run's JSONL file
        """
        return self._path / f"{run_id}.jsonl"

    def save_run(self, run: Run) -> None:
        """Save a run to a JSONL file.
        
        Args:
            run: The run to save
        """
        run_file = self._run_file(run.run_id)
        with open(run_file, "w") as f:
            json.dump(run.to_dict(), f)
            f.write("\n")

    def get_run(self, run_id: str) -> Run | None:
        """Retrieve a run by ID.
        
        Args:
            run_id: The unique identifier of the run
            
        Returns:
            The run if found, None otherwise
        """
        run_file = self._run_file(run_id)
        if not run_file.exists():
            return None
        
        with open(run_file) as f:
            data = json.loads(f.readline())
        
        return Run.from_dict(data)

    def list_runs(
        self,
        name: str | None = None,
        status: RunStatus | None = None,
    ) -> list[Run]:
        """List runs with optional filtering.
        
        Args:
            name: Filter by run name (optional)
            status: Filter by run status (optional)
            
        Returns:
            List of matching runs
        """
        runs: list[Run] = []
        
        for run_file in self._path.glob("*.jsonl"):
            run_id = run_file.stem
            run = self.get_run(run_id)
            if run is not None:
                runs.append(run)
        
        if name is not None:
            runs = [r for r in runs if r.name == name]
        
        if status is not None:
            runs = [r for r in runs if r.status == status]
        
        return runs

    def delete_run(self, run_id: str) -> None:
        """Delete a run from storage.
        
        Args:
            run_id: The unique identifier of the run to delete
        """
        run_file = self._run_file(run_id)
        if run_file.exists():
            run_file.unlink()


class SQLiteStore(RunStore):
    """SQLite database storage backend.
    
    Stores runs in a SQLite database. Zero external dependencies,
    great for single-file persistence with query capabilities.
    
    Example:
        >>> store = SQLiteStore("./runs.db")
        >>> store.save_run(Run(name="test"))
        >>> runs = store.list_runs()
    """

    def __init__(self, db_path: str | Path) -> None:
        """Initialize the SQLite store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        import sqlite3
        self._db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._create_table()

    def _create_table(self) -> None:
        """Create the runs table if it doesn't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                data TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON runs(name)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON runs(status)")
        self._conn.commit()

    def save_run(self, run: Run) -> None:
        """Save a run to SQLite."""
        self._conn.execute(
            "INSERT OR REPLACE INTO runs (run_id, name, status, data) VALUES (?, ?, ?, ?)",
            (run.run_id, run.name, run.status.value, json.dumps(run.to_dict()))
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> Run | None:
        """Retrieve a run by ID."""
        cursor = self._conn.execute(
            "SELECT data FROM runs WHERE run_id = ?", (run_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return Run.from_dict(json.loads(row[0]))

    def list_runs(
        self,
        name: str | None = None,
        status: RunStatus | None = None,
    ) -> list[Run]:
        """List runs with optional filtering."""
        query = "SELECT data FROM runs WHERE 1=1"
        params: list[Any] = []
        
        if name is not None:
            query += " AND name = ?"
            params.append(name)
        
        if status is not None:
            query += " AND status = ?"
            params.append(status.value)
        
        cursor = self._conn.execute(query, params)
        return [Run.from_dict(json.loads(row[0])) for row in cursor.fetchall()]

    def delete_run(self, run_id: str) -> None:
        """Delete a run from SQLite."""
        self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


class PostgresStore(RunStore):
    """PostgreSQL database storage backend.
    
    Stores runs in a PostgreSQL database. Great for team/production use
    with full SQL query capabilities.
    
    Requires: pip install psycopg2-binary
    
    Example:
        >>> store = PostgresStore("postgresql://user:pass@localhost/workledger")
        >>> store.save_run(Run(name="test"))
        >>> runs = store.list_runs()
    """

    def __init__(self, connection_string: str) -> None:
        """Initialize the PostgreSQL store.
        
        Args:
            connection_string: PostgreSQL connection string
        """
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "PostgresStore requires psycopg2. Install with: pip install psycopg2-binary"
            )
        
        self._conn = psycopg2.connect(connection_string)
        self._create_table()

    def _create_table(self) -> None:
        """Create the runs table if it doesn't exist."""
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    name TEXT,
                    status TEXT,
                    data JSONB
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_name ON runs(name)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")
        self._conn.commit()

    def save_run(self, run: Run) -> None:
        """Save a run to PostgreSQL."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (run_id, name, status, data) 
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    status = EXCLUDED.status,
                    data = EXCLUDED.data
                """,
                (run.run_id, run.name, run.status.value, json.dumps(run.to_dict()))
            )
        self._conn.commit()

    def get_run(self, run_id: str) -> Run | None:
        """Retrieve a run by ID."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT data FROM runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
        if row is None:
            return None
        return Run.from_dict(row[0] if isinstance(row[0], dict) else json.loads(row[0]))

    def list_runs(
        self,
        name: str | None = None,
        status: RunStatus | None = None,
    ) -> list[Run]:
        """List runs with optional filtering."""
        query = "SELECT data FROM runs WHERE TRUE"
        params: list[Any] = []
        
        if name is not None:
            query += " AND name = %s"
            params.append(name)
        
        if status is not None:
            query += " AND status = %s"
            params.append(status.value)
        
        with self._conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        
        return [
            Run.from_dict(row[0] if isinstance(row[0], dict) else json.loads(row[0]))
            for row in rows
        ]

    def delete_run(self, run_id: str) -> None:
        """Delete a run from PostgreSQL."""
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()


class RedisStore(RunStore):
    """Redis storage backend.
    
    Stores runs in Redis. Great for fast, ephemeral storage or
    distributed systems. Supports TTL for automatic expiration.
    
    Requires: pip install redis
    
    Example:
        >>> store = RedisStore("redis://localhost:6379/0")
        >>> store.save_run(Run(name="test"))
        >>> runs = store.list_runs()
    """

    def __init__(
        self, 
        url: str = "redis://localhost:6379/0",
        prefix: str = "workledger:",
        ttl: int | None = None,
    ) -> None:
        """Initialize the Redis store.
        
        Args:
            url: Redis connection URL
            prefix: Key prefix for all runs
            ttl: Optional TTL in seconds for automatic expiration
        """
        try:
            import redis
        except ImportError:
            raise ImportError(
                "RedisStore requires redis. Install with: pip install redis"
            )
        
        self._client = redis.from_url(url)
        self._prefix = prefix
        self._ttl = ttl

    def _key(self, run_id: str) -> str:
        """Get the Redis key for a run."""
        return f"{self._prefix}{run_id}"

    def save_run(self, run: Run) -> None:
        """Save a run to Redis."""
        key = self._key(run.run_id)
        data = json.dumps(run.to_dict())
        if self._ttl:
            self._client.setex(key, self._ttl, data)
        else:
            self._client.set(key, data)
        # Also add to index sets for filtering
        self._client.sadd(f"{self._prefix}_all", run.run_id)
        self._client.sadd(f"{self._prefix}_name:{run.name}", run.run_id)
        self._client.sadd(f"{self._prefix}_status:{run.status.value}", run.run_id)

    def get_run(self, run_id: str) -> Run | None:
        """Retrieve a run by ID."""
        data = self._client.get(self._key(run_id))
        if data is None:
            return None
        return Run.from_dict(json.loads(data))

    def list_runs(
        self,
        name: str | None = None,
        status: RunStatus | None = None,
    ) -> list[Run]:
        """List runs with optional filtering."""
        # Get candidate run IDs
        if name is not None and status is not None:
            run_ids = self._client.sinter(
                f"{self._prefix}_name:{name}",
                f"{self._prefix}_status:{status.value}"
            )
        elif name is not None:
            run_ids = self._client.smembers(f"{self._prefix}_name:{name}")
        elif status is not None:
            run_ids = self._client.smembers(f"{self._prefix}_status:{status.value}")
        else:
            run_ids = self._client.smembers(f"{self._prefix}_all")
        
        runs = []
        for run_id in run_ids:
            run_id_str = run_id.decode() if isinstance(run_id, bytes) else run_id
            run = self.get_run(run_id_str)
            if run is not None:
                runs.append(run)
        return runs

    def delete_run(self, run_id: str) -> None:
        """Delete a run from Redis."""
        run = self.get_run(run_id)
        if run:
            self._client.delete(self._key(run_id))
            self._client.srem(f"{self._prefix}_all", run_id)
            self._client.srem(f"{self._prefix}_name:{run.name}", run_id)
            self._client.srem(f"{self._prefix}_status:{run.status.value}", run_id)


class S3Store(RunStore):
    """AWS S3 storage backend.
    
    Stores runs as JSON objects in S3. Great for cloud-native
    deployments and long-term archival.
    
    Requires: pip install boto3
    
    Example:
        >>> store = S3Store("my-bucket", prefix="runs/")
        >>> store.save_run(Run(name="test"))
        >>> runs = store.list_runs()
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "runs/",
        region: str | None = None,
    ) -> None:
        """Initialize the S3 store.
        
        Args:
            bucket: S3 bucket name
            prefix: Key prefix for all runs
            region: AWS region (optional, uses default if not specified)
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "S3Store requires boto3. Install with: pip install boto3"
            )
        
        self._bucket = bucket
        self._prefix = prefix
        self._s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")

    def _key(self, run_id: str) -> str:
        """Get the S3 key for a run."""
        return f"{self._prefix}{run_id}.json"

    def save_run(self, run: Run) -> None:
        """Save a run to S3."""
        self._s3.put_object(
            Bucket=self._bucket,
            Key=self._key(run.run_id),
            Body=json.dumps(run.to_dict()),
            ContentType="application/json",
            Metadata={
                "name": run.name,
                "status": run.status.value,
            }
        )

    def get_run(self, run_id: str) -> Run | None:
        """Retrieve a run by ID."""
        try:
            response = self._s3.get_object(
                Bucket=self._bucket,
                Key=self._key(run_id)
            )
            data = json.loads(response["Body"].read().decode())
            return Run.from_dict(data)
        except self._s3.exceptions.NoSuchKey:
            return None
        except Exception:
            return None

    def list_runs(
        self,
        name: str | None = None,
        status: RunStatus | None = None,
    ) -> list[Run]:
        """List runs with optional filtering.
        
        Note: For large buckets, consider using S3 Select or
        maintaining a separate index for efficient queries.
        """
        runs = []
        paginator = self._s3.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".json"):
                    continue
                
                run_id = key[len(self._prefix):-5]  # Remove prefix and .json
                run = self.get_run(run_id)
                
                if run is None:
                    continue
                if name is not None and run.name != name:
                    continue
                if status is not None and run.status != status:
                    continue
                
                runs.append(run)
        
        return runs

    def delete_run(self, run_id: str) -> None:
        """Delete a run from S3."""
        self._s3.delete_object(
            Bucket=self._bucket,
            Key=self._key(run_id)
        )


class MongoDBStore(RunStore):
    """MongoDB storage backend.
    
    Stores runs as documents in MongoDB. Great for document-based
    storage with rich query capabilities.
    
    Requires: pip install pymongo
    
    Example:
        >>> store = MongoDBStore("mongodb://localhost:27017", database="workledger")
        >>> store.save_run(Run(name="test"))
        >>> runs = store.list_runs()
    """

    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017",
        database: str = "workledger",
        collection: str = "runs",
    ) -> None:
        """Initialize the MongoDB store.
        
        Args:
            connection_string: MongoDB connection URI
            database: Database name
            collection: Collection name for runs
        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError(
                "MongoDBStore requires pymongo. Install with: pip install pymongo"
            )
        
        self._client = MongoClient(connection_string)
        self._db = self._client[database]
        self._collection = self._db[collection]
        
        # Create indexes for efficient queries
        self._collection.create_index("run_id", unique=True)
        self._collection.create_index("name")
        self._collection.create_index("status")

    def save_run(self, run: Run) -> None:
        """Save a run to MongoDB."""
        doc = run.to_dict()
        doc["_id"] = run.run_id  # Use run_id as MongoDB _id
        self._collection.replace_one(
            {"_id": run.run_id},
            doc,
            upsert=True
        )

    def get_run(self, run_id: str) -> Run | None:
        """Retrieve a run by ID."""
        doc = self._collection.find_one({"_id": run_id})
        if doc is None:
            return None
        doc.pop("_id", None)  # Remove MongoDB _id before conversion
        return Run.from_dict(doc)

    def list_runs(
        self,
        name: str | None = None,
        status: RunStatus | None = None,
    ) -> list[Run]:
        """List runs with optional filtering."""
        query: dict = {}
        
        if name is not None:
            query["name"] = name
        
        if status is not None:
            query["status"] = status.value
        
        runs = []
        for doc in self._collection.find(query):
            doc.pop("_id", None)
            runs.append(Run.from_dict(doc))
        
        return runs

    def delete_run(self, run_id: str) -> None:
        """Delete a run from MongoDB."""
        self._collection.delete_one({"_id": run_id})

    def close(self) -> None:
        """Close the MongoDB connection."""
        self._client.close()


class GCSStore(RunStore):
    """Google Cloud Storage backend.
    
    Stores runs as JSON objects in GCS. Great for GCP-native
    deployments and long-term archival.
    
    Requires: pip install google-cloud-storage
    
    Example:
        >>> store = GCSStore("my-bucket", prefix="runs/")
        >>> store.save_run(Run(name="test"))
        >>> runs = store.list_runs()
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "runs/",
        project: str | None = None,
    ) -> None:
        """Initialize the GCS store.
        
        Args:
            bucket: GCS bucket name
            prefix: Key prefix for all runs
            project: GCP project ID (optional, uses default if not specified)
        """
        try:
            from google.cloud import storage
        except ImportError:
            raise ImportError(
                "GCSStore requires google-cloud-storage. "
                "Install with: pip install google-cloud-storage"
            )
        
        self._bucket_name = bucket
        self._prefix = prefix
        self._client = storage.Client(project=project) if project else storage.Client()
        self._bucket = self._client.bucket(bucket)

    def _blob_name(self, run_id: str) -> str:
        """Get the GCS blob name for a run."""
        return f"{self._prefix}{run_id}.json"

    def save_run(self, run: Run) -> None:
        """Save a run to GCS."""
        blob = self._bucket.blob(self._blob_name(run.run_id))
        blob.upload_from_string(
            json.dumps(run.to_dict()),
            content_type="application/json"
        )
        # Set metadata for filtering
        blob.metadata = {
            "name": run.name,
            "status": run.status.value,
        }
        blob.patch()

    def get_run(self, run_id: str) -> Run | None:
        """Retrieve a run by ID."""
        blob = self._bucket.blob(self._blob_name(run_id))
        
        if not blob.exists():
            return None
        
        data = json.loads(blob.download_as_string())
        return Run.from_dict(data)

    def list_runs(
        self,
        name: str | None = None,
        status: RunStatus | None = None,
    ) -> list[Run]:
        """List runs with optional filtering.
        
        Note: For large buckets, consider maintaining a separate
        index for efficient queries.
        """
        runs = []
        blobs = self._client.list_blobs(self._bucket_name, prefix=self._prefix)
        
        for blob in blobs:
            if not blob.name.endswith(".json"):
                continue
            
            run_id = blob.name[len(self._prefix):-5]  # Remove prefix and .json
            run = self.get_run(run_id)
            
            if run is None:
                continue
            if name is not None and run.name != name:
                continue
            if status is not None and run.status != status:
                continue
            
            runs.append(run)
        
        return runs

    def delete_run(self, run_id: str) -> None:
        """Delete a run from GCS."""
        blob = self._bucket.blob(self._blob_name(run_id))
        blob.delete()
