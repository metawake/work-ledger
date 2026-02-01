# Storage Backends

Work Ledger supports 8 storage backends for different use cases.

## Overview

| Backend | Install | Use Case |
|---------|---------|----------|
| **Memory** | built-in | Testing |
| **JSONL** | built-in | Local development |
| **SQLite** | built-in | Single-file persistence |
| **PostgreSQL** | `psycopg2-binary` | Team/production |
| **Redis** | `redis` | Fast ephemeral |
| **MongoDB** | `pymongo` | Document storage |
| **S3** | `boto3` | AWS cloud |
| **GCS** | `google-cloud-storage` | GCP cloud |

## Usage

```python
from work_ledger import (
    WorkLedger, 
    MemoryStore, JSONLStore, SQLiteStore,
    PostgresStore, RedisStore, S3Store,
    MongoDBStore, GCSStore
)
```

### Memory (Testing)

```python
ledger = WorkLedger(store=":memory:")
```

Runs are lost when the process exits. Great for tests.

### JSONL (Local Development)

```python
ledger = WorkLedger(store="./runs")
```

Creates a `./runs/` directory with one `.jsonl` file per run. Human-readable.

### SQLite (Single File)

```python
ledger = WorkLedger(store=SQLiteStore("./runs.db"))
```

Zero dependencies. Queryable. Good for local persistence.

### PostgreSQL (Production)

```bash
pip install psycopg2-binary
```

```python
ledger = WorkLedger(store=PostgresStore(
    "postgresql://user:pass@localhost/workledger"
))
```

Full SQL queries. Team collaboration. Production-ready.

### Redis (Fast Ephemeral)

```bash
pip install redis
```

```python
ledger = WorkLedger(store=RedisStore(
    "redis://localhost:6379/0",
    ttl=86400  # Optional: expire after 24 hours
))
```

Fast. Supports TTL for automatic cleanup.

### MongoDB (Document Storage)

```bash
pip install pymongo
```

```python
ledger = WorkLedger(store=MongoDBStore(
    "mongodb://localhost:27017",
    database="workledger"
))
```

Native JSON document storage. Rich queries.

### S3 (AWS)

```bash
pip install boto3
```

```python
ledger = WorkLedger(store=S3Store(
    "my-bucket",
    prefix="runs/"
))
```

Uses AWS credentials from environment. Good for archival.

### GCS (Google Cloud)

```bash
pip install google-cloud-storage
```

```python
ledger = WorkLedger(store=GCSStore(
    "my-bucket",
    prefix="runs/"
))
```

Uses GCP credentials from environment.

## Custom Storage

Implement the `RunStore` interface:

```python
from work_ledger.core.store import RunStore

class MyStore(RunStore):
    def save_run(self, run: Run) -> None:
        ...
    
    def get_run(self, run_id: str) -> Run | None:
        ...
    
    def list_runs(self, name=None, status=None) -> list[Run]:
        ...
    
    def delete_run(self, run_id: str) -> None:
        ...
```

## Installation

Storage backends are optional dependencies:

```bash
pip install work-ledger[postgres]
pip install work-ledger[redis]
pip install work-ledger[mongodb]
pip install work-ledger[s3]
pip install work-ledger[gcs]

# All storage backends
pip install work-ledger[storage]
```
