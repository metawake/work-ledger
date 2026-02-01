# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-02-01

### Added

#### Core
- `WorkLedger` class for recording agent runs
- `Run` and `Step` data models with full serialization
- `Metrics` for token usage, latency, and cost tracking
- `CausalLink` for explicit causality modeling

#### Storage Backends
- `MemoryStore` - in-memory storage for testing
- `JSONLStore` - file-based JSONL storage
- `SQLiteStore` - single-file database (no dependencies)
- `PostgresStore` - production SQL (requires psycopg2)
- `RedisStore` - fast ephemeral with TTL (requires redis)
- `S3Store` - AWS cloud storage (requires boto3)
- `MongoDBStore` - document storage (requires pymongo)
- `GCSStore` - Google Cloud Storage (requires google-cloud-storage)

#### Framework Integrations
- `wrap_agent()` - PydanticAI integration with replay support
- `wrap_graph()` - LangGraph integration
- `wrap_crew()` - CrewAI integration
- `wrap_chain()` - LangChain integration
- `wrap_query_engine()` - LlamaIndex integration
- `wrap_openai()` - OpenAI SDK integration with replay support
- `wrap_anthropic()` - Anthropic SDK integration with replay support

#### Replay
- Fixture capture on record - API responses saved automatically
- `replay_from` parameter on wrappers - replay without API calls
- `ReplayError` for divergence detection
- CLI `work-ledger replay` command

#### Testing Module
- `Fixture` and `Recording` for captured executions
- `RunDiff` and `StepDiff` for comparing runs
- `assert_run_matches`, `assert_no_regression` assertions
- `@recorded`, `@replay`, `@golden`, `@compare` decorators

#### CLI
- `work-ledger list` - list recorded runs
- `work-ledger show` - display run details
- `work-ledger diff` - compare two runs
- `work-ledger replay` - show replay info for a run

### Notes

This is an alpha release. The API may change in future versions.
