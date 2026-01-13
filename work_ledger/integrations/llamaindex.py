"""LlamaIndex integration for Work Ledger.

Thin wrappers that record LlamaIndex query and chat engine executions.

Example:
    >>> from llama_index.core import VectorStoreIndex
    >>> from work_ledger import WorkLedger
    >>> from work_ledger.integrations.llamaindex import wrap_query_engine
    >>> 
    >>> index = VectorStoreIndex.from_documents(documents)
    >>> engine = index.as_query_engine()
    >>> 
    >>> ledger = WorkLedger(store="./runs")
    >>> wrapped = wrap_query_engine(engine, ledger)
    >>> 
    >>> response = wrapped.query("What is machine learning?")
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from work_ledger.core.ledger import WorkLedger
from work_ledger.core.models import Run, Step, StepKind, RunStatus

if TYPE_CHECKING:
    pass


class WrappedQueryEngine:
    """Wrapper around a LlamaIndex QueryEngine that records runs.
    
    Supports fluent interface for configuration:
        >>> wrapped = wrap_query_engine(engine, ledger).with_name("my-rag")
    """

    def __init__(
        self,
        engine: Any,
        ledger: WorkLedger,
        run_name: str | None = None,
    ) -> None:
        self._engine = engine
        self._ledger = ledger
        self._run_name = run_name or getattr(engine, "name", None) or "llamaindex-query"

    def with_name(self, name: str) -> "WrappedQueryEngine":
        """Set a custom name for recorded runs.
        
        Args:
            name: Human-readable name for runs
            
        Returns:
            Self for method chaining
            
        Example:
            >>> wrapped = wrap_query_engine(engine, ledger).with_name("docs-rag")
        """
        self._run_name = name
        return self

    def query(self, query: str) -> Any:
        """Query and record the run."""
        run = self._create_run(query)
        result = None
        error = None
        
        try:
            result = self._engine.query(query)
            run.status = RunStatus.SUCCESS
            run.outputs = {"response": str(result)}
            self._record_source_nodes(run, result)
        except Exception as e:
            run.status = RunStatus.FAILED
            run.annotations["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            error = e
        finally:
            run.ended_at = datetime.now(timezone.utc)
            self._ledger._save_run(run)
        
        if error:
            raise error
        return result

    async def aquery(self, query: str) -> Any:
        """Async query and record the run."""
        run = self._create_run(query)
        result = None
        error = None
        
        try:
            result = await self._engine.aquery(query)
            run.status = RunStatus.SUCCESS
            run.outputs = {"response": str(result)}
            self._record_source_nodes(run, result)
        except Exception as e:
            run.status = RunStatus.FAILED
            run.annotations["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            error = e
        finally:
            run.ended_at = datetime.now(timezone.utc)
            self._ledger._save_run(run)
        
        if error:
            raise error
        return result

    def _create_run(self, query: str) -> Run:
        run = Run(name=self._run_name)
        run.started_at = datetime.now(timezone.utc)
        run.status = RunStatus.RUNNING
        run.inputs = {"query": query}
        return run

    def _record_source_nodes(self, run: Run, result: Any) -> None:
        """Record retrieved nodes as a retrieval step."""
        source_nodes = getattr(result, "source_nodes", [])
        
        if source_nodes:
            step = Step(name="retrieve-nodes", kind=StepKind.RETRIEVAL)
            step.started_at = datetime.now(timezone.utc)
            step.ended_at = datetime.now(timezone.utc)
            
            nodes = []
            for node in source_nodes:
                node_info = {
                    "id": getattr(node, "node_id", str(id(node))),
                    "score": getattr(node, "score", None),
                }
                if hasattr(node, "get_content"):
                    content = node.get_content()
                    node_info["content"] = content[:200] if len(content) > 200 else content
                elif hasattr(node, "text"):
                    content = node.text
                    node_info["content"] = content[:200] if len(content) > 200 else content
                nodes.append(node_info)
            
            step.outputs = {"nodes": nodes, "count": len(nodes)}
            run.add_step(step)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


class WrappedChatEngine:
    """Wrapper around a LlamaIndex ChatEngine that records runs.
    
    Supports fluent interface for configuration:
        >>> wrapped = wrap_chat_engine(engine, ledger).with_name("my-chat")
    """

    def __init__(
        self,
        engine: Any,
        ledger: WorkLedger,
        run_name: str | None = None,
    ) -> None:
        self._engine = engine
        self._ledger = ledger
        self._run_name = run_name or getattr(engine, "name", None) or "llamaindex-chat"

    def with_name(self, name: str) -> "WrappedChatEngine":
        """Set a custom name for recorded runs.
        
        Args:
            name: Human-readable name for runs
            
        Returns:
            Self for method chaining
            
        Example:
            >>> wrapped = wrap_chat_engine(engine, ledger).with_name("support-chat")
        """
        self._run_name = name
        return self

    def chat(self, message: str) -> Any:
        """Chat and record the run."""
        run = self._create_run(message)
        result = None
        error = None
        
        try:
            result = self._engine.chat(message)
            run.status = RunStatus.SUCCESS
            run.outputs = {"response": str(result)}
        except Exception as e:
            run.status = RunStatus.FAILED
            run.annotations["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            error = e
        finally:
            run.ended_at = datetime.now(timezone.utc)
            self._ledger._save_run(run)
        
        if error:
            raise error
        return result

    async def achat(self, message: str) -> Any:
        """Async chat and record the run."""
        run = self._create_run(message)
        result = None
        error = None
        
        try:
            result = await self._engine.achat(message)
            run.status = RunStatus.SUCCESS
            run.outputs = {"response": str(result)}
        except Exception as e:
            run.status = RunStatus.FAILED
            run.annotations["error"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            error = e
        finally:
            run.ended_at = datetime.now(timezone.utc)
            self._ledger._save_run(run)
        
        if error:
            raise error
        return result

    def _create_run(self, message: str) -> Run:
        run = Run(name=self._run_name)
        run.started_at = datetime.now(timezone.utc)
        run.status = RunStatus.RUNNING
        run.inputs = {"message": message}
        return run

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


def wrap_query_engine(
    engine: Any,
    ledger: WorkLedger,
    run_name: str | None = None,
) -> WrappedQueryEngine:
    """Wrap a LlamaIndex QueryEngine to record runs."""
    return WrappedQueryEngine(engine, ledger, run_name)


def wrap_chat_engine(
    engine: Any,
    ledger: WorkLedger,
    run_name: str | None = None,
) -> WrappedChatEngine:
    """Wrap a LlamaIndex ChatEngine to record runs."""
    return WrappedChatEngine(engine, ledger, run_name)
