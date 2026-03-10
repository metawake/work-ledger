"""LangChain integration for Work Ledger.

Provides two integration styles:

1. **Wrapper** (simple): ``wrap_chain`` / ``wrap_agent`` — wraps invoke calls.
2. **Callback handler** (idiomatic): ``WorkLedgerCallbackHandler`` —
   plugs into LangChain's callback system for granular step-level capture.

Callback handler example:
    >>> from langchain_openai import ChatOpenAI
    >>> from work_ledger import WorkLedger
    >>> from work_ledger.integrations.langchain import WorkLedgerCallbackHandler
    >>>
    >>> ledger = WorkLedger(store="./runs")
    >>> handler = WorkLedgerCallbackHandler(ledger)
    >>> llm = ChatOpenAI(callbacks=[handler])
    >>> llm.invoke("What is AI?")
    >>> handler.get_run()  # completed Run with steps

Wrapper example:
    >>> from work_ledger.integrations.langchain import wrap_chain
    >>> wrapped = wrap_chain(chain, ledger)
    >>> result = wrapped.invoke({"question": "What is AI?"})
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING
from uuid import UUID

from work_ledger.core.ledger import WorkLedger
from work_ledger.core.models import Run, Step, StepKind, RunStatus, Metrics

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult


class WorkLedgerCallbackHandler:
    """LangChain BaseCallbackHandler that records runs and steps to WorkLedger.

    Captures LLM calls, tool invocations, chain executions, and retriever
    queries as structured Steps within a single Run. Attach it via the
    ``callbacks`` parameter supported by all LangChain runnables.

    Args:
        ledger: WorkLedger instance for persistence.
        run_name: Human-readable name for the recorded run.
        auto_save: If True (default), persist the run when the outermost
            chain/LLM finishes.  Set to False to call ``save()`` manually.

    Example:
        >>> handler = WorkLedgerCallbackHandler(ledger, run_name="qa")
        >>> chain.invoke({"question": "hi"}, config={"callbacks": [handler]})
        >>> run = handler.get_run()
    """

    def __init__(
        self,
        ledger: WorkLedger,
        run_name: str = "langchain",
        auto_save: bool = True,
    ) -> None:
        self._ledger = ledger
        self._run = Run(name=run_name)
        self._run.status = RunStatus.RUNNING
        self._run.started_at = datetime.now(timezone.utc)
        self._auto_save = auto_save
        self._saved = False
        self._pending_steps: dict[UUID, Step] = {}
        self._depth = 0

    # -- public API ----------------------------------------------------------

    def get_run(self) -> Run:
        """Return the recorded Run (saving first if needed)."""
        if not self._saved:
            self.save()
        return self._run

    def save(self) -> None:
        """Persist the run to the ledger's store."""
        if self._saved:
            return
        self._run.ended_at = datetime.now(timezone.utc)
        if self._run.status == RunStatus.RUNNING:
            self._run.status = RunStatus.SUCCESS
        self._run.metrics = self._run.aggregate_metrics()
        self._ledger._save_run(self._run)
        self._saved = True

    # -- LLM callbacks -------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "llm"
        step = Step(name=name, kind=StepKind.MODEL)
        step.started_at = datetime.now(timezone.utc)
        step.inputs = {"prompts": prompts}
        if parent_run_id:
            parent = self._pending_steps.get(parent_run_id)
            if parent:
                step.caused_by = parent.step_id
        self._pending_steps[run_id] = step

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "chat_model"
        step = Step(name=name, kind=StepKind.MODEL)
        step.started_at = datetime.now(timezone.utc)
        step.inputs = {
            "messages": [
                [{"role": getattr(m, "type", "unknown"), "content": str(m.content)} for m in batch]
                for batch in messages
            ]
        }
        if parent_run_id:
            parent = self._pending_steps.get(parent_run_id)
            if parent:
                step.caused_by = parent.step_id
        self._pending_steps[run_id] = step

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        step = self._pending_steps.pop(run_id, None)
        if step is None:
            return
        step.ended_at = datetime.now(timezone.utc)
        generations = response.generations
        if generations:
            step.outputs = {
                "generations": [
                    [{"text": g.text} for g in batch]
                    for batch in generations
                ]
            }
        usage = (response.llm_output or {}).get("token_usage", {})
        step.metrics = Metrics(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
        self._run.add_step(step)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        step = self._pending_steps.pop(run_id, None)
        if step is None:
            return
        step.ended_at = datetime.now(timezone.utc)
        step.annotations["error"] = {"type": type(error).__name__, "message": str(error)}
        self._run.add_step(step)
        self._run.status = RunStatus.FAILED

    # -- Tool callbacks -------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name") or serialized.get("id", ["tool"])[-1] if serialized else "tool"
        step = Step(name=name, kind=StepKind.TOOL)
        step.started_at = datetime.now(timezone.utc)
        step.inputs = {"input": input_str}
        if parent_run_id:
            parent = self._pending_steps.get(parent_run_id)
            if parent:
                step.caused_by = parent.step_id
        self._pending_steps[run_id] = step

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        step = self._pending_steps.pop(run_id, None)
        if step is None:
            return
        step.ended_at = datetime.now(timezone.utc)
        step.outputs = {"output": output}
        self._run.add_step(step)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        step = self._pending_steps.pop(run_id, None)
        if step is None:
            return
        step.ended_at = datetime.now(timezone.utc)
        step.annotations["error"] = {"type": type(error).__name__, "message": str(error)}
        self._run.add_step(step)

    # -- Chain callbacks ------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._depth += 1
        if self._depth == 1:
            self._run.inputs = inputs if isinstance(inputs, dict) else {"input": str(inputs)}

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if self._depth == 1:
            self._run.outputs = outputs if isinstance(outputs, dict) else {"output": str(outputs)}
            if self._auto_save:
                self.save()
        self._depth = max(0, self._depth - 1)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._run.status = RunStatus.FAILED
        self._run.annotations["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        if self._depth == 1 and self._auto_save:
            self.save()
        self._depth = max(0, self._depth - 1)

    # -- Retriever callbacks --------------------------------------------------

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "retriever") if serialized else "retriever"
        step = Step(name=name, kind=StepKind.RETRIEVAL)
        step.started_at = datetime.now(timezone.utc)
        step.inputs = {"query": query}
        if parent_run_id:
            parent = self._pending_steps.get(parent_run_id)
            if parent:
                step.caused_by = parent.step_id
        self._pending_steps[run_id] = step

    def on_retriever_end(
        self,
        documents: list[Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        step = self._pending_steps.pop(run_id, None)
        if step is None:
            return
        step.ended_at = datetime.now(timezone.utc)
        step.outputs = {
            "documents": [
                {"content": getattr(d, "page_content", str(d)), "metadata": getattr(d, "metadata", {})}
                for d in documents
            ]
        }
        self._run.add_step(step)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        step = self._pending_steps.pop(run_id, None)
        if step is None:
            return
        step.ended_at = datetime.now(timezone.utc)
        step.annotations["error"] = {"type": type(error).__name__, "message": str(error)}
        self._run.add_step(step)

    # -- Agent callbacks (optional enrichment) --------------------------------

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        pass

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        pass

    # -- Text callbacks (no-op) -----------------------------------------------

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass


class WrappedChain:
    """Wrapper around a LangChain Runnable that records runs.
    
    Supports fluent interface for configuration:
        >>> wrapped = wrap_chain(chain, ledger).with_name("my-chain")
    """

    def __init__(
        self,
        chain: Any,
        ledger: WorkLedger,
        run_name: str | None = None,
    ) -> None:
        self._chain = chain
        self._ledger = ledger
        self._run_name = run_name or getattr(chain, "name", None) or "langchain"

    def with_name(self, name: str) -> "WrappedChain":
        """Set a custom name for recorded runs.
        
        Args:
            name: Human-readable name for runs
            
        Returns:
            Self for method chaining
            
        Example:
            >>> wrapped = wrap_chain(chain, ledger).with_name("qa-chain")
        """
        self._run_name = name
        return self

    def invoke(self, input: dict, config: dict = None) -> Any:
        """Invoke the chain and record the run."""
        run = self._create_run(input)
        result = None
        error = None
        
        try:
            result = self._chain.invoke(input, config)
            run.status = RunStatus.SUCCESS
            run.outputs = self._serialize_output(result)
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

    async def ainvoke(self, input: dict, config: dict = None) -> Any:
        """Async invoke the chain and record the run."""
        run = self._create_run(input)
        result = None
        error = None
        
        try:
            result = await self._chain.ainvoke(input, config)
            run.status = RunStatus.SUCCESS
            run.outputs = self._serialize_output(result)
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

    def _create_run(self, input: Any) -> Run:
        run = Run(name=self._run_name)
        run.started_at = datetime.now(timezone.utc)
        run.status = RunStatus.RUNNING
        run.inputs = self._serialize_input(input)
        return run

    def _serialize_input(self, input: Any) -> dict:
        if isinstance(input, dict):
            return {k: self._serialize_value(v) for k, v in input.items()}
        return {"input": str(input)}

    def _serialize_output(self, output: Any) -> dict:
        if isinstance(output, dict):
            return {k: self._serialize_value(v) for k, v in output.items()}
        if isinstance(output, str):
            return {"result": output}
        if hasattr(output, "content"):
            return {"content": str(output.content)}
        return {"result": str(output)}

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        return str(value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chain, name)


class WrappedAgent:
    """Wrapper around a LangChain AgentExecutor that records runs.
    
    Supports fluent interface for configuration:
        >>> wrapped = wrap_agent(agent, ledger).with_name("my-agent")
    """

    def __init__(
        self,
        agent: Any,
        ledger: WorkLedger,
        run_name: str | None = None,
    ) -> None:
        self._agent = agent
        self._ledger = ledger
        self._run_name = run_name or getattr(agent, "name", None) or "langchain-agent"

    def with_name(self, name: str) -> "WrappedAgent":
        """Set a custom name for recorded runs.
        
        Args:
            name: Human-readable name for runs
            
        Returns:
            Self for method chaining
            
        Example:
            >>> wrapped = wrap_agent(agent, ledger).with_name("research-agent")
        """
        self._run_name = name
        return self

    def invoke(self, input: dict, config: dict = None) -> dict:
        """Invoke the agent and record the run."""
        run = self._create_run(input)
        result = None
        error = None
        
        try:
            result = self._agent.invoke(input, config)
            run.status = RunStatus.SUCCESS
            run.outputs = {"output": result.get("output", str(result))}
            self._record_intermediate_steps(run, result)
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

    async def ainvoke(self, input: dict, config: dict = None) -> dict:
        """Async invoke the agent and record the run."""
        run = self._create_run(input)
        result = None
        error = None
        
        try:
            result = await self._agent.ainvoke(input, config)
            run.status = RunStatus.SUCCESS
            run.outputs = {"output": result.get("output", str(result))}
            self._record_intermediate_steps(run, result)
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

    def _create_run(self, input: Any) -> Run:
        run = Run(name=self._run_name)
        run.started_at = datetime.now(timezone.utc)
        run.status = RunStatus.RUNNING
        run.inputs = input if isinstance(input, dict) else {"input": str(input)}
        return run

    def _record_intermediate_steps(self, run: Run, result: dict) -> None:
        """Record agent tool calls as steps."""
        intermediate_steps = result.get("intermediate_steps", [])
        
        for action, observation in intermediate_steps:
            tool_name = getattr(action, "tool", "tool")
            tool_input = getattr(action, "tool_input", {})
            
            step = Step(name=tool_name, kind=StepKind.TOOL)
            step.started_at = datetime.now(timezone.utc)
            step.ended_at = datetime.now(timezone.utc)
            step.inputs = tool_input if isinstance(tool_input, dict) else {"input": str(tool_input)}
            step.outputs = {"result": str(observation)}
            run.add_step(step)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)


def wrap_chain(
    chain: Any,
    ledger: WorkLedger,
    run_name: str | None = None,
) -> WrappedChain:
    """Wrap a LangChain Runnable to record runs.
    
    Args:
        chain: LangChain Runnable (chain, LCEL pipeline)
        ledger: WorkLedger instance
        run_name: Custom name for runs
        
    Returns:
        Wrapped chain that records runs
    """
    return WrappedChain(chain, ledger, run_name)


def wrap_agent(
    agent: Any,
    ledger: WorkLedger,
    run_name: str | None = None,
) -> WrappedAgent:
    """Wrap a LangChain AgentExecutor to record runs.
    
    Args:
        agent: LangChain AgentExecutor
        ledger: WorkLedger instance
        run_name: Custom name for runs
        
    Returns:
        Wrapped agent that records runs
    """
    return WrappedAgent(agent, ledger, run_name)
