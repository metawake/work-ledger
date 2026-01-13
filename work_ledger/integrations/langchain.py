"""LangChain integration for Work Ledger.

Thin wrappers that record LangChain chain and agent executions.

Example:
    >>> from langchain_openai import ChatOpenAI
    >>> from langchain_core.prompts import ChatPromptTemplate
    >>> from work_ledger import WorkLedger
    >>> from work_ledger.integrations.langchain import wrap_chain
    >>> 
    >>> # Build your chain
    >>> prompt = ChatPromptTemplate.from_template("Answer: {question}")
    >>> chain = prompt | ChatOpenAI()
    >>> 
    >>> # Wrap it
    >>> ledger = WorkLedger(store="./runs")
    >>> wrapped = wrap_chain(chain, ledger)
    >>> 
    >>> result = wrapped.invoke({"question": "What is AI?"})
"""

from __future__ import annotations

import traceback
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from work_ledger.core.ledger import WorkLedger
from work_ledger.core.models import Run, Step, StepKind, RunStatus

if TYPE_CHECKING:
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
