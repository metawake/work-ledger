"""Tests for WorkLedgerCallbackHandler (LangChain BaseCallbackHandler)."""

from uuid import uuid4

from work_ledger.core.ledger import WorkLedger
from work_ledger.core.models import RunStatus, StepKind
from work_ledger.integrations.langchain import WorkLedgerCallbackHandler


def _make_handler(**kwargs):
    ledger = WorkLedger(store=":memory:")
    return WorkLedgerCallbackHandler(ledger, **kwargs), ledger


class _FakeLLMResult:
    """Mimics langchain_core.outputs.LLMResult without the dependency."""
    def __init__(self, generations=None, llm_output=None):
        self.generations = generations or []
        self.llm_output = llm_output


class _FakeGeneration:
    def __init__(self, text):
        self.text = text


class TestLLMCallbacks:

    def test_llm_start_end_records_step(self):
        h, _ = _make_handler(auto_save=False)
        run_id = uuid4()
        h.on_llm_start(
            serialized={"id": ["langchain", "ChatOpenAI"]},
            prompts=["hello"],
            run_id=run_id,
        )
        result = _FakeLLMResult(
            generations=[[_FakeGeneration("world")]],
            llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        )
        h.on_llm_end(result, run_id=run_id)
        run = h.get_run()

        assert len(run.steps) == 1
        step = run.steps[0]
        assert step.kind == StepKind.MODEL
        assert step.name == "ChatOpenAI"
        assert step.inputs["prompts"] == ["hello"]
        assert step.outputs["generations"][0][0]["text"] == "world"
        assert step.metrics.total_tokens == 15
        assert step.started_at is not None
        assert step.ended_at is not None

    def test_llm_error_marks_failed(self):
        h, _ = _make_handler(auto_save=False)
        run_id = uuid4()
        h.on_llm_start({"id": ["llm"]}, ["p"], run_id=run_id)
        h.on_llm_error(RuntimeError("boom"), run_id=run_id)
        run = h.get_run()

        assert run.status == RunStatus.FAILED
        assert len(run.steps) == 1
        assert run.steps[0].annotations["error"]["message"] == "boom"

    def test_chat_model_start(self):
        h, _ = _make_handler(auto_save=False)
        run_id = uuid4()

        class FakeMsg:
            type = "human"
            content = "hi"

        h.on_chat_model_start(
            serialized={"id": ["langchain", "GPT4"]},
            messages=[[FakeMsg()]],
            run_id=run_id,
        )
        h.on_llm_end(_FakeLLMResult(), run_id=run_id)
        run = h.get_run()

        assert run.steps[0].name == "GPT4"
        assert run.steps[0].inputs["messages"][0][0]["role"] == "human"


class TestToolCallbacks:

    def test_tool_start_end(self):
        h, _ = _make_handler(auto_save=False)
        run_id = uuid4()
        h.on_tool_start({"name": "calculator"}, "2+2", run_id=run_id)
        h.on_tool_end("4", run_id=run_id)
        run = h.get_run()

        assert len(run.steps) == 1
        step = run.steps[0]
        assert step.kind == StepKind.TOOL
        assert step.name == "calculator"
        assert step.inputs["input"] == "2+2"
        assert step.outputs["output"] == "4"

    def test_tool_error(self):
        h, _ = _make_handler(auto_save=False)
        run_id = uuid4()
        h.on_tool_start({"name": "t"}, "x", run_id=run_id)
        h.on_tool_error(ValueError("bad"), run_id=run_id)
        run = h.get_run()

        assert run.steps[0].annotations["error"]["type"] == "ValueError"


class TestChainCallbacks:

    def test_chain_captures_io(self):
        h, _ = _make_handler(auto_save=False)
        rid = uuid4()
        h.on_chain_start({}, {"query": "q"}, run_id=rid)
        h.on_chain_end({"answer": "a"}, run_id=rid)
        run = h.get_run()

        assert run.inputs == {"query": "q"}
        assert run.outputs == {"answer": "a"}
        assert run.status == RunStatus.SUCCESS

    def test_chain_error(self):
        h, _ = _make_handler(auto_save=False)
        rid = uuid4()
        h.on_chain_start({}, {"q": "x"}, run_id=rid)
        h.on_chain_error(RuntimeError("fail"), run_id=rid)
        run = h.get_run()

        assert run.status == RunStatus.FAILED

    def test_auto_save_on_chain_end(self):
        h, ledger = _make_handler(auto_save=True)
        rid = uuid4()
        h.on_chain_start({}, {}, run_id=rid)
        h.on_chain_end({}, run_id=rid)

        runs = ledger.list_runs()
        assert len(runs) == 1

    def test_nested_chains_only_save_once(self):
        h, ledger = _make_handler(auto_save=True)
        outer, inner = uuid4(), uuid4()
        h.on_chain_start({}, {"q": "x"}, run_id=outer)
        h.on_chain_start({}, {}, run_id=inner)
        h.on_chain_end({}, run_id=inner)
        assert ledger.list_runs() == []
        h.on_chain_end({"a": "y"}, run_id=outer)
        assert len(ledger.list_runs()) == 1


class TestRetrieverCallbacks:

    def test_retriever_start_end(self):
        h, _ = _make_handler(auto_save=False)
        rid = uuid4()

        class FakeDoc:
            page_content = "some text"
            metadata = {"source": "wiki"}

        h.on_retriever_start({"name": "vectordb"}, "search query", run_id=rid)
        h.on_retriever_end([FakeDoc()], run_id=rid)
        run = h.get_run()

        assert len(run.steps) == 1
        step = run.steps[0]
        assert step.kind == StepKind.RETRIEVAL
        assert step.inputs["query"] == "search query"
        assert step.outputs["documents"][0]["content"] == "some text"


class TestCausalLinks:

    def test_parent_run_id_links_steps(self):
        h, _ = _make_handler(auto_save=False)
        parent_id = uuid4()
        child_id = uuid4()

        h.on_llm_start({"id": ["llm"]}, ["p1"], run_id=parent_id)
        h.on_tool_start({"name": "t"}, "x", run_id=child_id, parent_run_id=parent_id)
        h.on_tool_end("y", run_id=child_id)
        h.on_llm_end(_FakeLLMResult(), run_id=parent_id)

        run = h.get_run()
        tool_step = [s for s in run.steps if s.kind == StepKind.TOOL][0]
        llm_step = [s for s in run.steps if s.kind == StepKind.MODEL][0]
        assert tool_step.caused_by == llm_step.step_id


class TestRunNameAndMetrics:

    def test_custom_run_name(self):
        h, _ = _make_handler(run_name="my-pipeline", auto_save=False)
        assert h.get_run().name == "my-pipeline"

    def test_aggregated_metrics(self):
        h, _ = _make_handler(auto_save=False)
        for i in range(3):
            rid = uuid4()
            h.on_llm_start({"id": ["llm"]}, ["p"], run_id=rid)
            h.on_llm_end(
                _FakeLLMResult(llm_output={"token_usage": {"total_tokens": 10}}),
                run_id=rid,
            )
        run = h.get_run()
        assert run.metrics.total_tokens == 30
