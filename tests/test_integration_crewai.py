"""Tests for CrewAI integration."""

import pytest
from typing import Any

from work_ledger import WorkLedger
from work_ledger.core.models import StepKind, RunStatus


# --- Mock CrewAI classes for testing ---

class MockAgent:
    """Mock CrewAI Agent."""
    def __init__(self, role: str, goal: str, backstory: str = ""):
        self.role = role
        self.goal = goal
        self.backstory = backstory


class MockTask:
    """Mock CrewAI Task."""
    def __init__(self, description: str, agent: MockAgent, expected_output: str = ""):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.output = None


class MockTaskOutput:
    """Mock task output."""
    def __init__(self, raw: str, task: MockTask):
        self.raw = raw
        self.description = task.description
        self.agent = task.agent.role


class MockCrewOutput:
    """Mock crew execution output."""
    def __init__(self, raw: str, tasks_output: list = None, token_usage: dict = None):
        self.raw = raw
        self.tasks_output = tasks_output or []
        self.token_usage = token_usage or {}
    
    def __str__(self):
        return self.raw


class MockCrew:
    """Mock CrewAI Crew for testing."""
    
    def __init__(self, agents: list, tasks: list, name: str = "test-crew"):
        self.agents = agents
        self.tasks = tasks
        self.name = name
        self._result = None
    
    def set_result(self, result: MockCrewOutput):
        """Set the result that kickoff() will return."""
        self._result = result
    
    def kickoff(self, inputs: dict = None) -> MockCrewOutput:
        """Execute the crew."""
        if self._result:
            return self._result
        
        # Default: simulate task execution
        tasks_output = []
        for task in self.tasks:
            output = MockTaskOutput(
                raw=f"Completed: {task.description}",
                task=task,
            )
            tasks_output.append(output)
        
        return MockCrewOutput(
            raw="Crew execution complete",
            tasks_output=tasks_output,
            token_usage={"total_tokens": 500, "prompt_tokens": 300, "completion_tokens": 200},
        )
    
    async def kickoff_async(self, inputs: dict = None) -> MockCrewOutput:
        """Async execution."""
        return self.kickoff(inputs)


class TestCrewAIIntegration:
    """Tests for CrewAI wrapper."""

    def test_wrap_crew_basic(self):
        """Wrapped crew records runs."""
        from work_ledger.integrations.crewai import wrap_crew
        
        ledger = WorkLedger(store=":memory:")
        
        agent = MockAgent(role="Researcher", goal="Research topics")
        task = MockTask(description="Research AI", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task], name="research-crew")
        
        wrapped = wrap_crew(crew, ledger)
        result = wrapped.kickoff()
        
        # Should return the crew's result
        assert "complete" in str(result).lower()
        
        # Should have recorded a run
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].name == "research-crew"
        assert runs[0].status == RunStatus.SUCCESS

    def test_records_input_output(self):
        """Wrapper records inputs and final output."""
        from work_ledger.integrations.crewai import wrap_crew
        
        ledger = WorkLedger(store=":memory:")
        
        agent = MockAgent(role="Writer", goal="Write content")
        task = MockTask(description="Write article", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task])
        crew.set_result(MockCrewOutput(raw="Final article content"))
        
        wrapped = wrap_crew(crew, ledger)
        wrapped.kickoff(inputs={"topic": "AI trends"})
        
        run = ledger.list_runs()[0]
        assert run.inputs["topic"] == "AI trends"
        assert run.outputs["result"] == "Final article content"

    def test_records_task_steps(self):
        """Wrapper records each task as a step."""
        from work_ledger.integrations.crewai import wrap_crew
        
        ledger = WorkLedger(store=":memory:")
        
        researcher = MockAgent(role="Researcher", goal="Research")
        writer = MockAgent(role="Writer", goal="Write")
        
        task1 = MockTask(description="Research topic", agent=researcher)
        task2 = MockTask(description="Write article", agent=writer)
        
        crew = MockCrew(agents=[researcher, writer], tasks=[task1, task2])
        
        wrapped = wrap_crew(crew, ledger)
        wrapped.kickoff()
        
        run = ledger.list_runs()[0]
        
        # Should have a step for each task
        assert len(run.steps) == 2
        assert run.steps[0].name == "Research topic"
        assert run.steps[1].name == "Write article"

    def test_records_agent_info(self):
        """Wrapper records which agent executed each task."""
        from work_ledger.integrations.crewai import wrap_crew
        
        ledger = WorkLedger(store=":memory:")
        
        agent = MockAgent(role="Analyst", goal="Analyze data")
        task = MockTask(description="Analyze trends", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task])
        
        wrapped = wrap_crew(crew, ledger)
        wrapped.kickoff()
        
        run = ledger.list_runs()[0]
        step = run.steps[0]
        
        assert step.outputs.get("agent_role") == "Analyst"

    def test_records_token_usage(self):
        """Wrapper records token usage metrics."""
        from work_ledger.integrations.crewai import wrap_crew
        
        ledger = WorkLedger(store=":memory:")
        
        agent = MockAgent(role="Agent", goal="Goal")
        task = MockTask(description="Task", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task])
        
        # Create a result with token usage AND tasks_output
        task_output = MockTaskOutput(raw="Task done", task=task)
        crew.set_result(MockCrewOutput(
            raw="Done",
            tasks_output=[task_output],
            token_usage={"total_tokens": 1000, "prompt_tokens": 600, "completion_tokens": 400},
        ))
        
        wrapped = wrap_crew(crew, ledger)
        wrapped.kickoff()
        
        run = ledger.list_runs()[0]
        assert run.metrics.total_tokens == 1000
        assert run.metrics.prompt_tokens == 600
        assert run.metrics.completion_tokens == 400

    def test_handles_exceptions(self):
        """Wrapper handles crew exceptions gracefully."""
        from work_ledger.integrations.crewai import wrap_crew
        
        ledger = WorkLedger(store=":memory:")
        
        class FailingCrew(MockCrew):
            def kickoff(self, inputs=None):
                raise RuntimeError("Agent failed")
        
        agent = MockAgent(role="Agent", goal="Goal")
        task = MockTask(description="Task", agent=agent)
        crew = FailingCrew(agents=[agent], tasks=[task])
        
        wrapped = wrap_crew(crew, ledger)
        
        with pytest.raises(RuntimeError):
            wrapped.kickoff()
        
        # Should still record the failed run
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.FAILED
        assert "error" in runs[0].annotations

    def test_custom_run_name(self):
        """Can specify custom run name."""
        from work_ledger.integrations.crewai import wrap_crew
        
        ledger = WorkLedger(store=":memory:")
        
        agent = MockAgent(role="Agent", goal="Goal")
        task = MockTask(description="Task", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task], name="default")
        
        wrapped = wrap_crew(crew, ledger, run_name="custom-workflow")
        wrapped.kickoff()
        
        run = ledger.list_runs()[0]
        assert run.name == "custom-workflow"

    def test_multiple_kickoffs(self):
        """Multiple kickoffs create separate runs."""
        from work_ledger.integrations.crewai import wrap_crew
        
        ledger = WorkLedger(store=":memory:")
        
        agent = MockAgent(role="Agent", goal="Goal")
        task = MockTask(description="Task", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task])
        
        wrapped = wrap_crew(crew, ledger)
        
        wrapped.kickoff(inputs={"run": 1})
        wrapped.kickoff(inputs={"run": 2})
        wrapped.kickoff(inputs={"run": 3})
        
        runs = ledger.list_runs()
        assert len(runs) == 3


class TestCrewAIAsyncIntegration:
    """Tests for async CrewAI wrapper."""

    @pytest.mark.asyncio
    async def test_async_kickoff(self):
        """Async kickoff is recorded."""
        from work_ledger.integrations.crewai import wrap_crew
        
        ledger = WorkLedger(store=":memory:")
        
        agent = MockAgent(role="Agent", goal="Goal")
        task = MockTask(description="Async task", agent=agent)
        crew = MockCrew(agents=[agent], tasks=[task])
        
        wrapped = wrap_crew(crew, ledger)
        result = await wrapped.kickoff_async(inputs={"async": True})
        
        assert "complete" in str(result).lower()
        
        runs = ledger.list_runs()
        assert len(runs) == 1
        assert runs[0].status == RunStatus.SUCCESS
