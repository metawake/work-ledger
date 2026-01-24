#!/usr/bin/env python3
"""
Sample 07: CrewAI Integration
==============================

Wrap your CrewAI crew to automatically record runs.

This sample shows:
- Wrapping a crew with Work Ledger
- Automatic recording of inputs/outputs
- Task execution tracking
- Agent role tracking per task

Run:
    python 07_crewai_integration.py

Note: This sample uses mock objects. With real CrewAI:
    pip install crewai
"""

from work_ledger import WorkLedger
from work_ledger.integrations.crewai import wrap_crew


# --- Mock CrewAI (for demo without real dependency) ---

class MockAgent:
    def __init__(self, role: str, goal: str):
        self.role = role
        self.goal = goal


class MockTask:
    def __init__(self, description: str, agent: MockAgent):
        self.description = description
        self.agent = agent


class MockTaskOutput:
    def __init__(self, description: str, agent_role: str, result: str):
        self.description = description
        self.agent = agent_role
        self.raw = result


class MockCrewOutput:
    def __init__(self, raw: str, tasks_output: list, token_usage: dict):
        self.raw = raw
        self.tasks_output = tasks_output
        self.token_usage = token_usage
    
    def __str__(self):
        return self.raw


class MockCrew:
    """Simulates a CrewAI Crew."""
    
    def __init__(self, agents: list, tasks: list, name: str = "research-crew"):
        self.agents = agents
        self.tasks = tasks
        self.name = name
    
    def kickoff(self, inputs: dict = None) -> MockCrewOutput:
        """Simulate crew execution."""
        topic = inputs.get("topic", "AI") if inputs else "AI"
        
        return MockCrewOutput(
            raw=f"Comprehensive report on {topic} completed.",
            tasks_output=[
                MockTaskOutput(
                    description="Research the topic",
                    agent_role="Researcher",
                    result=f"Found 10 key papers on {topic}",
                ),
                MockTaskOutput(
                    description="Analyze findings",
                    agent_role="Analyst",
                    result="Identified 3 major trends",
                ),
                MockTaskOutput(
                    description="Write report",
                    agent_role="Writer",
                    result=f"Final report: {topic} is transforming industry",
                ),
            ],
            token_usage={
                "total_tokens": 2500,
                "prompt_tokens": 1500,
                "completion_tokens": 1000,
            },
        )


# --- Demo ---

print("=" * 60)
print("CrewAI Integration Demo")
print("=" * 60)
print()

# Create ledger
ledger = WorkLedger(store=":memory:")

# Create crew (in real usage: Crew(agents=[...], tasks=[...]))
researcher = MockAgent(role="Researcher", goal="Find relevant information")
analyst = MockAgent(role="Analyst", goal="Analyze data")
writer = MockAgent(role="Writer", goal="Create reports")

task1 = MockTask(description="Research the topic", agent=researcher)
task2 = MockTask(description="Analyze findings", agent=analyst)
task3 = MockTask(description="Write report", agent=writer)

crew = MockCrew(
    agents=[researcher, analyst, writer],
    tasks=[task1, task2, task3],
    name="research-crew",
)

# Wrap the crew - this is the key line!
wrapped = wrap_crew(crew, ledger)

print("1. Wrap your crew:")
print("   wrapped = wrap_crew(crew, ledger)")
print()

# Run the crew - it's recorded automatically
print("2. Use it normally:")
print("   result = wrapped.kickoff(inputs={'topic': 'Machine Learning'})")
print()

result = wrapped.kickoff(inputs={"topic": "Machine Learning"})

print(f"   Result: {result}")
print()

# Check what was recorded
print("=" * 60)
print("What Work Ledger Recorded")
print("=" * 60)
print()

run = ledger.list_runs()[0]

print(f"Run: {run.name}")
print(f"Status: {run.status.value}")
print(f"Input: {run.inputs}")
print(f"Output: {run.outputs['result'][:50]}...")
print()

print(f"Agents in crew: {len(run.annotations.get('agents', []))}")
for agent in run.annotations.get("agents", []):
    print(f"  - {agent['role']}")
print()

print("Task Steps (each task = one step):")
for i, step in enumerate(run.steps, 1):
    print(f"  {i}. [{step.kind.value}] {step.name}")
    print(f"     Agent: {step.outputs.get('agent_role', 'N/A')}")
    print(f"     Result: {step.outputs.get('result', 'N/A')[:40]}...")
print()

print("Metrics:")
print(f"  Prompt tokens: {run.metrics.prompt_tokens}")
print(f"  Completion tokens: {run.metrics.completion_tokens}")
print(f"  Total tokens: {run.metrics.total_tokens}")
print()

# Multiple runs
print("=" * 60)
print("Track Multiple Crew Runs")
print("=" * 60)
print()

wrapped.kickoff(inputs={"topic": "Deep Learning"})
wrapped.kickoff(inputs={"topic": "LLMs"})

print(f"Total runs recorded: {len(ledger.list_runs())}")
for r in ledger.list_runs():
    topic = r.inputs.get("topic", "N/A")
    print(f"  - {r.run_id[:8]}... | {r.status.value} | topic: {topic}")

print()
print("=" * 60)
print("Real CrewAI Usage")
print("=" * 60)
print("""
from crewai import Crew, Agent, Task
from work_ledger import WorkLedger
from work_ledger.integrations.crewai import wrap_crew

# Your real crew
researcher = Agent(role="Researcher", goal="Research topics", ...)
writer = Agent(role="Writer", goal="Write articles", ...)
task1 = Task(description="Research", agent=researcher)
task2 = Task(description="Write", agent=writer)
crew = Crew(agents=[researcher, writer], tasks=[task1, task2])

# Wrap it
ledger = WorkLedger(store="./runs")
wrapped = wrap_crew(crew, ledger)

# Use normally - everything is recorded
result = wrapped.kickoff(inputs={"topic": "AI Safety"})

# Later: debug, diff, replay
runs = ledger.list_runs()
""")
