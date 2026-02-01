"""Work Ledger CLI.

Commands:
    work-ledger list <store>              List all runs
    work-ledger show <store> <run_id>     Show run details
    work-ledger diff <store> <id1> <id2>  Compare two runs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import Any

from work_ledger.core.store import RunStore
from work_ledger.core.models import Run
from work_ledger.testing.diff import RunDiff


def format_datetime(dt: datetime | None) -> str:
    """Format datetime for display."""
    if dt is None:
        return "-"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def format_duration(run: Run) -> str:
    """Format run duration."""
    if run.started_at and run.ended_at:
        delta = run.ended_at - run.started_at
        return f"{delta.total_seconds():.2f}s"
    return "-"


def cmd_list(args: argparse.Namespace) -> int:
    """List all runs in a store."""
    store = RunStore.create(args.store)
    runs = store.list_runs()
    
    if not runs:
        print("No runs found.")
        return 0
    
    if args.json:
        print(json.dumps([r.to_dict() for r in runs], indent=2, default=str))
        return 0
    
    # Sort by started_at descending
    runs.sort(key=lambda r: r.started_at or datetime.min, reverse=True)
    
    # Print table header
    print(f"{'ID':<12} {'Name':<25} {'Status':<10} {'Duration':<10} {'Started':<20}")
    print("-" * 80)
    
    for run in runs:
        run_id = run.run_id[:10] + ".."
        name = run.name[:23] + ".." if len(run.name) > 25 else run.name
        status = run.status.value
        duration = format_duration(run)
        started = format_datetime(run.started_at)
        
        print(f"{run_id:<12} {name:<25} {status:<10} {duration:<10} {started:<20}")
    
    print(f"\nTotal: {len(runs)} run(s)")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show details of a specific run."""
    store = RunStore.create(args.store)
    
    # Support partial ID matching
    run = store.get_run(args.run_id)
    
    if run is None:
        # Try partial match
        runs = store.list_runs()
        matches = [r for r in runs if r.run_id.startswith(args.run_id)]
        if len(matches) == 1:
            run = matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous ID '{args.run_id}'. Matches:")
            for r in matches:
                print(f"  {r.run_id}")
            return 1
        else:
            print(f"Run '{args.run_id}' not found.")
            return 1
    
    if args.json:
        print(json.dumps(run.to_dict(), indent=2, default=str))
        return 0
    
    # Print run details
    print(f"Run: {run.run_id}")
    print(f"Name: {run.name}")
    print(f"Status: {run.status.value}")
    print(f"Started: {format_datetime(run.started_at)}")
    print(f"Ended: {format_datetime(run.ended_at)}")
    print(f"Duration: {format_duration(run)}")
    print()
    
    # Metrics
    print("Metrics:")
    print(f"  Tokens: {run.metrics.total_tokens} (prompt: {run.metrics.prompt_tokens}, completion: {run.metrics.completion_tokens})")
    if run.metrics.latency_ms:
        print(f"  Latency: {run.metrics.latency_ms}ms")
    if run.metrics.cost_usd:
        print(f"  Cost: ${run.metrics.cost_usd:.4f}")
    print()
    
    # Inputs/Outputs
    if run.inputs:
        print("Inputs:")
        for k, v in run.inputs.items():
            v_str = str(v)[:60] + "..." if len(str(v)) > 60 else str(v)
            print(f"  {k}: {v_str}")
        print()
    
    if run.outputs:
        print("Outputs:")
        for k, v in run.outputs.items():
            v_str = str(v)[:60] + "..." if len(str(v)) > 60 else str(v)
            print(f"  {k}: {v_str}")
        print()
    
    # Steps
    if run.steps:
        print(f"Steps ({len(run.steps)}):")
        for i, step in enumerate(run.steps, 1):
            print(f"  {i}. [{step.kind.value}] {step.name}")
            if step.metrics.total_tokens:
                print(f"     Tokens: {step.metrics.total_tokens}")
    
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    """Compare two runs."""
    store = RunStore.create(args.store)
    
    run1 = store.get_run(args.id1)
    run2 = store.get_run(args.id2)
    
    if run1 is None:
        print(f"Run '{args.id1}' not found.")
        return 1
    
    if run2 is None:
        print(f"Run '{args.id2}' not found.")
        return 1
    
    diff = RunDiff(run1, run2)
    
    if args.json:
        result = {
            "similarity": diff.similarity,
            "has_changes": diff.has_changes,
            "status_changed": diff.status_changed,
            "input_diff": diff.input_diff,
            "output_diff": diff.output_diff,
            "metrics_diff": diff.metrics_diff,
            "steps_added": len(diff.steps_added),
            "steps_removed": len(diff.steps_removed),
        }
        print(json.dumps(result, indent=2, default=str))
        return 0
    
    print(f"Comparing runs:")
    print(f"  Expected: {run1.run_id[:20]}... ({run1.name})")
    print(f"  Actual:   {run2.run_id[:20]}... ({run2.name})")
    print()
    print(f"Similarity: {diff.similarity:.1%}")
    print()
    
    if diff.status_changed:
        print(f"Status: {run1.status.value} → {run2.status.value}")
        print()
    
    if diff.input_diff.get("changed") or diff.input_diff.get("added") or diff.input_diff.get("removed"):
        print("Input changes:")
        for k, v in diff.input_diff.get("changed", {}).items():
            print(f"  ~ {k}: {v['expected']} → {v['actual']}")
        for k in diff.input_diff.get("added", {}):
            print(f"  + {k}")
        for k in diff.input_diff.get("removed", {}):
            print(f"  - {k}")
        print()
    
    if diff.output_diff.get("changed") or diff.output_diff.get("added") or diff.output_diff.get("removed"):
        print("Output changes:")
        for k, v in diff.output_diff.get("changed", {}).items():
            exp = str(v['expected'])[:30]
            act = str(v['actual'])[:30]
            print(f"  ~ {k}: {exp}... → {act}...")
        for k in diff.output_diff.get("added", {}):
            print(f"  + {k}")
        for k in diff.output_diff.get("removed", {}):
            print(f"  - {k}")
        print()
    
    if diff.steps_added or diff.steps_removed:
        print("Step changes:")
        for step in diff.steps_added:
            print(f"  + {step.name} [{step.kind.value}]")
        for step in diff.steps_removed:
            print(f"  - {step.name} [{step.kind.value}]")
        print()
    
    if diff.metrics_diff:
        print("Metric changes:")
        for k, v in diff.metrics_diff.items():
            print(f"  {k}: {v['expected']} → {v['actual']}")
    
    if not diff.has_changes:
        print("No significant changes detected.")
    
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Show replay info for a run."""
    store = RunStore.create(args.store)
    run = store.get_run(args.run_id)
    
    if run is None:
        # Try partial match
        runs = store.list_runs()
        matches = [r for r in runs if r.run_id.startswith(args.run_id)]
        if len(matches) == 1:
            run = matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous ID '{args.run_id}'. Matches:")
            for r in matches:
                print(f"  {r.run_id}")
            return 1
        else:
            print(f"Run '{args.run_id}' not found.")
            return 1
    
    # Find steps with fixtures
    steps_with_fixtures = [
        s for s in run.steps
        if s.annotations.get("fixture")
    ]
    
    if args.json:
        result = {
            "run_id": run.run_id,
            "name": run.name,
            "fixtures": len(steps_with_fixtures),
            "replayable": len(steps_with_fixtures) > 0,
            "steps": [
                {
                    "name": s.name,
                    "kind": s.kind.value,
                    "has_fixture": bool(s.annotations.get("fixture")),
                    "fixture_type": s.annotations.get("fixture", {}).get("type"),
                }
                for s in run.steps
            ]
        }
        print(json.dumps(result, indent=2))
        return 0
    
    print(f"Run: {run.run_id}")
    print(f"Name: {run.name}")
    print(f"Status: {run.status.value}")
    print()
    
    if not steps_with_fixtures:
        print("No fixtures found. This run cannot be replayed.")
        print()
        print("To enable replay, use a wrapped client:")
        print("  wrapped = wrap_openai(client, ledger)")
        return 0
    
    print(f"Fixtures: {len(steps_with_fixtures)} API call(s) captured")
    print()
    print("Steps:")
    for i, step in enumerate(run.steps, 1):
        fixture = step.annotations.get("fixture")
        if fixture:
            fixture_type = fixture.get("type", "unknown")
            print(f"  {i}. [{step.kind.value}] {step.name}")
            print(f"     Fixture: ✓ {fixture_type}")
            if step.metrics.total_tokens:
                print(f"     Tokens: {step.metrics.total_tokens}")
        else:
            print(f"  {i}. [{step.kind.value}] {step.name}")
    
    print()
    print("To replay in Python:")
    print()
    
    # Detect integration type
    fixture_types = [
        s.annotations.get("fixture", {}).get("type", "")
        for s in steps_with_fixtures
    ]
    
    if any("openai" in t for t in fixture_types):
        print("  from work_ledger import WorkLedger, wrap_openai")
        print(f'  ledger = WorkLedger(store="{args.store}")')
        print(f'  wrapped = wrap_openai(client, ledger, replay_from="{run.run_id}")')
        print("  response = wrapped.chat.completions.create(...)  # No API call")
    elif any("anthropic" in t for t in fixture_types):
        print("  from work_ledger import WorkLedger, wrap_anthropic")
        print(f'  ledger = WorkLedger(store="{args.store}")')
        print(f'  wrapped = wrap_anthropic(client, ledger, replay_from="{run.run_id}")')
        print("  response = wrapped.messages.create(...)  # No API call")
    elif any("pydantic" in t for t in fixture_types):
        print("  from work_ledger import WorkLedger, wrap_agent")
        print(f'  ledger = WorkLedger(store="{args.store}")')
        print(f'  wrapped = wrap_agent(agent, ledger, replay_from="{run.run_id}")')
        print("  result = wrapped.run_sync(...)  # No API call")
    else:
        print("  # Use replay_from parameter with the appropriate wrapper")
        print(f'  wrapped = wrap_*(client, ledger, replay_from="{run.run_id}")')
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="work-ledger",
        description="Agent diagnostics for LLM workflows. Record. Replay. Diff.",
    )
    parser.add_argument(
        "--version", action="version", version="work-ledger 0.1.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # list command
    list_parser = subparsers.add_parser("list", help="List all runs")
    list_parser.add_argument("store", help="Path to store (e.g., ./runs)")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # show command
    show_parser = subparsers.add_parser("show", help="Show run details")
    show_parser.add_argument("store", help="Path to store")
    show_parser.add_argument("run_id", help="Run ID (or prefix)")
    show_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # diff command
    diff_parser = subparsers.add_parser("diff", help="Compare two runs")
    diff_parser.add_argument("store", help="Path to store")
    diff_parser.add_argument("id1", help="First run ID")
    diff_parser.add_argument("id2", help="Second run ID")
    diff_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # replay command
    replay_parser = subparsers.add_parser("replay", help="Show replay info for a run")
    replay_parser.add_argument("store", help="Path to store")
    replay_parser.add_argument("run_id", help="Run ID to replay")
    replay_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    try:
        if args.command == "list":
            return cmd_list(args)
        elif args.command == "show":
            return cmd_show(args)
        elif args.command == "diff":
            return cmd_diff(args)
        elif args.command == "replay":
            return cmd_replay(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
