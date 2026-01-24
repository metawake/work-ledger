#!/usr/bin/env python3
"""
Support Agent Demo - Catch an Agent Bug with Work Ledger

Run: python samples/demo_support_agent.py

This demo shows:
1. How to integrate Work Ledger (3 lines)
2. Recording two agent versions
3. Using diff to find the bug

No API keys needed - uses mock responses.
"""

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from work_ledger import WorkLedger, wrap_openai
from work_ledger.testing import RunDiff

DEMO_STORE = "./demo_runs"


def create_mock_client(tool_calls: list[dict], response_content: str):
    """Create a mock OpenAI client with specific behavior.

    Args:
        tool_calls: List of tool calls the agent makes
        response_content: Final text response
    """
    # Build tool call objects
    mock_tool_calls = []
    for tc in tool_calls:
        mock_tool_calls.append(
            SimpleNamespace(
                id=f"call_{tc['name']}",
                type="function",
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=tc.get("arguments", "{}"),
                ),
            )
        )

    # Build response
    response = SimpleNamespace(
        id="chatcmpl-demo",
        model="gpt-4",
        created=1234567890,
        choices=[
            SimpleNamespace(
                index=0,
                finish_reason="stop",
                message=SimpleNamespace(
                    role="assistant",
                    content=response_content,
                    tool_calls=mock_tool_calls if mock_tool_calls else None,
                ),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        ),
    )

    # Build mock client
    client = SimpleNamespace()
    client.chat = SimpleNamespace()
    client.chat.completions = SimpleNamespace()
    client.chat.completions.create = lambda **kwargs: response

    return client


def cleanup():
    """Remove previous demo runs."""
    if Path(DEMO_STORE).exists():
        shutil.rmtree(DEMO_STORE)


def main():
    print()
    print("=" * 60)
    print("  WORK LEDGER DEMO: Catch an Agent Bug")
    print("=" * 60)
    print()

    # Cleanup previous runs
    cleanup()

    # Setup ledger
    ledger = WorkLedger(store=DEMO_STORE)

    # =========================================================
    # BASELINE AGENT (v1) - Correct behavior
    # =========================================================
    print("1. Running baseline agent (v1)...")
    print("   User: 'I want to cancel my subscription'")
    print()

    client_v1 = create_mock_client(
        tool_calls=[{"name": "cancel_subscription", "arguments": '{"user_id": "123"}'}],
        response_content="Your subscription has been cancelled. "
        "You won't be charged again.",
    )

    wrapped_v1 = wrap_openai(client_v1, ledger, run_name="support-v1")
    result_v1 = wrapped_v1.chat.completions.create(
        messages=[{"role": "user", "content": "I want to cancel my subscription"}],
        model="gpt-4",
    )

    print(f"   Agent v1: {result_v1.choices[0].message.content}")
    print("   Tools called: cancel_subscription")
    print("   ✓ Recorded to Work Ledger")
    print()

    # =========================================================
    # BUGGY AGENT (v2) - After "be more helpful" prompt change
    # =========================================================
    print("2. Running updated agent (v2)...")
    print("   (After adding 'be proactively helpful' to system prompt)")
    print()

    client_v2 = create_mock_client(
        tool_calls=[
            {"name": "refund_payment", "arguments": '{"user_id": "123", "amount": 99}'},
            {"name": "cancel_subscription", "arguments": '{"user_id": "123"}'},
        ],
        response_content="I've cancelled your subscription AND refunded your "
        "last payment of $99. Have a great day!",
    )

    wrapped_v2 = wrap_openai(client_v2, ledger, run_name="support-v2")
    result_v2 = wrapped_v2.chat.completions.create(
        messages=[{"role": "user", "content": "I want to cancel my subscription"}],
        model="gpt-4",
    )

    print(f"   Agent v2: {result_v2.choices[0].message.content}")
    print("   Tools called: refund_payment, cancel_subscription")
    print("   ✓ Recorded to Work Ledger")
    print()

    # =========================================================
    # FIND THE BUG WITH DIFF
    # =========================================================
    print("=" * 60)
    print("  FINDING THE BUG")
    print("=" * 60)
    print()
    print("$ work-ledger diff ./demo_runs support-v1 support-v2")
    print()

    # Get the runs
    runs = ledger.list_runs()
    run_v1 = next(r for r in runs if r.name == "support-v1")
    run_v2 = next(r for r in runs if r.name == "support-v2")

    # Compute diff
    diff = RunDiff(run_v1, run_v2)

    # Display diff
    print("-" * 60)
    print(f"Comparing runs:")
    print(f"  Expected: {run_v1.run_id[:12]}... ({run_v1.name})")
    print(f"  Actual:   {run_v2.run_id[:12]}... ({run_v2.name})")
    print()
    print(f"Similarity: {diff.similarity:.1%}")
    print()

    # Find added steps by comparing step names
    v1_step_names = {s.name for s in run_v1.steps}
    v2_step_names = {s.name for s in run_v2.steps}
    added_steps = v2_step_names - v1_step_names

    if added_steps or diff.steps_added > 0:
        print("Step changes:")
        for step in run_v2.steps:
            if step.name in added_steps:
                if step.name == "refund_payment":
                    print(f"  + {step.name} [tool]    ← UNAUTHORIZED!")
                else:
                    print(f"  + {step.name} [{step.kind.value}]")
        print()

    if diff.output_changed:
        print("Output changes:")
        # output_diff has 'added', 'removed', 'changed' keys
        changed = diff.output_diff.get("changed", {})
        for key, vals in changed.items():
            old = vals.get("expected", "")
            new = vals.get("actual", "")
            old_short = str(old)[:30] + "..." if len(str(old)) > 30 else old
            new_short = str(new)[:30] + "..." if len(str(new)) > 30 else new
            print(f"  ~ {key}: \"{old_short}\" → \"{new_short}\"")
        print()

    print("-" * 60)
    print()
    print("🔍 Found it! The 'be helpful' prompt caused unauthorized refunds.")
    print()
    print("=" * 60)
    print("  Demo complete. Runs saved in ./demo_runs")
    print("  Try: work-ledger list ./demo_runs")
    print("       work-ledger show ./demo_runs support-v1")
    print("       work-ledger replay ./demo_runs support-v2")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
