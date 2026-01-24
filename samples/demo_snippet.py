# Before: Your normal OpenAI code
from openai import OpenAI
client = OpenAI()

# After: Add 2 lines, every call is recorded
from work_ledger import WorkLedger, wrap_openai
ledger = WorkLedger(store="./runs")
client = wrap_openai(client, ledger)

# Use normally - calls are recorded automatically
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
# Now you can: replay without API costs, diff to find bugs
