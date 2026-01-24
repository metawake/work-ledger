# Real API Integration Tests

These scripts test Work Ledger integrations against live APIs.

## Requirements

Set environment variables before running:

```bash
# For most tests (OpenAI-compatible)
export GROQ_API_KEY="your-groq-key"

# For Anthropic tests
export ANTHROPIC_API_KEY="your-anthropic-key"
```

Get free API keys:
- **Groq**: https://console.groq.com/keys (free, no credit card)
- **Anthropic**: https://console.anthropic.com/ ($5 free credit)

## Run Tests

```bash
cd samples/real

# OpenAI SDK via Groq
python openai_groq.py

# PydanticAI
python pydantic_ai.py

# LangChain
python langchain_chain.py
python langchain_agent.py

# LlamaIndex
python llamaindex.py

# Anthropic (needs separate key)
python anthropic.py
```

## What These Test

| Script | Integration | Validates |
|--------|-------------|-----------|
| `openai_groq.py` | OpenAI SDK | Recording, tokens, diff |
| `pydantic_ai.py` | PydanticAI | Agent runs, usage extraction |
| `langchain_chain.py` | LangChain | Chain recording |
| `langchain_agent.py` | LangChain | Agent + tools |
| `llamaindex.py` | LlamaIndex | Query engine, RAG |
| `anthropic.py` | Anthropic | Claude API, multi-turn |

These are **developer validation tests**, not user tutorials. See `samples/01-07*.py` for user-facing examples.
