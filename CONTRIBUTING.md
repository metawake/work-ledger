# Contributing to Work Ledger

Thank you for your interest in contributing to Work Ledger! This document provides guidelines and information for contributors.

## Philosophy

Before contributing, please understand Work Ledger's core philosophy:

- **Work > Logs** — We capture structured work artifacts, not just log lines
- **Small Core > Platform** — We resist feature creep and keep the core minimal
- **Cause > Timeline** — We model causality explicitly
- **Replay > Metrics** — Reproducibility is our primary value

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- A virtual environment tool (venv, conda, etc.)

### Setup

```bash
# Clone the repository
git clone https://github.com/metawake/work-ledger.git
cd work-ledger

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
pytest
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Follow the code standards in `.cursorrules`
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=work_ledger

# Run specific tests
pytest tests/test_specific.py

# Type checking
mypy work_ledger

# Linting
ruff check work_ledger
black --check work_ledger
```

### 4. Commit Your Changes

We use conventional commits:

```bash
git commit -m "feat: add support for custom step kinds"
git commit -m "fix: correct timestamp handling in replay"
git commit -m "docs: update integration guide for LangGraph"
git commit -m "test: add tests for causal link validation"
```

### 5. Submit a Pull Request

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Request review from maintainers

## Types of Contributions

### Bug Reports

When reporting bugs, please include:

1. Work Ledger version
2. Python version
3. Operating system
4. Minimal reproduction steps
5. Expected vs actual behavior
6. Relevant error messages or logs

### Feature Requests

When requesting features:

1. Describe the problem you're trying to solve
2. Explain why existing solutions don't work
3. Propose a potential solution (optional)
4. Consider if it aligns with our philosophy

### Code Contributions

#### Core Changes

Changes to `work_ledger/core/` require:
- Comprehensive tests
- Documentation updates
- Migration utilities if data model changes
- Review from at least two maintainers

#### Integration Adapters

New integrations should:
- Be thin (20-50 lines of glue code)
- Not pull in heavy dependencies
- Include example usage in docstrings
- Have integration tests

#### CLI Changes

CLI modifications should:
- Maintain backwards compatibility
- Support both human and JSON output
- Include help text for all options
- Have command-line tests

### Documentation

We welcome:
- Typo fixes
- Clarifications
- Examples and tutorials
- Translations

## Code Standards

### Style

```python
# Good: Clear, typed, documented
def record_step(
    self,
    name: str,
    kind: StepKind,
    inputs: dict[str, Any],
    caused_by: str | None = None,
) -> Step:
    """Record a step in the current run.
    
    Args:
        name: Human-readable step name
        kind: One of 'model', 'tool', 'retrieval', 'custom'
        inputs: Step input data
        caused_by: Optional step_id that caused this step
        
    Returns:
        The recorded Step object
        
    Example:
        >>> with run.step("fetch-docs", kind="retrieval") as step:
        ...     docs = retriever.search(query)
        ...     step.record_output(docs)
    """
```

### Testing

```python
# Good: Focused, clear assertions, meaningful names
def test_step_records_causal_link():
    """Steps should preserve causal relationships."""
    ledger = WorkLedger(store=":memory:")
    
    with ledger.run("test") as run:
        with run.step("first", kind="tool") as step1:
            step1.record_output({"result": 1})
        
        with run.step("second", kind="model", caused_by=step1.step_id) as step2:
            step2.record_output({"result": 2})
    
    assert step2.caused_by == step1.step_id
```

## What We're Looking For

### High Priority

- [ ] Core data model implementation
- [ ] JSONL and SQLite storage backends
- [ ] Replay-lite functionality
- [ ] Basic diff implementation
- [ ] LangGraph integration
- [ ] PydanticAI integration

### Medium Priority

- [ ] CLI improvements
- [ ] Additional storage backends
- [ ] More framework integrations
- [ ] Documentation and examples

### Low Priority (Future)

- [ ] ARS annotation support
- [ ] Advanced diff algorithms
- [ ] Performance optimizations

## What We're NOT Looking For

Please don't submit PRs for:

- ❌ SaaS or dashboard features
- ❌ Global metrics aggregation
- ❌ Real-time monitoring
- ❌ Framework-specific core logic
- ❌ Heavy dependencies

These are intentionally out of scope.

## Review Process

1. **Automated checks** — CI runs tests, linting, type checks
2. **Maintainer review** — At least one maintainer reviews
3. **Feedback iteration** — Address any requested changes
4. **Merge** — Once approved, a maintainer merges

Typical review time: 2-5 business days

## Community

### Getting Help

- Open a GitHub issue for bugs or features
- Start a GitHub discussion for questions
- Check existing issues before creating new ones

### Code of Conduct

Be respectful, constructive, and patient. We're building tools for serious engineers who care about their craft. Keep discussions technical and productive.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make Work Ledger better!
