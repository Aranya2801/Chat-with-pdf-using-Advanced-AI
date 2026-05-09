# Contributing to Chat-with-PDF Advanced AI

Thank you for your interest in contributing! 🎉

## Development Setup

```bash
git clone https://github.com/Aranya2801/Chat-with-pdf-using-Advanced-AI.git
cd Chat-with-pdf-using-Advanced-AI
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API key
```

## Code Standards

- **Formatter:** `black` — run `black src/ app.py`
- **Linter:** `ruff` — run `ruff check src/ app.py`
- **Imports:** `isort` — run `isort src/ app.py`
- **Type hints:** required for all public functions
- **Docstrings:** Google-style for all classes and public methods

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Write tests for new functionality in `tests/`
4. Ensure all tests pass: `pytest tests/ -v`
5. Run linters: `ruff check . && black --check .`
6. Commit with a clear message: `git commit -m "feat: add amazing feature"`
7. Push and open a PR against `main`

## Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add multi-language PDF support
fix: resolve chunking issue with scanned PDFs
docs: update README with Docker instructions
test: add unit tests for ReasoningAgent
refactor: simplify VectorStoreManager API
perf: cache embedding results to reduce API calls
```

## Reporting Bugs

Please use [GitHub Issues](https://github.com/Aranya2801/Chat-with-pdf-using-Advanced-AI/issues) and include:
- Python version
- Error traceback
- Steps to reproduce
- Expected vs actual behavior

## Questions?

Open a [Discussion](https://github.com/Aranya2801/Chat-with-pdf-using-Advanced-AI/discussions).
