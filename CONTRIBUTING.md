# Contributing to Chest X-Ray Classification

Thank you for your interest in contributing! This is a production-ready medical AI project, and contributions are welcome.

## Ways to Contribute

1. **Bug Reports** - Found an issue? Open a GitHub Issue with details
2. **Feature Requests** - Have an idea? Open an Issue for discussion
3. **Code Improvements** - Submit a Pull Request
4. **Documentation** - Fix typos, add examples, improve README
5. **Model Experiments** - Try new architectures and share results

## Development Setup

```bash
# Fork and clone
git clone https://github.com/ariusxscourger/cxr-classifier.git
cd cxr-classifier

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install in development mode with dev dependencies
pip install -e ".[dev,notebook]"

# Install pre-commit hooks
pre-commit install
```

## Code Standards

- **Python**: 3.10+
- **Formatting**: Black (line-length=100)
- **Import sorting**: isort (profile=black)
- **Linting**: ruff
- **Type hints**: Required for all new functions
- **Docstrings**: Google style

Run checks locally:
```bash
# Format
black src/ scripts/
isort src/ scripts/

# Lint
ruff check src/ scripts/

# Type check
mypy src/

# Tests
pytest tests/ -v
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** from `main`: `git checkout -b feature/your-feature-name`
3. **Make changes** with clear, atomic commits
4. **Run tests and checks** locally
5. **Submit PR** with:
   - Clear title and description
   - Reference any related issues
   - Screenshots for UI changes
   - Updated documentation if needed

## Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description

Longer description if needed.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Adding New Models

To add a new model architecture:

1. Add to `configs/config.yaml` model options comment
2. Test with `create_model()` in `src/chestxray/models.py`
3. Add to model comparison in `notebooks/exploration.ipynb`
4. Document in README model zoo table

## Reporting Issues

Use GitHub Issues with:
- **Bug**: Steps to reproduce, expected vs actual behavior, environment
- **Feature**: Use case, proposed solution, alternatives considered
- **Question**: Search existing issues first

## License

By contributing, you agree your contributions will be licensed under the MIT License.
