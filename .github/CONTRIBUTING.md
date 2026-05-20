# Contributing

Thanks for taking the time to contribute. `ytcc-pipeline` is a small project; this guide is intentionally short.

## Setup

```bash
git clone https://github.com/ozefe/ytcc-pipeline
cd ytcc-pipeline
python3.14 -m venv .venv
source .venv/bin/activate
pip install -e . --group dev
```

## Before you open a pull request

```bash
ruff check
ruff format --check
pyright
pytest
```

All four must pass. CI runs the same commands.

## Working on a change

- Write the test first, then the code. New behavior without a test will be sent back.
- One logical change per pull request. Smaller is better.
- Match the existing code style; `ruff format` handles the mechanics.
- Update relevant documentation in the same pull request.

## Commit and pull request style

- Imperative commit subjects ("Add formula bucket sweep", not "Added" or "Adds").
- Pull request description explains the **why** -- the diff already shows the what.
- Link the issue you're closing, if any.

## Reporting bugs

Open an issue with: Python version, `ytcc-pipeline` version, a minimal reproducer, and the full traceback. Vague reports get vague answers.

## Code of conduct

By participating you agree to follow the project [Code of Conduct](CODE_OF_CONDUCT.md).
