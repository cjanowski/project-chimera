# Project Chimera

Experimentation framework to compare Dense Transformer vs Mixture-of-Experts (MoE) models.

This repository provides:
- Python 3.10 environment via pyproject.toml
- Reproducibility and device utilities
- Project structure for src, tests, notebooks, and scripts
- Basic CI (GitHub Actions) for linting and tests (to be added)
- Pre-commit hooks (to be added)

## Project Structure

- src/project_chimera: Python package with core modules
- tests: Unit tests
- notebooks: Jupyter notebooks for exploration
- scripts: CLI utilities (training, data download)
- data: Dataset directory with a README

## Quickstart

1) Create a virtual environment and install:
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e ".[dev]"

2) Run tests:
   pytest

3) Format/lint (once pre-commit is installed later):
   pre-commit install
   pre-commit run --all-files

## Python Version

Python 3.10.x (see pyproject.toml).

## License

MIT