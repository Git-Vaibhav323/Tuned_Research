# Contributing to ResearchPilot

Thanks for your interest. This is a personal research project but contributions are welcome.

---

## How to contribute

1. Fork the repo and create a branch from `main`
2. Make your changes — keep commits small and focused
3. Test that the Streamlit app and dashboard still launch
4. Open a pull request with a clear description

---

## What's useful

- Bug fixes in the RAG pipeline (`rag/`)
- Improvements to the Streamlit UI (`streamlit_app/app.py`)
- Additional Shiny dashboard tabs (`dashboard/app.R`)
- New ML pipeline scripts (`ml/`)
- Documentation improvements (`docs/`)

---

## What to avoid

- Committing data files, model weights, or figures — they are gitignored for a reason
- Changing the locked dataset or model evaluation results
- Adding cloud API dependencies — the project is intentionally fully local

---

## Code style

- Python: follow PEP 8, use type hints where practical
- R: follow tidyverse conventions
- Keep functions small and single-purpose
- Add a docstring / comment block to any new function

---

## Reporting bugs

Open an issue with:
- What you were trying to do
- What happened vs what you expected
- Your OS, Python version, R version, Ollama version
- Relevant terminal output
