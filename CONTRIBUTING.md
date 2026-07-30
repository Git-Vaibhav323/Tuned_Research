# Contributing to ResearchPilot

Thank you for your interest in contributing to **ResearchPilot** — a human-centered research intelligence system.

The repository foundation is complete. Contributions should identify whether they belong to **DA1 (data/EDA), DA2 (database/modeling), or DA3 (assistant/dashboard)** and preserve the dependency order between those phases.

---

## Code of Conduct

Be respectful, constructive, and inclusive. Assume good intent. Harassment or discriminatory behavior is not tolerated.

---

## How to Contribute

### 1. Open an issue first (recommended)

For non-trivial changes, open an issue describing:

- The problem or opportunity  
- Proposed approach  
- Affected modules / folders  
- Phase alignment (see `docs/development_roadmap.md`)  

### 2. Fork and branch

```bash
git checkout -b feature/short-description
# or
git checkout -b docs/short-description
```

Use clear branch names: `feature/`, `fix/`, `docs/`, `chore/`.

### 3. Follow project boundaries

| Do | Don't |
|----|-------|
| Keep changes scoped | Expand MVP scope without discussion |
| Update docs when behavior/plans change | Commit large model weights or private papers |
| Respect licenses for any sample data | Add secrets, API keys, or `.env` files |
| Put reusable code under `src/researchpilot/` | Skip DA1/DA2 and treat the project as LLM-only |

### 4. Documentation standards

- Prefer clear Markdown with headings and tables  
- Link related docs under `docs/`  
- Avoid duplicating the same design in three places—link instead  

### 5. Python conventions (when code lands)

- Python 3.10+  
- Reusable data, feature, model, visualization, and assistant code under `src/researchpilot/`  
- Backend-specific packages under `backend/` and evaluation code under `evaluation/`
- Tests under `tests/unit/` and `tests/integration/`  
- Configs in `configs/` rather than hard-coded paths  

### 6. Commit messages

Use concise, imperative subjects:

```text
Add dataset schema outline to dataset_plan.md
Fix typo in architecture module contracts
```

### 7. Pull requests

PRs should include:

- Summary of changes  
- Linked issue (if any)  
- Notes on docs updated  
- Confirmation that no large binaries / secrets are included  

Maintainers may ask for smaller PRs if a change is too broad.

---

## Development Setup

The foundation does **not** require installing dependencies. When implementation begins:

1. Create a virtual environment  
2. Install from `requirements.txt` (versions will be pinned later)  
3. Follow phase-specific quickstarts in `docs/` and `README.md`  

---

## Dataset & model contributions

- Do **not** upload copyrighted PDFs or unlicensed corpora.  
- Document license and provenance in `data/metadata/`.  
- Prefer links + scripts over committing bulk raw papers.  
- Model checkpoints generally stay out of git (see `.gitignore`).  

---

## Security & responsible use

- Never commit credentials.  
- Flag potential unsafe statistical or medical advice patterns in reviews.  
- ResearchPilot outputs are assistive—not professional legal/medical advice.  

---

## Questions

Open a GitHub issue with the `question` label (when the remote is configured), or discuss via the project maintainers’ preferred channel.

---

Thank you for helping build research tools that stay transparent, modular, and human-centered.
