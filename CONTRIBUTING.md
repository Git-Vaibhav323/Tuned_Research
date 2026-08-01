# Contributing to ResearchPilot

Thank you for contributing to **ResearchPilot**.

Work must align with the active phase:

1. **Phase 1** — Dataset collection, preprocessing, EDA, documentation  
2. **Phase 2** — Machine learning and comparative analysis  
3. **Phase 3** — Fine-tuning, RAG, dashboard  

Do not skip ahead and make the repository LLM-only.

---

## Guidelines

| Do | Don't |
|----|-------|
| Put reusable code under `src/researchpilot/` | Hide entire pipelines only in notebooks |
| Use phase configs under `configs/phase*/` | Commit secrets, private papers, or large weights |
| Document licenses in `data/metadata/` | Scrape paywalled content |
| Update docs when plans change | Expand scope without discussion |

---

## Branch naming

```text
feature/phase1-cleaning
feature/phase2-model-comparison
docs/update-dataset-plan
```

---

## Setup

Foundation does not require installing packages. When a phase starts:

1. Create a virtual environment  
2. Install from `requirements.txt`  
3. Follow `docs/development_roadmap.md`  

---

## License and data

- Prefer open-licensed sources  
- Record provenance  
- Keep large raw files out of git when possible  

---

## Security

Never commit credentials or `.env` files with secrets. ResearchPilot outputs are assistive, not professional legal/medical advice.
