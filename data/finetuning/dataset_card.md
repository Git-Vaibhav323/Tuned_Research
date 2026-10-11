# ResearchPilot Fine-Tuning Dataset

**Generated:** 2026-10-11 06:02 UTC
**Source:** data/final/final_dataset.csv (OpenAlex API, 2,000 AI/ML papers)
**Seed:** 42

## Splits

| Split | Papers | Examples |
|-------|--------|----------|
| train | 1393 | 8343 |
| validation | 298 | 1785 |
| test | 299 | 1794 |
| **total** | **1990** | **11922** |

## Templates

| Template | Purpose | Count |
|----------|---------|-------|
| tmpl_summarise | Structured abstract summary | 1990 |
| tmpl_oa_explain | Open-access status explanation | 1990 |
| tmpl_research_questions | Inferred research questions | 1984 |
| tmpl_methodology | Methodology inference | 1984 |
| tmpl_limitations | Likely limitations | 1984 |
| tmpl_citation_context | Citation impact explanation | 1990 |

## Important notes

- ALL examples are **synthetic** — generated from abstract metadata by rule-based templates.
- NO examples have been human-reviewed. Use `review_template.jsonl` to add verified examples.
- The system prompt instructs the model to admit uncertainty and avoid fabrication.
- The 2,000 papers are used as a RAG knowledge base (Phase 3), not as training answers.
- Do NOT use this dataset to claim the model "knows" the content of these papers.
- Split is by paper ID — no paper's examples appear in more than one split.

## Schema

Each JSONL line: `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}], "_meta": {...}}`

The `_meta` field is stripped before sending to the model during training.
