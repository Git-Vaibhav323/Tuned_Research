# ResearchPilot — Project Scope

**Document type:** Scope & boundaries  
**Phase:** 1 — Foundation  
**Version:** MVP v1 scope freeze (planning)

---

## 1. Product Statement

ResearchPilot is a **data science and human-centered research intelligence system** developed for a Programming for Data Science course. It first builds a documented analytical dataset and predictive modeling workflow, then uses those assets within an AI-assisted research product.

---

## 2. In Scope

### DA1 — Data science foundation

- Dataset collection, provenance, licensing, and documentation
- Cleaning, preprocessing, validation, and exploratory data analysis
- Python and R visualizations
- Analysis-ready artifacts for downstream work

### DA2 — Database and modeling

- Database schema, ingestion, and analytical SQL
- Feature engineering
- Multiple suitable ML/DL algorithms and a baseline
- Evaluation metrics, error analysis, and comparative visualizations

### DA3 — AI application

### Modules

1. **Adaptive Research Translator**  
   - Beginner-friendly rewrites  
   - Key takeaways  
   - Concept explanations with examples  

2. **Statistical Advisor**  
   - Test recommendation  
   - Selection rationale  
   - Assumptions  
   - Alternatives  

3. **Abstract Improver**  
   - Grammar  
   - Academic tone  
   - Clarity  
   - Novelty framing suggestions  

4. **Research Gap Finder**  
   - Multi-paper comparison  
   - Limitations  
   - Future work suggestions  
   - Gap idea generation  

### Platform (across phases)

- Document ingest (PDF/text)  
- Structured dataset analysis and database-backed queries
- Conventional machine-learning inference
- Optional retrieval grounding  
- Fine-tuned model inference  
- Evaluation harness and benchmarks  
- Demo API + UI  

### Language

- English for MVP  

---

## 3. Out of Scope (MVP)

The following are **explicitly excluded** from Version 1:

| Area | Notes |
|------|-------|
| Full automated literature review | Beyond gap finder inputs |
| General-purpose analysis of arbitrary user datasets | DA1/DA2 use the selected course dataset; the advisor remains advisory |
| Journal submission / peer review automation | Writing help only for abstracts |
| Multilingual UI/models | Deferred |
| Real-time collaboration / multi-user auth | Deferred |
| Mobile native apps | Deferred |
| Proprietary closed-model SaaS dependency as sole path | Prefer open/self-hostable stack |
| Legal advice, medical diagnosis, or clinical decision support | Never |

---

## 4. Foundation Scope (This Repository State)

### Included now

- Folder structure and placeholders  
- Dedicated DA1, DA2, and DA3 boundaries
- Documentation set under `docs/`  
- Dependency manifest (`requirements.txt`)  
- Community files (LICENSE, CONTRIBUTING, CHANGELOG, .gitignore)  

### Excluded now

- Backend APIs  
- UI implementation  
- ML / training / RAG / embeddings / vector DB code  
- Datasets and model weights  
- Install / runtime environment setup beyond docs  

---

## 5. Assumptions

1. Contributors can run Python 3.10+ environments in later phases.  
2. The chosen dataset supports both meaningful EDA and a defensible DA2 modeling task.
3. Compute for deep learning or LLM work will be available by DA2/DA3 if those approaches are justified.  
4. Sufficient open-licensed or author-permitted data can be curated for project quality.  
5. Users understand outputs are assistive and require human judgment.  
6. MVP targets researchers, students, and interdisciplinary collaborators—not production clinical/legal use.

---

## 6. Constraints

- Respect copyright and dataset licenses.  
- Keep repository professional and reproducible.  
- Prefer modular design over monolith scripts.  
- Avoid shipping secrets, private papers, or credentials.  

---

## 7. Success Definition (MVP)

The course project is successful when:

1. DA1 produces a documented, clean dataset with reproducible Python/R analysis.
2. DA2 demonstrates database integration, feature engineering, and fair comparison of multiple ML/DL algorithms.
3. DA3 exposes data insights, model results, and four assistant modules through an interactive dashboard.
4. A user can receive faithful translations, statistical guidance, abstract improvements, and grounded research-gap ideas.
5. Final evaluation reports data quality, predictive-model performance, and assistant quality as separate evidence.

---

## 8. Stakeholders

| Role | Interest |
|------|----------|
| Researchers / students | Usable assistance |
| Maintainers | Clean architecture, reproducible science |
| Contributors | Clear contribution paths |
| Evaluators | Transparent benchmarks |

---

## 9. Change Control

Scope changes (new modules, languages, or deployment targets) should be proposed via issue / ADR and reflected in:

- This document  
- [development_roadmap.md](development_roadmap.md)  
- [CHANGELOG.md](../CHANGELOG.md)  

---

## 10. Related Documents

- [architecture.md](architecture.md)  
- [dataset_plan.md](dataset_plan.md)  
- [benchmark_plan.md](benchmark_plan.md)  
- [development_roadmap.md](development_roadmap.md)  
