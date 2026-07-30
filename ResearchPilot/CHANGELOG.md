# Changelog

All notable changes to **ResearchPilot** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once tagged releases begin.

---

## [Unreleased]

### Planned

- DA1: dataset documentation, cleaning, preprocessing, EDA, and R visualizations
- DA2: database integration, feature engineering, ML/DL comparison, and metrics
- DA3: AI research assistant, LLM integration, dashboard, and final evaluation

### Changed

- Reframed the architecture as a complete data science workflow rather than an LLM-first repository
- Added `src/researchpilot/` packages for data, features, models, visualization, and assistant layers
- Added `database/`, `r/`, `reports/`, `data/external/`, and `data/interim/`
- Updated architecture, scope, dataset, benchmark, roadmap, dependencies, and contribution guidance for DA1/DA2/DA3

---

## [0.1.0-foundation] — 2026-07-30

### Added

- Initial professional repository structure for ResearchPilot
- Data, notebooks, backend, frontend, models, evaluation, scripts, docs, tests, and configs directories
- Placeholder Python packages with `__init__.py` under backend, evaluation, scripts, and tests
- Comprehensive `README.md` (overview, problem, objectives, features, architecture, stack, roadmap)
- Documentation set:
  - `docs/architecture.md`
  - `docs/project_architecture.md`
  - `docs/dataset_plan.md`
  - `docs/development_roadmap.md`
  - `docs/project_scope.md`
  - `docs/benchmark_plan.md`
- `requirements.txt` with planned dependencies (not installed)
- `.gitignore` tailored for Python AI/ML projects
- `LICENSE` (MIT)
- `CONTRIBUTING.md`
- `CHANGELOG.md`

### Notes

- The foundation release contains **documentation and organization only**
- No APIs, UI, ML pipelines, datasets, or vector database code in this release marker

---

[Unreleased]: https://github.com/example/ResearchPilot/compare/v0.1.0-foundation...HEAD
[0.1.0-foundation]: https://github.com/example/ResearchPilot/releases/tag/v0.1.0-foundation
