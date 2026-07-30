# Database Assets

This directory is reserved for DA2 database integration.

- `schemas/` — logical/physical schema definitions
- `migrations/` — versioned schema changes
- `queries/` — documented analytical SQL

SQLite is suitable for a portable course demonstration; PostgreSQL may be used if deployment requirements justify it. The final choice should be recorded before implementation.

The database will ingest validated DA1 outputs from `data/processed/`. Cleaning logic remains in `src/researchpilot/data/`; SQL should not become an undocumented parallel preprocessing pipeline.
