-- Q6: Latest load provenance
-- Purpose: Prove Phase 1 CSV path and row count recorded at ingest time.

SELECT
    load_id,
    source_csv,
    db_path,
    n_rows_loaded,
    n_columns,
    phase1_locked,
    notes,
    loaded_at
FROM load_manifest
ORDER BY load_id DESC
LIMIT 5;
