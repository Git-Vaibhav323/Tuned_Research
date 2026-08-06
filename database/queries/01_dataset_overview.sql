-- Q1: Dataset overview after M1 load
-- Purpose: Confirm row count and basic coverage from Phase 1 CSV.

SELECT
    COUNT(*) AS n_papers,
    COUNT(DISTINCT publication_year) AS n_years,
    MIN(publication_year) AS year_min,
    MAX(publication_year) AS year_max,
    ROUND(AVG(cited_by_count), 1) AS avg_citations,
    ROUND(AVG(citation_log), 3) AS avg_citation_log
FROM papers;
