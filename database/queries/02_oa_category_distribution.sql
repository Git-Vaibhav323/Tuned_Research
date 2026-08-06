-- Q2: Open-access category distribution
-- Purpose: Class balance for Phase 2 primary task (oa_category classification).

SELECT
    oa_category,
    COUNT(*) AS n_papers,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM papers), 2) AS pct
FROM papers
GROUP BY oa_category
ORDER BY n_papers DESC;
