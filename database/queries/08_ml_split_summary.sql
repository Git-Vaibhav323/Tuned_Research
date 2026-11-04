-- Q8: ML feature store split summary (M2+)
-- Purpose: Confirm stratified train/val/test counts after feature build.

SELECT
    split,
    COUNT(*) AS n_papers,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM ml_features), 2) AS pct
FROM ml_features
GROUP BY split
ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'val' THEN 2 WHEN 'test' THEN 3 END;
