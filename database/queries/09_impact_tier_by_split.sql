-- Q9: Impact tier distribution by split (M2+)
-- Purpose: Check relative impact labels after train-only tertile thresholds.

SELECT
    split,
    impact_tier,
    COUNT(*) AS n_papers
FROM ml_features
GROUP BY split, impact_tier
ORDER BY split, impact_tier;
