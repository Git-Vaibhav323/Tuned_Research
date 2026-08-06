-- Q7: Model leaderboard placeholder
-- Purpose: Ready for M4+; returns empty until experiment_runs is populated.

SELECT
    track,
    model_name,
    cv_score,
    metrics_json,
    created_at
FROM experiment_runs
ORDER BY cv_score DESC
LIMIT 20;
