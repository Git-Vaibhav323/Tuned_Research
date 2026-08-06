-- Q4: OpenAlex OA status vs ResearchPilot oa_category
-- Purpose: Show how fine-grained oa_status maps into the 3-class modeling target.

SELECT
    oa_status,
    oa_category,
    COUNT(*) AS n_papers
FROM papers
GROUP BY oa_status, oa_category
ORDER BY oa_category, n_papers DESC;
