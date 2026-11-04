-- Q3: Papers by publication year
-- Purpose: Temporal coverage of the OpenAlex AI/ML corpus (2022–2025).

SELECT
    publication_year,
    COUNT(*) AS n_papers,
    ROUND(AVG(cited_by_count), 1) AS avg_citations,
    ROUND(AVG(paper_age), 2) AS avg_paper_age,
    SUM(is_open_access) AS n_open_access
FROM papers
GROUP BY publication_year
ORDER BY publication_year;
