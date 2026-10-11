"""
ResearchPilot — training/prepare_dataset.py

Builds a fine-tuning dataset from the existing OpenAlex paper corpus.

IMPORTANT DESIGN DECISIONS (read before modifying):
  - The 2,000 OpenAlex papers are NOT treated as a supervised Q&A dataset.
  - Each abstract is used to generate templated instruction examples.
  - Templates are deterministic and rule-based — no LLM generation here.
  - Every example is clearly labelled with its source paper ID.
  - Synthetic examples are flagged as synthetic=True so reviewers know.
  - The 2,000 papers are the FUTURE RAG knowledge base, not training data.
  - A small human-review template is also generated for real annotation.

Run:
    py -3.12 training/prepare_dataset.py

Output:
    data/finetuning/train.jsonl
    data/finetuning/validation.jsonl
    data/finetuning/test.jsonl
    data/finetuning/dataset_card.md
    data/finetuning/smoke_test.jsonl   (5 examples for quick sanity check)
"""

import csv
import json
import hashlib
import random
import re
import sys
import os
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data" / "finetuning"
SRC_CSV   = ROOT / "data" / "final" / "final_dataset.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH  = DATA_DIR / "train.jsonl"
VAL_PATH    = DATA_DIR / "validation.jsonl"
TEST_PATH   = DATA_DIR / "test.jsonl"
SMOKE_PATH  = DATA_DIR / "smoke_test.jsonl"
CARD_PATH   = DATA_DIR / "dataset_card.md"
REVIEW_PATH = DATA_DIR / "review_template.jsonl"

SEED = 42
random.seed(SEED)

# ── helpers ───────────────────────────────────────────────────────────────────
def clean_text(t: str) -> str:
    """Strip leading/trailing whitespace, collapse internal whitespace."""
    t = t.strip()
    t = re.sub(r"\s+", " ", t)
    return t

def paper_id_hash(openalex_id: str) -> str:
    """Short deterministic key for a paper."""
    return hashlib.md5(openalex_id.encode()).hexdigest()[:8]

def split_concepts(raw: str) -> list[str]:
    """Parse concepts_clean field into a list of up to 6 terms."""
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(";")]
    return [p for p in parts if len(p) > 2][:6]

def oa_label(status: str) -> str:
    mapping = {
        "gold":    "fully open access (gold OA — published in an open-access journal)",
        "green":   "open access via a preprint or repository (green OA)",
        "hybrid":  "hybrid open access (paid OA in a subscription journal)",
        "bronze":  "free to read on the publisher site but not formally open access (bronze)",
        "diamond": "diamond open access (free to read and publish, no APC)",
        "closed":  "closed access (paywalled)",
    }
    return mapping.get(status.lower(), status)

def make_conversation(system: str, user: str, assistant: str) -> dict:
    """Build a messages-format example for SFT."""
    return {
        "messages": [
            {"role": "system",    "content": system.strip()},
            {"role": "user",      "content": user.strip()},
            {"role": "assistant", "content": assistant.strip()},
        ]
    }

SYSTEM_PROMPT = (
    "You are ResearchPilot, an AI research assistant specialising in AI and machine "
    "learning literature. You help researchers understand papers, methods, and findings. "
    "You clearly distinguish what the source material says from your own interpretation. "
    "When you are uncertain, you say so. You never fabricate citations, results, or "
    "author names."
)

# ── template factories ────────────────────────────────────────────────────────

def tmpl_summarise(row: dict) -> dict | None:
    """Ask the model to summarise a paper from its title and abstract."""
    title    = clean_text(row["title"])
    abstract = clean_text(row["abstract"])
    year     = row.get("publication_year", "")
    if len(abstract) < 150:
        return None
    user = (
        f"Please summarise the following research paper in plain language "
        f"so that a postgraduate student can understand it quickly.\n\n"
        f"Title: {title}\n"
        f"Year: {year}\n"
        f"Abstract: {abstract}"
    )
    # Build a structured summary from the available fields
    concepts = split_concepts(row.get("concepts_clean", ""))
    concepts_str = ", ".join(concepts) if concepts else "not specified"
    oa         = oa_label(row.get("oa_status", ""))
    cited      = row.get("cited_by_count", "unknown")
    assistant_text = (
        f"**Paper:** {title} ({year})\n\n"
        f"**What it is about:** This paper addresses topics in {concepts_str}. "
        f"Based on the abstract: {abstract[:300].rstrip('.')}.\n\n"
        f"**Access:** This paper is {oa}.\n\n"
        f"**Impact:** It has received approximately {cited} citations.\n\n"
        f"*Note: This summary is generated from the title and abstract only. "
        f"For a full analysis, read the complete paper.*"
    )
    return make_conversation(SYSTEM_PROMPT, user, assistant_text)


def tmpl_oa_explain(row: dict) -> dict | None:
    """Explain the open-access status of a paper."""
    title  = clean_text(row["title"])
    status = row.get("oa_status", "")
    cat    = row.get("oa_category", "")
    url    = row.get("oa_url", "")
    if not status:
        return None
    user = (
        f"Is this paper freely available? How can I access it?\n\n"
        f"Title: {title}"
    )
    label = oa_label(status)
    url_note = (
        f" It may be accessible at: {url}" if url and url.startswith("http") else ""
    )
    assistant_text = (
        f"This paper is classified as **{label}** (OA category: {cat}).\n\n"
        f"{url_note}\n\n"
        f"*Open-access status is sourced from OpenAlex and reflects the metadata "
        f"at collection time. Availability may have changed.*"
    ).strip()
    return make_conversation(SYSTEM_PROMPT, user, assistant_text)


def tmpl_research_questions(row: dict) -> dict | None:
    """Generate research questions a paper might answer."""
    title    = clean_text(row["title"])
    abstract = clean_text(row["abstract"])
    if len(abstract) < 200:
        return None
    concepts = split_concepts(row.get("concepts_clean", ""))
    topics   = " and ".join(concepts[:3]) if concepts else "the described topic"
    user = (
        f"Based on the title and abstract of this paper, what research questions "
        f"does it appear to address?\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract[:400]}"
    )
    assistant_text = (
        f"Based on the title and abstract, this paper appears to address the "
        f"following research questions:\n\n"
        f"1. How can {topics} be applied or improved in the context described?\n"
        f"2. What are the performance characteristics or trade-offs of the proposed "
        f"approach compared to existing methods?\n"
        f"3. What empirical evidence supports the claims made in the paper?\n\n"
        f"*These questions are inferred from the abstract. The paper itself may "
        f"frame its contributions differently. Always read the introduction and "
        f"conclusion sections for the authors' own framing.*"
    )
    return make_conversation(SYSTEM_PROMPT, user, assistant_text)


def tmpl_methodology(row: dict) -> dict | None:
    """Ask about methodology implied by the abstract."""
    title    = clean_text(row["title"])
    abstract = clean_text(row["abstract"])
    if len(abstract) < 200:
        return None
    user = (
        f"What research methodology does this paper appear to use? "
        f"What kind of study is it?\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract[:500]}"
    )
    # Detect methodology keywords for a structured answer
    ab_lower = abstract.lower()
    method_hints = []
    if any(w in ab_lower for w in ["dataset", "benchmark", "trained on", "fine-tun"]):
        method_hints.append("empirical evaluation on a dataset or benchmark")
    if any(w in ab_lower for w in ["propose", "novel", "we introduce", "we present"]):
        method_hints.append("presentation of a new method or architecture")
    if any(w in ab_lower for w in ["survey", "review", "literature", "systematically"]):
        method_hints.append("literature survey or systematic review")
    if any(w in ab_lower for w in ["experiment", "ablation", "compari"]):
        method_hints.append("experimental comparison or ablation study")
    if any(w in ab_lower for w in ["theorem", "proof", "theoretical"]):
        method_hints.append("theoretical analysis")
    if not method_hints:
        method_hints.append("empirical study (inferred from abstract)")
    methods_str = "; ".join(method_hints)
    assistant_text = (
        f"Based on the abstract, this paper appears to employ: **{methods_str}**.\n\n"
        f"Key signals from the abstract:\n"
        f"{abstract[:300].rstrip('.')}\n\n"
        f"*Methodology is inferred from language patterns in the abstract. "
        f"The Methods section of the full paper provides authoritative detail.*"
    )
    return make_conversation(SYSTEM_PROMPT, user, assistant_text)


def tmpl_limitations(row: dict) -> dict | None:
    """Discuss likely limitations based on paper type and abstract."""
    title    = clean_text(row["title"])
    abstract = clean_text(row["abstract"])
    concepts = split_concepts(row.get("concepts_clean", ""))
    if len(abstract) < 200:
        return None
    user = (
        f"What are the likely limitations of a paper with this title and abstract? "
        f"What should a reader be cautious about?\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract[:400]}"
    )
    field = concepts[0] if concepts else "AI/ML"
    assistant_text = (
        f"Without reading the full paper, common limitations for work in **{field}** "
        f"of this type include:\n\n"
        f"1. **Dataset scope** — results may not generalise beyond the specific "
        f"dataset or domain used.\n"
        f"2. **Computational constraints** — the approach may require resources "
        f"not available to all researchers.\n"
        f"3. **Evaluation metrics** — the chosen metrics may not capture all "
        f"relevant aspects of performance.\n"
        f"4. **Reproducibility** — implementation details may be incomplete.\n\n"
        f"*I cannot identify the actual limitations without reading the paper. "
        f"Check the Limitations and Discussion sections of the full text for the "
        f"authors' own assessment.*"
    )
    return make_conversation(SYSTEM_PROMPT, user, assistant_text)


def tmpl_citation_context(row: dict) -> dict | None:
    """Explain what high/low citation count means for a paper."""
    title  = clean_text(row["title"])
    cited  = row.get("cited_by_count", "")
    year   = row.get("publication_year", "")
    cpy    = row.get("citation_per_year", "")
    try:
        cited_n = int(float(cited))
        cpy_n   = float(cpy) if cpy else 0.0
    except (ValueError, TypeError):
        return None
    if cited_n < 0:
        return None
    user = (
        f"How influential is this paper? Should I read it?\n\n"
        f"Title: {title}\n"
        f"Year: {year}\n"
        f"Citations: {cited_n}"
    )
    if cpy_n > 500:
        influence = "extremely high — this is a landmark paper in its area"
    elif cpy_n > 100:
        influence = "very high — this is a widely cited paper"
    elif cpy_n > 20:
        influence = "above average — well-cited in its field"
    elif cpy_n > 5:
        influence = "moderate — reasonable citation rate"
    else:
        influence = "early-stage or niche — relatively few citations so far"
    assistant_text = (
        f"This paper has received **{cited_n} citations** since {year} "
        f"(~{cpy_n:.0f}/year), which is **{influence}**.\n\n"
        f"Whether you should read it depends on your research needs. "
        f"Citation count indicates influence but not correctness — a paper "
        f"can be highly cited and still have been superseded or partially "
        f"challenged by subsequent work.\n\n"
        f"*Recommendation: read the abstract, check recent papers that cite "
        f"this one, then decide if the full text is worth your time.*"
    )
    return make_conversation(SYSTEM_PROMPT, user, assistant_text)


# ordered list of template functions
TEMPLATES = [
    tmpl_summarise,
    tmpl_oa_explain,
    tmpl_research_questions,
    tmpl_methodology,
    tmpl_limitations,
    tmpl_citation_context,
]

# ── load papers ───────────────────────────────────────────────────────────────
print(f"Loading papers from {SRC_CSV} ...")
with open(SRC_CSV, encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
print(f"  Loaded {len(rows)} rows.")

# Filter: must have a usable abstract
rows = [r for r in rows if len(clean_text(r.get("abstract", ""))) >= 150]
print(f"  After abstract filter (>=150 chars): {len(rows)} rows.")

# ── stratified split by paper (no leakage between splits) ────────────────────
# Group by paper id so all examples from one paper go to one split.
# Split: 70% train, 15% val, 15% test.
paper_ids = [r["id"] for r in rows]
unique_ids = list(dict.fromkeys(paper_ids))  # preserve order, deduplicate
random.shuffle(unique_ids)

n = len(unique_ids)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

train_ids = set(unique_ids[:n_train])
val_ids   = set(unique_ids[n_train : n_train + n_val])
test_ids  = set(unique_ids[n_train + n_val:])

print(f"\n  Papers → train: {len(train_ids)} | val: {len(val_ids)} | test: {len(test_ids)}")

# ── generate examples ─────────────────────────────────────────────────────────
examples_by_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
skipped = 0
template_counts: Counter = Counter()

for row in rows:
    pid = row["id"]
    if pid in train_ids:
        split = "train"
    elif pid in val_ids:
        split = "validation"
    else:
        split = "test"

    for tmpl_fn in TEMPLATES:
        ex = tmpl_fn(row)
        if ex is None:
            skipped += 1
            continue
        # Attach metadata (not sent to model, used for auditing)
        ex["_meta"] = {
            "source_id":    pid,
            "source_hash":  paper_id_hash(pid),
            "template":     tmpl_fn.__name__,
            "split":        split,
            "synthetic":    True,
            "reviewed":     False,
            "created_at":   datetime.now(datetime.timezone.utc).isoformat() + "Z",
        }
        examples_by_split[split].append(ex)
        template_counts[tmpl_fn.__name__] += 1

# Shuffle within each split
for split in examples_by_split:
    random.shuffle(examples_by_split[split])

total = sum(len(v) for v in examples_by_split.values())
print(f"\n  Generated {total} examples  ({skipped} skipped by templates)")
for split, exs in examples_by_split.items():
    print(f"    {split}: {len(exs)}")
print("\n  Template breakdown:")
for name, cnt in template_counts.most_common():
    print(f"    {name}: {cnt}")

# ── validate examples ─────────────────────────────────────────────────────────
print("\nValidating examples ...")
errors = 0
seen_hashes: set[str] = set()

def validate_example(ex: dict, idx: int, split: str) -> list[str]:
    issues = []
    msgs = ex.get("messages", [])
    if not isinstance(msgs, list) or len(msgs) != 3:
        issues.append(f"[{split}#{idx}] Expected 3 messages, got {len(msgs)}")
        return issues
    roles = [m.get("role") for m in msgs]
    if roles != ["system", "user", "assistant"]:
        issues.append(f"[{split}#{idx}] Wrong role order: {roles}")
    for m in msgs:
        if not m.get("content", "").strip():
            issues.append(f"[{split}#{idx}] Empty content in role={m.get('role')}")
    # Deduplication check
    content_hash = hashlib.md5(
        (msgs[1]["content"] + msgs[2]["content"]).encode()
    ).hexdigest()
    if content_hash in seen_hashes:
        issues.append(f"[{split}#{idx}] Duplicate example detected")
    seen_hashes.add(content_hash)
    # Response should not be empty
    resp = msgs[2].get("content", "")
    if len(resp) < 30:
        issues.append(f"[{split}#{idx}] Assistant response too short ({len(resp)} chars)")
    return issues

for split, exs in examples_by_split.items():
    for i, ex in enumerate(exs):
        issues = validate_example(ex, i, split)
        for issue in issues:
            print(f"  WARN: {issue}")
            errors += 1

if errors == 0:
    print("  All examples valid.")
else:
    print(f"  {errors} validation issues found (see WARNs above).")

# ── write JSONL files ─────────────────────────────────────────────────────────
print("\nWriting JSONL files ...")
for split, exs in examples_by_split.items():
    path = {"train": TRAIN_PATH, "validation": VAL_PATH, "test": TEST_PATH}[split]
    with open(path, "w", encoding="utf-8") as fh:
        for ex in exs:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(exs)} examples  ({path.stat().st_size // 1024} KB)")

# ── smoke test (5 examples, one per key template) ─────────────────────────────
smoke = examples_by_split["test"][:5]
with open(SMOKE_PATH, "w", encoding="utf-8") as fh:
    for ex in smoke:
        fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
print(f"\n  smoke_test.jsonl: {len(smoke)} examples")

# ── human review template (10 blank slots for real annotation) ────────────────
review_template = {
    "messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": "[FILL IN: user question about a real paper]"},
        {"role": "assistant", "content": "[FILL IN: verified answer — must be grounded in the paper]"},
    ],
    "_meta": {
        "source_id":    "[FILL: OpenAlex ID or DOI]",
        "template":     "human_reviewed",
        "split":        "train",
        "synthetic":    False,
        "reviewed":     True,
        "reviewer":     "[FILL: your name]",
        "created_at":   datetime.now(datetime.timezone.utc).isoformat() + "Z",
    }
}
with open(REVIEW_PATH, "w", encoding="utf-8") as fh:
    for _ in range(10):
        fh.write(json.dumps(review_template, ensure_ascii=False) + "\n")
print(f"  review_template.jsonl: 10 blank slots for human-reviewed examples")

# ── dataset card ──────────────────────────────────────────────────────────────
card = f"""# ResearchPilot Fine-Tuning Dataset

**Generated:** {datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Source:** data/final/final_dataset.csv (OpenAlex API, 2,000 AI/ML papers)
**Seed:** {SEED}

## Splits

| Split | Papers | Examples |
|-------|--------|----------|
| train | {len(train_ids)} | {len(examples_by_split['train'])} |
| validation | {len(val_ids)} | {len(examples_by_split['validation'])} |
| test | {len(test_ids)} | {len(examples_by_split['test'])} |
| **total** | **{len(unique_ids)}** | **{total}** |

## Templates

| Template | Purpose | Count |
|----------|---------|-------|
| tmpl_summarise | Structured abstract summary | {template_counts['tmpl_summarise']} |
| tmpl_oa_explain | Open-access status explanation | {template_counts['tmpl_oa_explain']} |
| tmpl_research_questions | Inferred research questions | {template_counts['tmpl_research_questions']} |
| tmpl_methodology | Methodology inference | {template_counts['tmpl_methodology']} |
| tmpl_limitations | Likely limitations | {template_counts['tmpl_limitations']} |
| tmpl_citation_context | Citation impact explanation | {template_counts['tmpl_citation_context']} |

## Important notes

- ALL examples are **synthetic** — generated from abstract metadata by rule-based templates.
- NO examples have been human-reviewed. Use `review_template.jsonl` to add verified examples.
- The system prompt instructs the model to admit uncertainty and avoid fabrication.
- The 2,000 papers are used as a RAG knowledge base (Phase 3), not as training answers.
- Do NOT use this dataset to claim the model "knows" the content of these papers.
- Split is by paper ID — no paper's examples appear in more than one split.

## Schema

Each JSONL line: `{{"messages": [{{"role": "system", ...}}, {{"role": "user", ...}}, {{"role": "assistant", ...}}], "_meta": {{...}}}}`

The `_meta` field is stripped before sending to the model during training.
"""
with open(CARD_PATH, "w", encoding="utf-8") as fh:
    fh.write(card)
print(f"\n  dataset_card.md written.")

print("\n" + "=" * 55)
print("  prepare_dataset.py  COMPLETE")
print(f"  Total examples: {total}")
print(f"  Validation issues: {errors}")
print("=" * 55)
