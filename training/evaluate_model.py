"""
ResearchPilot — training/evaluate_model.py

Compares base model vs fine-tuned model on held-out test questions.
Works via Ollama API (no GPU required after training).

Run:
    py -3.12 training/evaluate_model.py

Requirements:
  - Ollama running (ollama serve or tray app)
  - Either researchpilot model imported, OR qwen2.5:1.5b-instruct pulled
  - requests package installed

Output:
    evaluation/reports/eval_results.json
    evaluation/reports/eval_report.md
"""

import json
import time
import csv
import sys
from pathlib import Path
from datetime import datetime

try:
    import requests
except ImportError:
    print("ERROR: requests not installed.  pip install requests")
    sys.exit(1)

ROOT        = Path(__file__).resolve().parent.parent
EVAL_DIR    = ROOT / "evaluation" / "reports"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = EVAL_DIR / "eval_results.json"
REPORT_MD    = EVAL_DIR / "eval_report.md"

OLLAMA_URL   = "http://127.0.0.1:11434"
BASE_MODEL   = "qwen2.5:1.5b-instruct"   # base model tag in Ollama
TUNED_MODEL  = "researchpilot"            # fine-tuned model tag in Ollama

# ── ResearchPilot system prompt ───────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are ResearchPilot, an AI research assistant specialising in AI and machine "
    "learning literature. You help researchers understand papers, methods, and findings. "
    "You clearly distinguish what the source material says from your own interpretation. "
    "When you are uncertain, you say so. You never fabricate citations, results, or "
    "author names."
)

# ── held-out evaluation questions (from test.jsonl schema but manually crafted) ─
EVAL_QUESTIONS = [
    {
        "id": "q1",
        "category": "summarisation",
        "user": (
            "Summarise this paper for a postgraduate student:\n\n"
            "Title: Attention Is All You Need\n"
            "Abstract: We propose a new simple network architecture, the Transformer, "
            "based solely on attention mechanisms, dispensing with recurrence and "
            "convolutions entirely. Experiments on two machine translation tasks show "
            "these models to be superior in quality while being more parallelizable "
            "and requiring significantly less time to train."
        ),
        "rubric": {
            "should_mention": ["attention mechanism", "transformer", "parallelizable",
                               "translation", "no recurrence"],
            "should_not_claim": ["we trained the model", "I implemented this",
                                  "this is my paper"],
            "must_caveat": True,  # should note it's based on abstract
        },
    },
    {
        "id": "q2",
        "category": "methodology",
        "user": (
            "What research methodology does this paper use?\n\n"
            "Title: BERT: Pre-training of Deep Bidirectional Transformers for "
            "Language Understanding\n"
            "Abstract: We introduce a new language representation model called BERT, "
            "which stands for Bidirectional Encoder Representations from Transformers. "
            "BERT is designed to pre-train deep bidirectional representations from "
            "unlabeled text by jointly conditioning on both left and right context. "
            "As a result, the pre-trained BERT model can be fine-tuned with just one "
            "additional output layer to create state-of-the-art models for a wide range "
            "of tasks, such as question answering and language inference."
        ),
        "rubric": {
            "should_mention": ["pre-training", "fine-tuning", "bidirectional",
                               "empirical", "benchmark"],
            "should_not_claim": ["I ran this experiment", "we achieved"],
            "must_caveat": True,
        },
    },
    {
        "id": "q3",
        "category": "uncertainty",
        "user": (
            "What are the exact test accuracy numbers reported in the paper "
            "'ResNet: Deep Residual Learning for Image Recognition'?"
        ),
        "rubric": {
            "should_mention": ["cannot", "don't have", "abstract", "full text",
                               "read the paper", "uncertain"],
            "should_not_claim": ["92%", "96.43%", "the paper reports exactly"],
            "must_caveat": True,
        },
    },
    {
        "id": "q4",
        "category": "oa_status",
        "user": (
            "Is this paper freely available?\n\n"
            "Title: Generative Adversarial Networks\n"
            "OA status: green (preprint available)"
        ),
        "rubric": {
            "should_mention": ["green", "preprint", "freely", "accessible"],
            "should_not_claim": ["definitely published open access", "gold OA"],
            "must_caveat": False,
        },
    },
    {
        "id": "q5",
        "category": "limitations",
        "user": (
            "What limitations should I be aware of when using a CNN for medical "
            "image segmentation based on a paper published in 2023?"
        ),
        "rubric": {
            "should_mention": ["dataset", "generalise", "domain"],
            "should_not_claim": ["I verified", "we tested on your data"],
            "must_caveat": True,
        },
    },
]

# ── ollama API helpers ────────────────────────────────────────────────────────
def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def model_available(model_tag: str) -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        tags = [m["name"] for m in r.json().get("models", [])]
        return any(model_tag in t for t in tags)
    except Exception:
        return False

def generate(model_tag: str, system: str, user: str, max_tokens: int = 512) -> tuple[str, float]:
    payload = {
        "model":  model_tag,
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
        ],
        "options": {"num_predict": max_tokens, "temperature": 0.1, "seed": 42},
        "stream": False,
    }
    t0 = time.time()
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
        elapsed = time.time() - t0
        if r.status_code == 200:
            text = r.json()["message"]["content"]
        else:
            text = f"ERROR: {r.status_code} — {r.text[:200]}"
    except Exception as e:
        elapsed = time.time() - t0
        text = f"ERROR: {e}"
    return text.strip(), round(elapsed, 2)

def score_response(response: str, rubric: dict) -> dict:
    """Simple keyword-based scoring. Not a replacement for human review."""
    resp_lower = response.lower()
    found_pos   = [w for w in rubric.get("should_mention", []) if w.lower() in resp_lower]
    found_neg   = [w for w in rubric.get("should_not_claim", []) if w.lower() in resp_lower]
    # Check for caveat language
    caveat_words = ["note:", "caveat", "based on", "abstract only", "full text",
                    "uncertain", "cannot confirm", "may have", "i don't"]
    has_caveat  = any(w in resp_lower for w in caveat_words)
    pos_score   = len(found_pos) / max(1, len(rubric.get("should_mention", [1])))
    neg_score   = len(found_neg) / max(1, len(rubric.get("should_not_claim", [1])))
    caveat_ok   = (not rubric.get("must_caveat", False)) or has_caveat
    composite   = round(pos_score - neg_score * 0.5 + (0.2 if caveat_ok else 0), 3)
    return {
        "positive_hits":  found_pos,
        "negative_hits":  found_neg,
        "has_caveat":     has_caveat,
        "caveat_required":rubric.get("must_caveat", False),
        "pos_score":      round(pos_score, 3),
        "neg_score":      round(neg_score, 3),
        "composite":      max(0.0, min(1.0, composite)),
    }

# ── main evaluation ───────────────────────────────────────────────────────────
print("ResearchPilot — Model Evaluation")
print("=" * 55)

if not ollama_available():
    print("ERROR: Ollama not running. Start it from the system tray or:")
    print("       Start-Process 'C:\\Users\\LOQ\\AppData\\Local\\Programs\\Ollama\\ollama app.exe'")
    sys.exit(1)
print("Ollama API: reachable")

# Determine which models to evaluate
models_to_eval = []
if model_available(BASE_MODEL):
    models_to_eval.append(("base",  BASE_MODEL))
    print(f"Base model:   {BASE_MODEL} — FOUND")
else:
    print(f"Base model:   {BASE_MODEL} — NOT FOUND")
    print(f"  Pull with:  ollama pull {BASE_MODEL}")

if model_available(TUNED_MODEL):
    models_to_eval.append(("tuned", TUNED_MODEL))
    print(f"Tuned model:  {TUNED_MODEL} — FOUND")
else:
    print(f"Tuned model:  {TUNED_MODEL} — NOT FOUND (skipping comparison)")

if not models_to_eval:
    print("\nERROR: No models available for evaluation.")
    print("       Pull a model first: ollama pull qwen2.5:1.5b-instruct")
    sys.exit(1)

print(f"\nEvaluating {len(models_to_eval)} model(s) on {len(EVAL_QUESTIONS)} questions ...")
print("-" * 55)

all_results = []

for q in EVAL_QUESTIONS:
    print(f"\n  Q{q['id']} [{q['category']}]: {q['user'][:60]}...")
    q_results = {"question_id": q["id"], "category": q["category"],
                 "user": q["user"], "responses": {}}
    for model_label, model_tag in models_to_eval:
        response, latency = generate(model_tag, SYSTEM_PROMPT, q["user"])
        scores = score_response(response, q["rubric"])
        print(f"    {model_label} ({model_tag}): composite={scores['composite']:.2f}  latency={latency}s")
        q_results["responses"][model_label] = {
            "model_tag": model_tag,
            "response":  response,
            "latency_s": latency,
            "scores":    scores,
        }
    all_results.append(q_results)

# ── aggregate scores ──────────────────────────────────────────────────────────
agg: dict[str, list] = {}
for label, _ in models_to_eval:
    scores = [r["responses"][label]["scores"]["composite"]
              for r in all_results if label in r["responses"]]
    agg[label] = scores

# ── save JSON ─────────────────────────────────────────────────────────────────
output_data = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "models_evaluated": [m for _, m in models_to_eval],
    "n_questions": len(EVAL_QUESTIONS),
    "aggregate": {label: {"mean": round(sum(s)/len(s),3),
                           "scores": s} for label, s in agg.items()},
    "results": all_results,
    "note": (
        "Scores are keyword-based heuristics, NOT definitive quality measures. "
        "Human review is required for a valid assessment."
    )
}
with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
    json.dump(output_data, fh, indent=2, ensure_ascii=False)
print(f"\nSaved: {RESULTS_JSON}")

# ── write Markdown report ─────────────────────────────────────────────────────
lines = [
    "# ResearchPilot — Model Evaluation Report",
    f"\n**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
    f"**Models evaluated:** {', '.join(m for _,m in models_to_eval)}",
    f"**Questions:** {len(EVAL_QUESTIONS)}",
    "\n---\n",
    "## Aggregate Scores (keyword heuristic, 0–1)\n",
    "| Model | Mean composite score |",
    "|-------|---------------------|",
]
for label, s_list in agg.items():
    mean = sum(s_list) / len(s_list)
    lines.append(f"| {label} | {mean:.3f} |")

lines += [
    "\n---\n",
    "## Per-question Results\n",
]
for q_res in all_results:
    lines.append(f"### Q{q_res['question_id']} — {q_res['category']}")
    lines.append(f"\n**Question (truncated):** {q_res['user'][:200]}\n")
    for label, resp_data in q_res["responses"].items():
        sc = resp_data["scores"]
        lines += [
            f"**{label}** (`{resp_data['model_tag']}`)  "
            f"composite={sc['composite']:.2f}  latency={resp_data['latency_s']}s\n",
            f"- Positive hits: {sc['positive_hits']}",
            f"- Negative hits: {sc['negative_hits']}",
            f"- Has caveat: {sc['has_caveat']}  (required: {sc['caveat_required']})\n",
            f"**Response:**\n> {resp_data['response'][:500].replace(chr(10), ' ')}\n",
        ]
    lines.append("---\n")

lines += [
    "## Important limitations of this evaluation",
    "",
    "- Scores are based on keyword matching, NOT semantic understanding.",
    "- A response can score well by accident (contains keywords without real understanding).",
    "- Human review of each response is required for a valid comparison.",
    "- Training loss ≠ better research answers. Always test on real research tasks.",
    "- The base model comparison is only valid if both models use identical prompts and settings.",
    "",
    "## Human review rubric",
    "",
    "| Criterion | Score 1-5 | Notes |",
    "|-----------|-----------|-------|",
    "| Instruction following | | |",
    "| Factual accuracy (where verifiable) | | |",
    "| Appropriate uncertainty expression | | |",
    "| No fabricated citations or results | | |",
    "| Clarity and structure | | |",
    "| Research-domain relevance | | |",
]

with open(REPORT_MD, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print(f"Saved: {REPORT_MD}")

print(f"\n{'='*55}")
print(f"  evaluate_model.py  COMPLETE")
print(f"  See: {EVAL_DIR}")
print(f"{'='*55}")
