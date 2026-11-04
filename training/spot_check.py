import json
from pathlib import Path

base = Path("e:/Tuned_Research/data/finetuning")
for split in ["train", "validation", "test", "smoke_test"]:
    f = base / f"{split}.jsonl"
    if not f.exists():
        continue
    lines = f.read_text(encoding="utf-8").strip().splitlines()
    ex    = json.loads(lines[0])
    msgs  = ex["messages"]
    meta  = ex["_meta"]
    print(f"--- {split}.jsonl ({len(lines)} examples) ---")
    print(f"  roles:     {[m['role'] for m in msgs]}")
    print(f"  template:  {meta['template']}")
    print(f"  synthetic: {meta['synthetic']}  reviewed: {meta['reviewed']}")
    print(f"  user:      {msgs[1]['content'][:70]}")
    print(f"  assistant: {msgs[2]['content'][:90]}")
    print()

# Check for cross-split leakage (paper IDs must be disjoint)
print("--- Leakage check ---")
splits_ids = {}
for split in ["train", "validation", "test"]:
    f = base / f"{split}.jsonl"
    ids = set()
    for line in f.read_text(encoding="utf-8").strip().splitlines():
        ex = json.loads(line)
        ids.add(ex["_meta"]["source_id"])
    splits_ids[split] = ids
    print(f"  {split}: {len(ids)} unique paper IDs")

tv = splits_ids["train"] & splits_ids["validation"]
tt = splits_ids["train"] & splits_ids["test"]
vt = splits_ids["validation"] & splits_ids["test"]
print(f"  train ∩ val:   {len(tv)}  (must be 0)")
print(f"  train ∩ test:  {len(tt)}  (must be 0)")
print(f"  val   ∩ test:  {len(vt)}  (must be 0)")
if len(tv) == len(tt) == len(vt) == 0:
    print("  PASS: No cross-split leakage")
else:
    print("  FAIL: Leakage detected!")
