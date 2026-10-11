"""
ResearchPilot — training/merge_adapter.py

Merges the LoRA adapter into the base model and saves a full merged model.
The merged model can then be converted to GGUF for Ollama import.

Run:
    py -3.12 training/merge_adapter.py

Requirements:
  - Adapter must exist at models/researchpilot-lora/
  - PyTorch + transformers + peft must be installed
  - Enough DISK space: base model (~3 GB) + merged model (~3 GB) = ~6 GB
  - Note: merging is CPU-safe (does not require CUDA)

Output:
    models/exports/researchpilot-merged/
        config.json
        tokenizer files
        model.safetensors (or sharded)
"""

import json
import sys
import shutil
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
ADAPTER_DIR = ROOT / "models" / "researchpilot-lora"
EXPORT_DIR  = ROOT / "models" / "exports" / "researchpilot-merged"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── check adapter exists ──────────────────────────────────────────────────────
adapter_config = ADAPTER_DIR / "adapter_config.json"
if not adapter_config.exists():
    print(f"ERROR: Adapter not found at {ADAPTER_DIR}")
    print("       Run training first: py -3.12 training/train_lora.py")
    sys.exit(1)

with open(adapter_config) as fh:
    acfg = json.load(fh)

base_model_id = acfg.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
print(f"Adapter base model: {base_model_id}")
print(f"Adapter directory:  {ADAPTER_DIR}")
print(f"Output directory:   {EXPORT_DIR}")

# ── imports ───────────────────────────────────────────────────────────────────
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError as e:
    print(f"ERROR: Missing package — {e}")
    print("       Run: .\\training\\install_packages.ps1")
    sys.exit(1)

# ── load base model (cpu to save VRAM, merge is math not inference) ───────────
print(f"\nLoading base model {base_model_id} on CPU ...")
print("(This may take 2-4 minutes — model is ~3 GB)")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    trust_remote_code=True,
)
print(f"  Base model loaded  |  params: {sum(p.numel() for p in base_model.parameters()):,}")

# ── load adapter and merge ────────────────────────────────────────────────────
print(f"\nLoading LoRA adapter from {ADAPTER_DIR} ...")
peft_model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_DIR),
    torch_dtype=torch.float16,
)
print("  Adapter loaded. Merging ...")
merged_model = peft_model.merge_and_unload()
print(f"  Merge complete  |  params: {sum(p.numel() for p in merged_model.parameters()):,}")

# ── verify merged model loads a forward pass ──────────────────────────────────
print("\nVerifying merged model with a short forward pass ...")
try:
    inputs = tokenizer("Test input", return_tensors="pt")
    with torch.no_grad():
        _ = merged_model(**inputs, labels=inputs["input_ids"])
    print("  Forward pass OK")
except Exception as e:
    print(f"  WARN: Forward pass failed: {e}")
    print("        Saving anyway — adapter may still be usable")

# ── save merged model ─────────────────────────────────────────────────────────
print(f"\nSaving merged model to {EXPORT_DIR} ...")
merged_model.save_pretrained(str(EXPORT_DIR), safe_serialization=True)
tokenizer.save_pretrained(str(EXPORT_DIR))

saved_files = list(EXPORT_DIR.iterdir())
total_size_gb = sum(f.stat().st_size for f in saved_files) / 1024**3
print(f"  Saved {len(saved_files)} files  |  total size: {total_size_gb:.2f} GB")
for f in sorted(saved_files):
    print(f"    {f.name}  ({f.stat().st_size // 1024} KB)")

# ── instructions for GGUF conversion ─────────────────────────────────────────
print(f"""
{'='*60}
  NEXT STEP: Convert to GGUF for Ollama

  Option A — llama.cpp convert script (recommended):
  -------------------------------------------------------
  1. Install llama.cpp (Windows pre-built):
     winget install ggerganov.llama.cpp
     OR download from: https://github.com/ggerganov/llama.cpp/releases

  2. Convert merged model to GGUF:
     cd path\\to\\llama.cpp
     python convert_hf_to_gguf.py ^
       {EXPORT_DIR} ^
       --outtype f16 ^
       --outfile models\\exports\\researchpilot.gguf

  3. Quantize to Q4_K_M (optional, reduces size ~50%):
     .\\llama-quantize.exe ^
       models\\exports\\researchpilot.gguf ^
       models\\exports\\researchpilot-q4km.gguf ^
       Q4_K_M

  4. Import into Ollama:
     See ollama\\README.md for exact PowerShell commands.

  Option B — skip fine-tuning, use Qwen base via Ollama directly:
  -------------------------------------------------------
     ollama pull qwen2.5:1.5b-instruct
     See ollama\\Modelfile for ResearchPilot system prompt.
{'='*60}
""")

print("merge_adapter.py  COMPLETE")
