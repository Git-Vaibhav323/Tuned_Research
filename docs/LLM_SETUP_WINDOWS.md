# ResearchPilot — LLM Setup on Windows (Native, No WSL)

**System:** Windows 11 Home | RTX 3050 Laptop 6 GB VRAM | AMD Ryzen 5 7000  
**Python:** 3.12.10 at `C:\Users\LOQ\AppData\Local\Python\pythoncore-3.12-64\python.exe`  
**Ollama:** 0.40.1 at `C:\Users\LOQ\AppData\Local\Programs\Ollama\ollama.exe`

---

## Part 1 — Immediate Quick Start (Ollama Only, No Training)

This works today without installing any Python ML packages.

### Step 1 — Verify Ollama is running

```powershell
$ollama = "C:\Users\LOQ\AppData\Local\Programs\Ollama\ollama.exe"
& $ollama list
```

Expected: lists `llama3.2:latest` (already downloaded).

### Step 2 — Pull Qwen2.5-1.5B-Instruct

```powershell
& $ollama pull qwen2.5:1.5b-instruct
```

~1.1 GB download. Uses the pre-quantized Q4_K_M GGUF from Ollama registry.

### Step 3 — Create ResearchPilot model with system prompt

```powershell
Set-Location "e:\Tuned_Research"
& $ollama create researchpilot --file ollama\Modelfile
```

### Step 4 — Test

```powershell
& $ollama run researchpilot "What is the difference between LoRA and full fine-tuning?"
```

### Step 5 — Run the evaluation script (Python, no GPU needed)

```powershell
py -3.12 training\evaluate_model.py
```

---

## Part 2 — Install ML Training Packages

Only needed if you want to run `train_lora.py`.

### Prerequisites

- Python 3.12 (confirmed at `C:\Users\LOQ\AppData\Local\Python\pythoncore-3.12-64\python.exe`)
- NVIDIA driver 591.66+ (already installed — confirmed via nvidia-smi)
- ~8–10 GB free disk space on E: drive (for PyTorch + model weights)
- C: drive has only 13 GB free — install packages to default user lib (E: is used for models)

### Run the installer

```powershell
Set-Location "e:\Tuned_Research"
.\training\install_packages.ps1
```

This installs:
- `torch 2.3.1+cu121` — PyTorch with CUDA 12.1 wheels
- `transformers 4.44.2` — Hugging Face model loading
- `datasets 2.21.0` — dataset handling
- `peft 0.12.0` — LoRA adapters
- `trl 0.9.6` — SFTTrainer
- `accelerate 0.34.2` — distributed training utilities
- `bitsandbytes 0.43.3` — 4-bit quantization (Windows wheel)
- Supporting: sentencepiece, tokenizers, safetensors, scipy, tqdm, rich

**Why CUDA 12.1 wheels with driver CUDA 13.1?**  
NVIDIA drivers are backward compatible. Driver CUDA 13.1 supports PyTorch
built for CUDA 12.1. This is the correct and tested combination.

**Why not Python 3.14?**  
PyTorch 2.3.x has no Python 3.14 wheel. Use Python 3.12.

### Verify installation

```powershell
py -3.12 training\check_environment.py
```

All items should show `[PASS]`.

---

## Part 3 — Prepare Fine-Tuning Dataset

```powershell
Set-Location "e:\Tuned_Research"
py -3.12 training\prepare_dataset.py
```

Output files:
- `data/finetuning/train.jsonl`
- `data/finetuning/validation.jsonl`
- `data/finetuning/test.jsonl`
- `data/finetuning/dataset_card.md`
- `data/finetuning/smoke_test.jsonl`

---

## Part 4 — Run Training

### Before training: free GPU VRAM

Close these applications:
- ChatGPT Classic (currently using GPU)
- Microsoft Edge / WebView2 processes
- Any other GPU-accelerated apps

Check free VRAM:
```powershell
nvidia-smi --query-gpu=memory.free --format=csv,noheader
```
Need: ≥5,500 MiB for fp16 LoRA, ≥2,500 MiB for QLoRA.

### Smoke test (5 steps, ~2 min — confirms code runs)

```powershell
py -3.12 training\train_lora.py --smoke-test
```

### Full training (fp16 LoRA — recommended for 6GB VRAM)

```powershell
py -3.12 training\train_lora.py --no-qlora --epochs 3
```

Estimated time: ~45–90 min for 3 epochs on ~8,000 examples.

### Full training (QLoRA 4-bit — if bitsandbytes works)

```powershell
py -3.12 training\train_lora.py --epochs 3
```

Estimated time: ~60–120 min (slower per step but more memory efficient).

### If you get CUDA Out of Memory

1. Close GPU apps (ChatGPT, browser)
2. Try smoke-test first: `--smoke-test`
3. Force fp16: `--no-qlora`
4. Reduce sequence length: edit `MAX_SEQ_LEN = 192` in `train_lora.py`

---

## Part 5 — Merge Adapter and Import to Ollama

### Merge LoRA adapter into base model

```powershell
py -3.12 training\merge_adapter.py
```

Output: `models/exports/researchpilot-merged/`

### Convert to GGUF (requires llama.cpp)

Install llama.cpp:
```powershell
winget install ggerganov.llama.cpp
# OR download from: https://github.com/ggerganov/llama.cpp/releases
```

Convert:
```powershell
py -3.12 "C:\llama.cpp\convert_hf_to_gguf.py" `
    models\exports\researchpilot-merged `
    --outtype f16 `
    --outfile models\exports\researchpilot-f16.gguf

"C:\llama.cpp\llama-quantize.exe" `
    models\exports\researchpilot-f16.gguf `
    ollama\researchpilot-q4km.gguf `
    Q4_K_M
```

### Import into Ollama

Edit `ollama/Modelfile`: uncomment the `FROM ./researchpilot-q4km.gguf` line.

```powershell
Set-Location "e:\Tuned_Research"
$ollama = "C:\Users\LOQ\AppData\Local\Programs\Ollama\ollama.exe"
& $ollama rm researchpilot
& $ollama create researchpilot --file ollama\Modelfile
& $ollama run researchpilot "Summarise this paper: [paste title and abstract]"
```

---

## Part 6 — RAG Integration (Next Phase)

RAG allows the model to retrieve and cite actual papers from the corpus,
grounding its answers in the 2,000 OpenAlex papers.

Required packages (install after training is working):
```powershell
py -3.12 -m pip install sentence-transformers chromadb langchain
```

Architecture:
```
User question
    → embed with sentence-transformers/all-MiniLM-L6-v2
    → query ChromaDB vector store (2,000 papers indexed)
    → retrieve top-k relevant abstracts
    → build context prompt
    → send to researchpilot Ollama model
    → return grounded answer with source paper IDs
```

See `rag/` directory for implementation (Phase 3).

---

## Package versions (target)

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.12.10 | Use py launcher: `py -3.12` |
| torch | 2.3.1+cu121 | CUDA 12.1 build |
| transformers | 4.44.2 | |
| datasets | 2.21.0 | |
| peft | 0.12.0 | |
| trl | 0.9.6 | SFTTrainer API |
| accelerate | 0.34.2 | |
| bitsandbytes | 0.43.3 | Windows CUDA wheel |
| ollama | 0.40.1 | Already installed |

---

## Known limitations on Windows

| Issue | Status | Workaround |
|-------|--------|-----------|
| Unsloth (fast LoRA) | Blocked — needs Linux Triton | Use standard PEFT LoRA |
| Flash Attention 2 | Blocked — needs nvcc compiler | Disabled in train_lora.py |
| bitsandbytes 4-bit | May fail on some CUDA combos | Auto-fallback to fp16 LoRA |
| DeepSpeed | Not tested on Windows | Not used |
| NCCL multi-GPU | N/A (laptop, 1 GPU) | N/A |
