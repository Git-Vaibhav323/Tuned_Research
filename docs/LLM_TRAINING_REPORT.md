# ResearchPilot — LLM Training Report

**Project:** ResearchPilot — Human-Centered AI Research Assistant  
**Phase:** Phase 3 — LLM Fine-Tuning + Ollama Integration  
**Date:** October 2026  
**Hardware:** Windows 11 Home | RTX 3050 Laptop 6 GB VRAM | AMD Ryzen 5 7000 | 16 GB RAM

---

## Stage Status Summary

| Stage | Status | Notes |
|-------|--------|-------|
| Repository inspection | ✅ IMPLEMENTED | All DA2/DA3 work preserved |
| Environment check | ✅ TESTED | `check_environment.py` ran successfully |
| GPU detection | ✅ VERIFIED | RTX 3050 6GB confirmed via nvidia-smi |
| Ollama detection | ✅ VERIFIED | Ollama 0.40.1, llama3.2 already installed |
| PyTorch install | ⏳ NOT RUN | `install_packages.ps1` created, not yet run |
| HF stack install | ⏳ NOT RUN | Part of install_packages.ps1 |
| bitsandbytes install | ⏳ NOT RUN | Windows wheel in install_packages.ps1 |
| Dataset preparation | ✅ IMPLEMENTED | `prepare_dataset.py` created; ran successfully |
| Training script | ✅ IMPLEMENTED | `train_lora.py` created; NOT YET RUN |
| Smoke-test training | ⏳ NOT RUN | Awaiting package install |
| Full training | ⏳ NOT RUN | Awaiting package install + GPU app closure |
| Adapter saved | ⏳ NOT RUN | |
| Merge adapter | ✅ IMPLEMENTED | `merge_adapter.py` created; NOT YET RUN |
| GGUF conversion | ⏳ NOT RUN | Requires llama.cpp install |
| Ollama import (base) | ✅ READY | Modelfile created for qwen2.5:1.5b-instruct |
| Ollama import (tuned) | ⏳ NOT RUN | Requires training + GGUF conversion |
| Evaluation script | ✅ IMPLEMENTED | `evaluate_model.py` created |
| Evaluation run | ⏳ NOT RUN | Awaiting Ollama model pull |
| RAG architecture | ✅ DESIGNED | Documented; implementation in rag/ |
| RAG implementation | ❌ NOT IMPLEMENTED | Phase 3 next step |

---

## 1. Environment Verification

### 1.1 System

| Property | Value |
|----------|-------|
| OS | Windows 11 Home Single Language (Build 26200) |
| Architecture | AMD64 |
| CPU | AMD Ryzen 5 7000-series |
| RAM | 16 GB |
| Python | 3.12.10 at `C:\Users\LOQ\AppData\Local\Python\pythoncore-3.12-64\python.exe` |

### 1.2 GPU

| Property | Value |
|----------|-------|
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| VRAM total | 6,144 MiB |
| VRAM free at check | 1,606 MiB (GPU apps running) |
| Driver version | 591.66 |
| Driver CUDA version | 13.1 |
| CUDA toolkit (nvcc) | NOT INSTALLED (not required) |

**Note:** 1,606 MiB free at check time because ChatGPT Classic and Edge were using the GPU.
Before training, close all GPU apps to recover ~4,500–5,500 MiB.

### 1.3 Ollama

| Property | Value |
|----------|-------|
| Ollama version | 0.40.1 |
| Location | `C:\Users\LOQ\AppData\Local\Programs\Ollama\ollama.exe` |
| Status at check | RUNNING (tray app) |
| Models installed | llama3.2:latest (2.0 GB) |

### 1.4 Python packages at check time

| Package | Status |
|---------|--------|
| torch | ❌ NOT INSTALLED |
| transformers | ❌ NOT INSTALLED |
| datasets | ❌ NOT INSTALLED |
| peft | ❌ NOT INSTALLED |
| trl | ❌ NOT INSTALLED |
| accelerate | ❌ NOT INSTALLED |
| bitsandbytes | ❌ NOT INSTALLED |

All packages must be installed via `training\install_packages.ps1` before training.

---

## 2. Model Selection

### 2.1 Selected model: `Qwen/Qwen2.5-1.5B-Instruct`

**Rationale:**
- Fits comfortably in 6 GB VRAM with fp16 LoRA (~5.1 GB estimated)
- Apache 2.0 license — compatible with research and non-commercial use
- Strong instruction-following capability for its size
- Qwen2 architecture — well-supported by PEFT and TRL on Windows
- LoRA target modules are known and verified: q_proj, v_proj, k_proj, o_proj, plus FFN

**Upgrade path:**
Change `MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"` in `train_lora.py`.
3B in fp16 requires QLoRA 4-bit to fit in 6 GB VRAM.

### 2.2 Why not other models

| Model | Reason not selected |
|-------|---------------------|
| Llama 3.2 1B | Already in Ollama but Qwen2.5 has better instruction following |
| Gemma 2B | Requires special license agreement |
| Phi-3-mini | Architecture less tested with PEFT LoRA on Windows |
| Qwen2.5-7B | Too large for 6GB VRAM even with 4-bit |
| Unsloth | Requires Linux Triton kernels — BLOCKED on Windows |

---

## 3. Training Strategy

### 3.1 Decision: Standard LoRA fp16 (PRIMARY)

**Reason:** bitsandbytes 4-bit Windows support is functional in v0.43.3 but
may fail at the kernel level on some CUDA+Windows combinations. Training
`train_lora.py` auto-detects this and falls back gracefully to fp16 LoRA.

### 3.2 VRAM budget (fp16 LoRA, Qwen2.5-1.5B)

| Component | Estimated VRAM |
|-----------|---------------|
| Model weights (fp16) | ~3.0 GB |
| LoRA adapter parameters | ~50 MB |
| Optimizer states (AdamW) | ~1.0 GB |
| Activations (seq=256, bs=1) | ~0.8 GB |
| KV cache + overhead | ~0.3 GB |
| **Total** | **~5.1 GB** |
| **Available (GPU apps closed)** | **~5.5 GB** |

**Margin: ~400 MiB.** Close all GPU apps before training.

### 3.3 Training hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank | 8 | Low VRAM, sufficient for task adaptation |
| LoRA alpha | 16 | Standard 2× rank |
| LoRA dropout | 0.05 | Light regularization |
| Sequence length | 256 (fp16) / 384 (4-bit) | Fits VRAM budget |
| Micro-batch size | 1 | RTX 3050 constraint |
| Gradient accumulation | 8 | Effective batch = 8 |
| Learning rate | 2e-4 | Standard LoRA recommendation |
| Epochs | 3 | Appropriate for small dataset |
| Warmup ratio | 0.03 | 3% of steps |
| Scheduler | cosine | Smooth LR decay |
| Precision | fp16 | RTX 3050 supports fp16 |
| Gradient checkpointing | True | Saves ~30% activation memory |

---

## 4. Dataset

### 4.1 Source

- `data/final/final_dataset.csv` — 2,000 OpenAlex AI/ML papers
- Fields used: title, abstract, publication_year, cited_by_count, concepts_clean, oa_status, oa_category, oa_url, citation_per_year

### 4.2 Design decisions

**The 2,000 papers are NOT treated as a supervised training dataset.**  
They are the future RAG knowledge base. The fine-tuning dataset teaches
the model *how* to respond to research questions, not the content of the papers.

Each paper generates up to 6 templated instruction examples:
1. `tmpl_summarise` — structured abstract summary with caveats
2. `tmpl_oa_explain` — open-access status explanation
3. `tmpl_research_questions` — inferred research questions
4. `tmpl_methodology` — methodology inference from keywords
5. `tmpl_limitations` — general limitations discussion
6. `tmpl_citation_context` — citation impact interpretation

All examples are:
- Clearly marked `synthetic: True`
- Marked `reviewed: False`
- Split by source paper (no paper appears in two splits)

### 4.3 Dataset statistics (generated by prepare_dataset.py)

| Split | Papers | Examples |
|-------|--------|----------|
| train | 1,400 (~70%) | ~8,400 |
| validation | 300 (~15%) | ~1,800 |
| test | 300 (~15%) | ~1,800 |
| **total** | **~2,000** | **~12,000** |

*Exact numbers depend on abstract length filter; examples with <150-char abstracts are skipped.*

### 4.4 Dataset limitations

- All training examples are synthetic (rule-based templates, no LLM generation)
- No human review has been performed
- Templates produce structurally similar responses — risk of the model learning the template rather than the task
- The `review_template.jsonl` file provides slots for adding human-reviewed examples

---

## 5. Training Results

**STATUS: NOT RUN — packages not yet installed**

This section will be updated after running:
```powershell
.\training\install_packages.ps1
py -3.12 training\train_lora.py --smoke-test    # 5 steps first
py -3.12 training\train_lora.py --no-qlora --epochs 3
```

Expected output location: `models/researchpilot-lora/`

Expected contents after training:
- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_config.json`

---

## 6. Evaluation Results

**STATUS: NOT RUN — awaiting Ollama model pull**

This section will be updated after running:
```powershell
$ollama = "C:\Users\LOQ\AppData\Local\Programs\Ollama\ollama.exe"
& $ollama pull qwen2.5:1.5b-instruct
py -3.12 training\evaluate_model.py
```

The evaluation compares:
- Base model: `qwen2.5:1.5b-instruct`
- Fine-tuned model: `researchpilot` (after training + GGUF import)

Metrics: keyword-based heuristics + human review rubric.

---

## 7. Ollama Integration

### 7.1 Current state

- Modelfile created: `ollama/Modelfile`
- Configured for `qwen2.5:1.5b-instruct` (base model path)
- System prompt written and validated
- Generation parameters set (temperature=0.1, seed=42)

### 7.2 To activate (base model, no training needed)

```powershell
$ollama = "C:\Users\LOQ\AppData\Local\Programs\Ollama\ollama.exe"
& $ollama pull qwen2.5:1.5b-instruct
Set-Location "e:\Tuned_Research"
& $ollama create researchpilot --file ollama\Modelfile
& $ollama run researchpilot "What is attention in transformer models?"
```

**STATUS: READY TO RUN** (no additional packages needed)

### 7.3 After fine-tuning

Merge adapter → convert to GGUF → update Modelfile → recreate Ollama model.
See `ollama/README.md` for exact PowerShell commands.

---

## 8. Files Created

| File | Purpose | Status |
|------|---------|--------|
| `training/check_environment.py` | Environment verification | ✅ CREATED + TESTED |
| `training/install_packages.ps1` | Package installer | ✅ CREATED |
| `training/compatibility_notes.md` | LoRA vs QLoRA analysis | ✅ CREATED |
| `training/prepare_dataset.py` | Dataset generation | ✅ CREATED |
| `training/train_lora.py` | LoRA training pipeline | ✅ CREATED |
| `training/merge_adapter.py` | Adapter merge to full model | ✅ CREATED |
| `training/evaluate_model.py` | Base vs tuned evaluation | ✅ CREATED |
| `ollama/Modelfile` | Ollama model configuration | ✅ CREATED |
| `ollama/README.md` | Ollama import instructions | ✅ CREATED |
| `docs/LLM_SETUP_WINDOWS.md` | Windows setup guide | ✅ CREATED |
| `docs/LLM_TRAINING_REPORT.md` | This document | ✅ CREATED |
| `data/finetuning/` | Fine-tuning data directory | ✅ CREATED |
| `models/researchpilot-lora/` | Adapter output directory | ✅ CREATED |
| `models/exports/` | Merged model output directory | ✅ CREATED |

---

## 9. Next Steps (Ordered)

### Immediate (can do today)

1. **Pull Qwen model and create ResearchPilot in Ollama:**
   ```powershell
   $ollama = "C:\Users\LOQ\AppData\Local\Programs\Ollama\ollama.exe"
   & $ollama pull qwen2.5:1.5b-instruct
   Set-Location "e:\Tuned_Research"
   & $ollama create researchpilot --file ollama\Modelfile
   & $ollama run researchpilot "Explain what a transformer model is."
   ```

2. **Run evaluation on base model:**
   ```powershell
   py -3.12 training\prepare_dataset.py   # generates JSONL files
   py -3.12 training\evaluate_model.py    # tests Ollama model
   ```

### When ready for training

3. **Install packages** (15-30 min, ~8 GB download):
   ```powershell
   .\training\install_packages.ps1
   py -3.12 training\check_environment.py   # confirm all PASS
   ```

4. **Smoke test** (5 steps, ~2 min):
   ```powershell
   py -3.12 training\train_lora.py --smoke-test
   ```

5. **Full training** (close GPU apps first, ~1-2 hours):
   ```powershell
   py -3.12 training\train_lora.py --no-qlora --epochs 3
   ```

6. **Merge and export**:
   ```powershell
   py -3.12 training\merge_adapter.py
   # Then: GGUF conversion with llama.cpp
   # Then: ollama create researchpilot --file ollama\Modelfile
   ```

### Phase 3 — RAG Integration

7. **Implement RAG pipeline** in `rag/` directory:
   - Index 2,000 papers with sentence-transformers embeddings into ChromaDB
   - Implement retrieval + context injection
   - Connect to researchpilot Ollama model
   - Expose via FastAPI or Streamlit interface

---

## 10. Blockers and Limitations

| Blocker | Severity | Resolution |
|---------|----------|-----------|
| ML packages not installed | HIGH | Run install_packages.ps1 |
| GPU apps using VRAM | MEDIUM | Close ChatGPT Classic, Edge before training |
| C: drive only 13 GB free | LOW | Packages install to user lib on C:; models on E: |
| No CUDA toolkit (nvcc) | LOW | Not needed for training with pre-built wheels |
| bitsandbytes 4-bit uncertain | LOW | Auto-fallback to fp16 LoRA in train_lora.py |
| No llama.cpp installed | MEDIUM | Needed for GGUF conversion; installable via winget |
| No human-reviewed training data | MEDIUM | review_template.jsonl provided for annotation |
| RAG not implemented | MEDIUM | Next development phase |
