# Windows Training Compatibility — ResearchPilot
## Hardware: RTX 3050 Laptop 6 GB VRAM | Driver CUDA 13.1 | Windows 11

---

## Decision: Standard LoRA in fp16 (PRIMARY), QLoRA 4-bit (SECONDARY if bitsandbytes works)

### Why not QLoRA-first on Windows?

bitsandbytes 0.43+ ships official Windows CUDA wheels. QLoRA **should** work.
However the 4-bit Linear layer depends on compiled CUDA kernels. On Windows:
- Driver CUDA 13.1 is compatible with PyTorch CUDA 12.1 wheels (backward compat)
- bitsandbytes Windows wheels are pre-compiled; no nvcc needed at runtime
- **BUT**: if bitsandbytes CUDA kernels fail at runtime, training silently falls
  back to slow CPU implementation — which is unusable

The `check_environment.py` script tests this at startup and aborts clearly.

### LoRA in fp16 — confirmed feasible

Model: `Qwen/Qwen2.5-1.5B-Instruct`

| Config | Value | Rationale |
|--------|-------|-----------|
| Precision | fp16 | RTX 3050 supports fp16; no nvcc needed |
| LoRA rank | 8 | Low memory, sufficient for task adaptation |
| LoRA alpha | 16 | Standard 2× rank |
| LoRA dropout | 0.05 | Light regularization |
| Target modules | q_proj, v_proj | Qwen2 attention projections |
| Sequence length | 256–384 tokens | Fits with micro-batch=1 in fp16 |
| Micro-batch size | 1 | RTX 3050 6GB constraint |
| Gradient accum | 8 | Effective batch = 8 |
| Learning rate | 2e-4 | Standard LoRA LR |
| Epochs | 3 | Small dataset |
| Gradient checkpointing | True | Saves ~30% activation memory |

### VRAM estimate for 1.5B fp16 + LoRA

| Component | Approx VRAM |
|-----------|-------------|
| Model weights (fp16) | ~3.0 GB |
| LoRA adapter | ~50 MB |
| Optimizer (AdamW) | ~1.0 GB |
| Activations (seq=256, bs=1) | ~0.8 GB |
| KV cache | ~0.3 GB |
| **Total** | **~5.1 GB** |
| **RTX 3050 available (GPU apps closed)** | **~5.5 GB** |

**Verdict: FITS if GPU apps (ChatGPT, browser) are closed before training.**

### QLoRA 4-bit fallback

If bitsandbytes 4-bit kernels work on this Windows+CUDA combination:

| Component | Approx VRAM |
|-----------|-------------|
| Model weights (4-bit NF4) | ~1.1 GB |
| LoRA adapter | ~50 MB |
| Optimizer (paged AdamW) | ~0.5 GB |
| Activations (seq=384, bs=1) | ~0.6 GB |
| **Total** | **~2.2 GB** |

Enables longer sequences and keeps GPU apps running during training.

### Why NOT Unsloth

Unsloth 2024+ requires Linux or WSL for its Triton-based CUDA kernels.
The Windows wheel does not exist for Triton. Do not install.

### Why Python 3.12, not 3.14

PyTorch 2.3.x wheels are available for Python 3.8–3.12.
Python 3.14 has NO PyTorch wheel yet (too new). Use Python 3.12.

### Confirmed working stack on Windows (from public reports Oct 2025+)

- `torch==2.3.1+cu121`
- `transformers==4.44.x`
- `peft==0.12.x`
- `trl==0.9.x`
- `accelerate==0.34.x`
- `bitsandbytes==0.43.3` (Windows CUDA wheel)

### Selected model

**`Qwen/Qwen2.5-1.5B-Instruct`** (HuggingFace Hub)
- License: Apache 2.0 — compatible with research and non-commercial use
- Architecture: Qwen2 (GQA, RoPE, SwiGLU)
- Tokenizer: tiktoken-style BPE, chat template via `apply_chat_template()`
- LoRA target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Model size on disk: ~3.1 GB (fp16 safetensors)
- Hub ID: `Qwen/Qwen2.5-1.5B-Instruct`
- Revision: `main` (pin to specific commit for reproducibility)

To upgrade to 3B later, change `MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"`.
3B in fp16 requires ~6.0 GB VRAM — only feasible with QLoRA 4-bit on this GPU.

---

## Training approach decision tree

```
Check bitsandbytes 4-bit at startup
      |
      +-- 4-bit works? --> QLoRA (NF4, rank=8, seq=384, paged AdamW)
      |
      +-- 4-bit fails? --> Standard LoRA (fp16, rank=8, seq=256, AdamW)
```

`train_lora.py` implements both paths and selects automatically.
