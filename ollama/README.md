# ResearchPilot Ollama Integration

## Quick start (no training required)

This uses the Qwen2.5 base model with the ResearchPilot system prompt.
Run these commands in PowerShell from the project root.

### Step 1 — Pull the base model

```powershell
$ollama = "C:\Users\LOQ\AppData\Local\Programs\Ollama\ollama.exe"
& $ollama pull qwen2.5:1.5b-instruct
```

Expected download: ~1.1 GB (Q4_K_M quantized GGUF)

### Step 2 — Create the ResearchPilot model

```powershell
Set-Location "e:\Tuned_Research"
$ollama = "C:\Users\LOQ\AppData\Local\Programs\Ollama\ollama.exe"
& $ollama create researchpilot --file ollama\Modelfile
```

Expected output:
```
transferring model data
creating model layer
using existing layer ...
writing manifest
success
```

### Step 3 — Test it

```powershell
& $ollama run researchpilot "Summarise this paper: Title: Attention Is All You Need. Abstract: We propose the Transformer architecture based solely on attention mechanisms."
```

### Step 4 — Verify it is listed

```powershell
& $ollama list
```

---

## Using via Ollama API (Python)

The ResearchPilot model exposes a chat API at `http://127.0.0.1:11434`.

```python
import requests

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL      = "researchpilot"

def ask(question: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=120)
    return r.json()["message"]["content"]

# Example
print(ask("What is RAG and why is it useful for research assistants?"))
```

---

## After fine-tuning: import the trained GGUF model

Once you have run `training/merge_adapter.py` and converted to GGUF:

### Step 1 — Edit Modelfile to use the fine-tuned model

Open `ollama/Modelfile` and change:
```
FROM qwen2.5:1.5b-instruct
```
to:
```
FROM ./researchpilot-q4km.gguf
```

### Step 2 — Copy the GGUF file to the ollama directory

```powershell
Copy-Item "models\exports\researchpilot-q4km.gguf" "ollama\"
```

### Step 3 — Recreate the model

```powershell
Set-Location "e:\Tuned_Research"
$ollama = "C:\Users\LOQ\AppData\Local\Programs\Ollama\ollama.exe"
& $ollama rm researchpilot          # remove old version
& $ollama create researchpilot --file ollama\Modelfile
& $ollama run researchpilot "Hello, what can you help me with?"
```

---

## GGUF conversion (llama.cpp)

Install llama.cpp on Windows:

```powershell
# Option 1: winget (if available)
winget install ggerganov.llama.cpp

# Option 2: download pre-built from GitHub releases
# https://github.com/ggerganov/llama.cpp/releases
# Download: llama-bXXXX-bin-win-cuda-cu12.1-x64.zip
# Extract to: C:\llama.cpp\
```

Convert the merged model:

```powershell
$llamacpp = "C:\llama.cpp"
$mergedDir = "e:\Tuned_Research\models\exports\researchpilot-merged"
$outGguf   = "e:\Tuned_Research\models\exports\researchpilot-f16.gguf"
$outQ4     = "e:\Tuned_Research\ollama\researchpilot-q4km.gguf"

# Convert to GGUF fp16
py -3.12 "$llamacpp\convert_hf_to_gguf.py" `
    $mergedDir `
    --outtype f16 `
    --outfile $outGguf

# Quantize to Q4_K_M (~2x smaller, minimal quality loss)
& "$llamacpp\llama-quantize.exe" $outGguf $outQ4 Q4_K_M
```

Expected GGUF sizes:
- Qwen2.5-1.5B fp16: ~3.0 GB
- Qwen2.5-1.5B Q4_K_M: ~1.0 GB

---

## Troubleshooting

**"failed to get console mode for stderr"**
This is a harmless Windows terminal warning from Ollama. Ignore it.

**"error loading model: unable to load model"**
The GGUF file may be corrupt or from an incompatible llama.cpp version.
Re-convert with a fresh llama.cpp build matching your Ollama version.

**Model responds slowly**
Ollama uses CPU offloading when VRAM is full. Close other GPU applications
to free VRAM for faster inference.

**"model not found"**
Run `ollama list` to see installed models.
Run `ollama pull qwen2.5:1.5b-instruct` to download the base model.
