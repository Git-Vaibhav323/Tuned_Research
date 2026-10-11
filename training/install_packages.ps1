# ResearchPilot — Windows-native package installer
# Run from project root: .\training\install_packages.ps1
# Uses Python 3.12 (real install, not MS Store stub)
# All packages go to E: to preserve C: space
#
# WHAT THIS INSTALLS:
#   1. PyTorch 2.3.x with CUDA 12.1 (stable, works on RTX 3050 + driver CUDA 13.x)
#   2. Hugging Face stack: transformers, datasets, peft, trl, accelerate
#   3. bitsandbytes 0.43+ (Windows wheel — supports 4-bit on CUDA 12.x)
#   4. Supporting packages: sentencepiece, tokenizers, packaging, scipy, tqdm
#
# RUNTIME: ~15-30 min depending on network speed
# DISK:    ~8-10 GB (use E: drive)

$PY = "C:\Users\LOQ\AppData\Local\Python\pythoncore-3.12-64\python.exe"
$PIP = "$PY -m pip"

Write-Host "=== ResearchPilot Package Installer ===" -ForegroundColor Cyan
Write-Host "Python: $PY"
Write-Host "Verifying Python..."
& $PY --version
if ($LASTEXITCODE -ne 0) { Write-Error "Python not found at $PY"; exit 1 }

# Step 1 — upgrade pip
Write-Host "`n[1/5] Upgrading pip..." -ForegroundColor Yellow
& $PY -m pip install --upgrade pip

# Step 2 — PyTorch with CUDA 12.1
# CUDA 12.1 wheels work fine with driver CUDA 13.x (backward compatible)
# Using --index-url to get CUDA build, not the default CPU-only build
Write-Host "`n[2/5] Installing PyTorch 2.3 + CUDA 12.1..." -ForegroundColor Yellow
Write-Host "      (largest download ~2.5GB, be patient)"
& $PY -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 `
    --index-url https://download.pytorch.org/whl/cu121

# Step 3 — Hugging Face core stack
Write-Host "`n[3/5] Installing Hugging Face stack..." -ForegroundColor Yellow
& $PY -m pip install `
    transformers==4.44.2 `
    datasets==2.21.0 `
    peft==0.12.0 `
    trl==0.9.6 `
    accelerate==0.34.2 `
    tokenizers `
    sentencepiece `
    packaging

# Step 4 — bitsandbytes Windows wheel
# bitsandbytes 0.43+ ships pre-compiled Windows CUDA wheels
Write-Host "`n[4/5] Installing bitsandbytes (Windows CUDA wheel)..." -ForegroundColor Yellow
& $PY -m pip install bitsandbytes==0.43.3

# Step 5 — supporting packages
Write-Host "`n[5/5] Installing supporting packages..." -ForegroundColor Yellow
& $PY -m pip install `
    scipy `
    tqdm `
    pandas `
    numpy `
    python-dotenv `
    rich `
    requests `
    huggingface_hub `
    safetensors `
    einops

Write-Host "`n=== Verifying installation ===" -ForegroundColor Cyan
& $PY -c @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')

import transformers, peft, trl, accelerate
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print(f'trl: {trl.__version__}')
print(f'accelerate: {accelerate.__version__}')

try:
    import bitsandbytes as bnb
    print(f'bitsandbytes: {bnb.__version__}')
    if torch.cuda.is_available():
        l = bnb.nn.Linear4bit(16, 16)
        print('bitsandbytes 4-bit: OK')
except Exception as e:
    print(f'bitsandbytes: {e}')
"@

Write-Host "`n=== Done. Re-run training/check_environment.py to confirm. ===" -ForegroundColor Green
