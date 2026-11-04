"""
ResearchPilot — training/check_environment.py
Checks the Windows-native environment for LLM training readiness.
Run: py -3.12 training/check_environment.py
"""

import sys
import os
import platform
import shutil
import subprocess
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── helpers ──────────────────────────────────────────────────────────────────
def ok(msg):  print(f"  [PASS] {msg}")
def warn(msg): print(f"  [WARN] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")

def check_pkg(name, min_ver=None):
    try:
        mod = importlib.import_module(name.replace("-", "_"))
        ver = getattr(mod, "__version__", "unknown")
        if min_ver and ver != "unknown":
            from packaging.version import Version
            if Version(ver) < Version(min_ver):
                warn(f"{name} {ver} (want >= {min_ver})")
                return ver
        ok(f"{name} {ver}")
        return ver
    except ImportError:
        fail(f"{name} NOT installed")
        return None

def run_cmd(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           shell=isinstance(cmd, str))
        return (r.stdout + r.stderr).strip()
    except Exception as e:
        return f"ERROR: {e}"

# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  ResearchPilot — Environment Check")
print("=" * 60)

# ── 1. OS ────────────────────────────────────────────────────────────────────
print("\n[1] Operating System")
plat = platform.platform()
if "Windows" in plat:
    ok(f"{plat}")
else:
    warn(f"{plat} (expected Windows)")
print(f"     Machine: {platform.machine()} | Processor: {platform.processor()[:60]}")

# ── 2. Python ────────────────────────────────────────────────────────────────
print("\n[2] Python")
ver = sys.version_info
if ver >= (3, 10):
    ok(f"Python {ver.major}.{ver.minor}.{ver.micro}  at {sys.executable}")
else:
    fail(f"Python {ver.major}.{ver.minor}.{ver.micro} — need >= 3.10")
# Warn if this is the MS Store stub
if "WindowsApps" in sys.executable and "PythonSoftwareFoundation" in sys.executable:
    warn("This is the Microsoft Store Python stub — use 'py -3.12' instead")

# ── 3. CUDA via nvidia-smi ────────────────────────────────────────────────────
print("\n[3] NVIDIA GPU (nvidia-smi)")
smi_out = run_cmd(["nvidia-smi",
                   "--query-gpu=name,memory.total,memory.free,driver_version",
                   "--format=csv,noheader"])
if "NVIDIA" in smi_out and "ERROR" not in smi_out:
    ok(f"GPU detected: {smi_out.strip()}")
    # Parse free VRAM
    try:
        parts = smi_out.strip().split(",")
        total_mib = int(parts[1].strip().split()[0])
        free_mib  = int(parts[2].strip().split()[0])
        print(f"     VRAM total: {total_mib} MiB  |  free: {free_mib} MiB")
        if free_mib < 1500:
            warn(f"Only {free_mib} MiB free — close other GPU apps before training")
        elif free_mib < 3000:
            warn(f"{free_mib} MiB free — tight for training; use micro-batch=1")
        else:
            ok(f"{free_mib} MiB free — sufficient for 1.5B LoRA training")
    except Exception:
        warn("Could not parse VRAM numbers from nvidia-smi output")
    # CUDA version from driver
    driver_out = run_cmd("nvidia-smi")
    for line in driver_out.splitlines():
        if "CUDA Version" in line:
            ok(f"Driver CUDA: {line.strip()}")
else:
    fail(f"nvidia-smi failed: {smi_out[:120]}")

# ── 4. PyTorch ────────────────────────────────────────────────────────────────
print("\n[4] PyTorch + CUDA")
torch_ver = check_pkg("torch")
if torch_ver:
    import torch
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        ok(f"CUDA available | {dev} | {vram:.1f} GB VRAM")
        ok(f"CUDA version (PyTorch): {torch.version.cuda}")
    else:
        fail("PyTorch installed but CUDA NOT available")
        warn("Need PyTorch with CUDA — see docs/LLM_SETUP_WINDOWS.md for install cmd")
else:
    fail("PyTorch not installed — run install_packages.ps1 first")

# ── 5. Training packages ──────────────────────────────────────────────────────
print("\n[5] Hugging Face / training packages")
check_pkg("transformers", "4.40.0")
check_pkg("datasets",     "2.18.0")
check_pkg("peft",         "0.10.0")
check_pkg("trl",          "0.8.0")
check_pkg("accelerate",   "0.28.0")
check_pkg("bitsandbytes", "0.43.0")
check_pkg("packaging")
check_pkg("sentencepiece")
check_pkg("tokenizers")

# ── 6. bitsandbytes Windows status ───────────────────────────────────────────
print("\n[6] bitsandbytes Windows support")
try:
    import bitsandbytes as bnb
    ver = bnb.__version__
    # Test 4-bit capability (only available with CUDA)
    try:
        import torch
        if torch.cuda.is_available():
            _ = bnb.nn.Linear4bit(16, 16)
            ok(f"bitsandbytes {ver} — 4-bit Linear layer created successfully")
        else:
            warn(f"bitsandbytes {ver} — CUDA unavailable, 4-bit untestable")
    except Exception as e:
        warn(f"bitsandbytes {ver} — 4-bit layer test failed: {e}")
except ImportError:
    fail("bitsandbytes not installed")
    warn("QLoRA requires bitsandbytes; standard LoRA works without it")

# ── 7. Ollama ─────────────────────────────────────────────────────────────────
print("\n[7] Ollama")
ollama_paths = [
    Path(os.environ.get("LOCALAPPDATA","")) / "Programs" / "Ollama" / "ollama.exe",
    Path("C:/Program Files/Ollama/ollama.exe"),
]
ollama_exe = next((p for p in ollama_paths if p.exists()), None)
if ollama_exe:
    ok(f"Ollama found: {ollama_exe}")
    ver_out = run_cmd([str(ollama_exe), "--version"])
    ok(f"Ollama version: {ver_out.strip()}")
    models_out = run_cmd([str(ollama_exe), "list"])
    print(f"     Installed models:\n{models_out}")
else:
    fail("Ollama not found — download from https://ollama.com/download/windows")

# ── 8. Disk space ─────────────────────────────────────────────────────────────
print("\n[8] Disk space")
for drive in ["E:\\", "C:\\"]:
    usage = shutil.disk_usage(drive)
    free_gb = usage.free / 1024**3
    total_gb = usage.total / 1024**3
    msg = f"{drive}  {free_gb:.1f} GB free / {total_gb:.0f} GB total"
    if free_gb < 5:
        fail(f"{msg}  — VERY LOW, cannot download model")
    elif free_gb < 15:
        warn(f"{msg}  — low, use E: for model downloads")
    else:
        ok(msg)

# ── 9. Project data files ─────────────────────────────────────────────────────
print("\n[9] ResearchPilot data files")
important = {
    "final_dataset.csv":         ROOT / "data" / "final" / "final_dataset.csv",
    "researchpilot_r.db":        ROOT / "database" / "researchpilot_r.db",
    "train.jsonl":               ROOT / "data" / "finetuning" / "train.jsonl",
    "validation.jsonl":          ROOT / "data" / "finetuning" / "validation.jsonl",
    "test.jsonl":                ROOT / "data" / "finetuning" / "test.jsonl",
}
for name, path in important.items():
    if path.exists():
        size_kb = path.stat().st_size // 1024
        ok(f"{name}  ({size_kb} KB)")
    else:
        status = warn if "jsonl" in name else fail
        status(f"{name}  NOT FOUND at {path}")

# ── 10. CUDA toolkit (nvcc) ───────────────────────────────────────────────────
print("\n[10] CUDA toolkit (nvcc)")
nvcc = run_cmd("nvcc --version")
if "release" in nvcc.lower():
    ok(f"nvcc: {nvcc.splitlines()[0]}")
else:
    warn("nvcc not found — CUDA toolkit not installed")
    warn("Training with PyTorch+CUDA works WITHOUT nvcc (driver CUDA is enough)")
    warn("Only needed if compiling CUDA extensions (e.g. flash-attention)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING READINESS SUMMARY")
print("=" * 60)

try:
    import torch
    cuda_ok = torch.cuda.is_available()
except ImportError:
    cuda_ok = False

try:
    import transformers, peft, trl, accelerate
    hf_ok = True
except ImportError:
    hf_ok = False

try:
    import bitsandbytes
    bnb_ok = True
except ImportError:
    bnb_ok = False

print(f"  PyTorch+CUDA:       {'READY' if cuda_ok else 'NOT READY — install first'}")
print(f"  HuggingFace stack:  {'READY' if hf_ok  else 'NOT READY — install first'}")
print(f"  bitsandbytes:       {'READY' if bnb_ok else 'NOT READY — QLoRA unavailable without it'}")
print(f"  Ollama:             {'READY' if ollama_exe else 'NOT READY'}")

if not cuda_ok or not hf_ok:
    print("\n  NEXT STEP: Run install_packages.ps1 to install dependencies.")
    print("  See docs/LLM_SETUP_WINDOWS.md for exact commands.")
elif not bnb_ok:
    print("\n  RECOMMENDATION: Standard LoRA (fp16, no 4-bit) on RTX 3050 6GB.")
    print("  Qwen2.5-1.5B-Instruct fits in fp16 for training.")
else:
    print("\n  READY FOR TRAINING.")
    print("  Recommended: QLoRA 4-bit NF4 on Qwen2.5-1.5B-Instruct.")

print("=" * 60)
