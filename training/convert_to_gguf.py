"""
ResearchPilot — convert merged model to GGUF using llama.cpp source
Downloads only the files needed for conversion from the correct commit.
Run: py -3.12 training/convert_to_gguf.py
"""
import os
import sys
import subprocess
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
MERGED    = ROOT / "models" / "exports" / "researchpilot-merged"
OUT_F16   = ROOT / "models" / "exports" / "researchpilot-f16.gguf"
OUT_Q4    = ROOT / "ollama" / "researchpilot-q4km.gguf"
LLAMA_DIR = ROOT / "llama-b11429-bin-win-cuda-12.4-x64"
CONV_DIR  = ROOT / "llama_convert"
CONV_DIR.mkdir(exist_ok=True)

COMMIT = "d812350"  # llama.cpp v0.6.0 commit

print("Step 1: Installing required packages...")
subprocess.run([sys.executable, "-m", "pip", "install",
                "gguf", "sentencepiece", "transformers",
                "numpy", "torch", "--quiet"], check=True)

print("Step 2: Downloading conversion scripts from correct commit...")
import urllib.request

files_to_download = {
    f"https://raw.githubusercontent.com/ggerganov/llama.cpp/{COMMIT}/convert_hf_to_gguf.py":
        CONV_DIR / "convert_hf_to_gguf.py",
}

# Get list of files in conversion/ folder
conv_files = [
    "__init__.py", "tensor_mapping.py",
]

for fname in conv_files:
    url = f"https://raw.githubusercontent.com/ggerganov/llama.cpp/{COMMIT}/conversion/{fname}"
    files_to_download[url] = CONV_DIR / "conversion" / fname

(CONV_DIR / "conversion").mkdir(exist_ok=True)

for url, dest in files_to_download.items():
    print(f"  Downloading {dest.name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  OK: {dest.name}")
    except Exception as e:
        print(f"  WARN: Could not download {dest.name}: {e}")

# Check what we got
conv_script = CONV_DIR / "convert_hf_to_gguf.py"
if not conv_script.exists():
    print("\nERROR: Could not download conversion script.")
    print("Manual alternative:")
    print(f"  1. Go to: https://github.com/ggerganov/llama.cpp/tree/{COMMIT}")
    print("  2. Download convert_hf_to_gguf.py and the conversion/ folder")
    print(f"  3. Place them in: {CONV_DIR}")
    sys.exit(1)

print(f"\nStep 3: Running conversion...")
print(f"  Input:  {MERGED}")
print(f"  Output: {OUT_F16}")

result = subprocess.run([
    sys.executable,
    str(conv_script),
    str(MERGED),
    "--outtype", "f16",
    "--outfile", str(OUT_F16),
], capture_output=False)

if result.returncode != 0:
    print("\nConversion failed. Trying with --vocab-type bpe...")
    result = subprocess.run([
        sys.executable,
        str(conv_script),
        str(MERGED),
        "--outtype", "f16",
        "--outfile", str(OUT_F16),
        "--vocab-type", "bpe",
    ])

if OUT_F16.exists():
    size_gb = OUT_F16.stat().st_size / 1024**3
    print(f"\nGGUF f16 created: {OUT_F16} ({size_gb:.2f} GB)")

    print(f"\nStep 4: Quantizing to Q4_K_M...")
    quantize_exe = LLAMA_DIR / "llama-quantize.exe"
    if quantize_exe.exists():
        subprocess.run([
            str(quantize_exe),
            str(OUT_F16),
            str(OUT_Q4),
            "Q4_K_M"
        ])
        if OUT_Q4.exists():
            size_gb = OUT_Q4.stat().st_size / 1024**3
            print(f"\nQ4_K_M GGUF created: {OUT_Q4} ({size_gb:.2f} GB)")
            print("\nNow update ollama/Modelfile:")
            print("  Change: FROM qwen2.5:1.5b-instruct")
            print("  To:     FROM ./researchpilot-q4km.gguf")
            print("\nThen run:")
            print("  $ollama = 'C:\\Users\\LOQ\\AppData\\Local\\Programs\\Ollama\\ollama.exe'")
            print("  & $ollama rm researchpilot")
            print("  Set-Location 'E:\\Tuned_Research'")
            print("  & $ollama create researchpilot --file ollama\\Modelfile")
    else:
        print(f"llama-quantize.exe not found at {quantize_exe}")
else:
    print("\nConversion failed — GGUF file not created.")
    print("\nFallback: The base model with ResearchPilot system prompt is")
    print("already running. Your fine-tuned adapter is saved at:")
    print(f"  {ROOT / 'models' / 'researchpilot-lora'}")
    print("The GGUF conversion can be done later when llama.cpp source is available.")
