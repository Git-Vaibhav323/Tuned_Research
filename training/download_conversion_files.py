"""
Downloads all files needed for llama.cpp GGUF conversion from the exact v0.6.0 commit.
Run: py -3.12 training/download_conversion_files.py
"""
import urllib.request
import sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
CONV_DIR = ROOT / "llama_convert"
COMMIT   = "d812350"

BASE_URL = f"https://raw.githubusercontent.com/ggerganov/llama.cpp/{COMMIT}"

# All files in the conversion/ package at this commit
CONVERSION_FILES = [
    "__init__.py", "base.py", "tensor_mapping.py",
    "qwen2.py", "qwen2moe.py", "llama.py", "mistral.py",
    "gemma.py", "gemma2.py", "phi2.py", "phi3.py",
    "stablelm.py", "starcoder2.py", "xverse.py",
    "refact.py", "internlm2.py", "minicpm.py",
    "bloom.py", "falcon.py", "gpt2.py", "gptneox.py",
    "mpt.py", "baichuan.py", "bert.py", "nomic_bert.py",
    "chatglm.py", "orion.py", "command_r.py",
    "olmo.py", "olmo2.py", "arctic.py", "deepseek.py",
    "deepseek2.py", "exaone.py", "granite.py",
    "granite_moe.py", "chameleon.py", "rwkv6.py",
    "t5.py", "t5encoder.py", "jais.py", "nemotron.py",
    "solar.py", "smollm3.py",
]

(CONV_DIR / "conversion").mkdir(parents=True, exist_ok=True)

print(f"Downloading conversion files from commit {COMMIT}...")
success = 0
failed  = []

for fname in CONVERSION_FILES:
    url  = f"{BASE_URL}/conversion/{fname}"
    dest = CONV_DIR / "conversion" / fname
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  OK  {fname}")
        success += 1
    except Exception:
        failed.append(fname)

print(f"\nDownloaded: {success}  |  Not found: {len(failed)}")
if failed:
    print(f"Missing (will try to continue anyway): {failed[:5]}")

# Verify base.py exists — it's the critical one
base_py = CONV_DIR / "conversion" / "base.py"
if base_py.exists():
    print("\nbase.py found — conversion should work now.")
    print("Run: py -3.12 training/convert_to_gguf.py")
else:
    print("\nERROR: base.py not found. The conversion/ structure may have changed.")
    print("Trying alternative: download the whole conversion package as a zip...")

    # Try downloading the source zip and extracting just conversion/
    import zipfile, io
    zip_url = f"https://github.com/ggerganov/llama.cpp/archive/{COMMIT}.zip"
    print(f"Downloading source zip from {zip_url} (~10 MB)...")
    try:
        with urllib.request.urlopen(zip_url, timeout=60) as resp:
            data = resp.read()
        z = zipfile.ZipFile(io.BytesIO(data))
        prefix = f"llama.cpp-{COMMIT}/conversion/"
        extracted = 0
        for name in z.namelist():
            if name.startswith(prefix) and name.endswith(".py"):
                fname = name.split("/")[-1]
                dest  = CONV_DIR / "conversion" / fname
                dest.write_bytes(z.read(name))
                print(f"  Extracted: {fname}")
                extracted += 1
        print(f"\nExtracted {extracted} files from source zip.")
        if (CONV_DIR / "conversion" / "base.py").exists():
            print("base.py found — run: py -3.12 training/convert_to_gguf.py")
        else:
            print("Still missing base.py — conversion not possible with this approach.")
    except Exception as e:
        print(f"Zip download failed: {e}")
