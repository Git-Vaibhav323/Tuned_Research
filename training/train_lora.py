"""
ResearchPilot — training/train_lora.py

Windows-native LoRA / QLoRA fine-tuning for Qwen2.5-1.5B-Instruct.

Strategy (auto-selected at runtime):
  PRIMARY:   Standard LoRA in fp16     (always available with PyTorch+CUDA)
  SECONDARY: QLoRA 4-bit NF4           (if bitsandbytes 4-bit works on this GPU)

Run:
    py -3.12 training/train_lora.py

To force fp16 LoRA (skip 4-bit probe):
    py -3.12 training/train_lora.py --no-qlora

To override model:
    py -3.12 training/train_lora.py --model Qwen/Qwen2.5-3B-Instruct

BEFORE RUNNING:
  1. Run install_packages.ps1 to install PyTorch + HF stack
  2. Re-run check_environment.py to confirm GPU is detected
  3. Close ChatGPT, browser, and other GPU apps to free VRAM
  4. Run prepare_dataset.py to generate data/finetuning/train.jsonl

GPU memory requirement:
  fp16 LoRA:  ~5.1 GB  (need ~5.5 GB free — close GPU apps first)
  QLoRA 4-bit: ~2.2 GB (more comfortable)
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

# ── argument parsing (before heavy imports) ───────────────────────────────────
parser = argparse.ArgumentParser(description="ResearchPilot LoRA fine-tuning")
parser.add_argument("--model",      default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="HuggingFace model ID")
parser.add_argument("--no-qlora",   action="store_true",
                    help="Force fp16 LoRA, skip 4-bit probe")
parser.add_argument("--smoke-test", action="store_true",
                    help="Run 5 steps only (sanity check, no real training)")
parser.add_argument("--epochs",     type=int, default=3)
parser.add_argument("--max-steps",  type=int, default=-1,
                    help="Override epochs with a fixed step count")
parser.add_argument("--output",     default=None,
                    help="Output directory for adapter (default: models/researchpilot-lora)")
args = parser.parse_args()

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data" / "finetuning"
OUTPUT    = Path(args.output) if args.output else ROOT / "models" / "researchpilot-lora"
CKPT_DIR  = ROOT / "models" / "checkpoints"
OUTPUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ── verify prerequisites ──────────────────────────────────────────────────────
for fname in ("train.jsonl", "validation.jsonl"):
    if not (DATA_DIR / fname).exists():
        print(f"ERROR: {DATA_DIR / fname} not found.")
        print("       Run: py -3.12 training/prepare_dataset.py")
        sys.exit(1)

# ── imports (after arg parse so --help works without torch installed) ─────────
print("Loading libraries ...")
try:
    import torch
except ImportError:
    print("ERROR: PyTorch not installed. Run: .\\training\\install_packages.ps1")
    sys.exit(1)

if not torch.cuda.is_available():
    print("ERROR: CUDA not available. Check PyTorch installation.")
    print("       Expected: torch 2.3.1+cu121")
    print("       Installed:", torch.__version__)
    sys.exit(1)

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
import accelerate

print(f"  PyTorch:        {torch.__version__}")
print(f"  transformers:   {transformers.__version__}")
print(f"  accelerate:     {accelerate.__version__}")
import peft; print(f"  peft:           {peft.__version__}")
import trl;  print(f"  trl:            {trl.__version__}")

from trl import SFTTrainer, SFTConfig

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"\nGPU: {gpu_name}  |  VRAM: {vram_gb:.1f} GB")

# ── probe 4-bit support ───────────────────────────────────────────────────────
use_qlora = False
if not args.no_qlora:
    try:
        import bitsandbytes as bnb
        _ = bnb.nn.Linear4bit(16, 16)
        use_qlora = True
        print(f"bitsandbytes {bnb.__version__}: 4-bit OK → using QLoRA")
    except Exception as e:
        print(f"bitsandbytes 4-bit probe failed ({e}) → falling back to fp16 LoRA")
else:
    print("--no-qlora flag set → using fp16 LoRA")

# ── configuration ─────────────────────────────────────────────────────────────
MODEL_ID = args.model
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
MAX_SEQ_LEN  = 256 if not use_qlora else 384
MICRO_BS     = 1
GRAD_ACCUM   = 8   # effective batch = 8
LR           = 2e-4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
SEED         = 42

# Qwen2 / Qwen2.5 LoRA target modules (verified from model config)
LORA_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

print(f"\n{'='*55}")
print(f"  Training configuration")
print(f"{'='*55}")
print(f"  Model:          {MODEL_ID}")
print(f"  Mode:           {'QLoRA 4-bit NF4' if use_qlora else 'LoRA fp16'}")
print(f"  LoRA rank:      {LORA_R}  alpha={LORA_ALPHA}  dropout={LORA_DROPOUT}")
print(f"  Max seq len:    {MAX_SEQ_LEN}")
print(f"  Micro batch:    {MICRO_BS}  grad_accum={GRAD_ACCUM}  (eff. batch={MICRO_BS*GRAD_ACCUM})")
print(f"  Learning rate:  {LR}")
print(f"  Epochs:         {args.epochs}")
print(f"  Smoke-test:     {args.smoke_test}")
print(f"  Output:         {OUTPUT}")
print(f"{'='*55}\n")

# ── load tokenizer ────────────────────────────────────────────────────────────
print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    padding_side="right",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"  Vocab size: {tokenizer.vocab_size}")

# ── load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model ({MODEL_ID}) ...")

if use_qlora:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

model.config.use_cache = False
if not use_qlora:
    model.enable_input_require_grads()

# ── LoRA config ───────────────────────────────────────────────────────────────
# Verify that the target modules actually exist in this model
actual_names = {name for name, _ in model.named_modules()}
valid_targets = [t for t in LORA_TARGETS
                 if any(t in n for n in actual_names)]
if not valid_targets:
    print("WARN: None of the default target modules found. Using all linear layers.")
    valid_targets = None   # PEFT will auto-detect

lora_cfg = LoraConfig(
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    target_modules = valid_targets,
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM,
)

if not use_qlora:
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

# ── dataset loading ───────────────────────────────────────────────────────────
print("\nLoading datasets ...")

def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples

train_raw = load_jsonl(DATA_DIR / "train.jsonl")
val_raw   = load_jsonl(DATA_DIR / "validation.jsonl")
print(f"  train: {len(train_raw)}  |  validation: {len(val_raw)}")

if args.smoke_test:
    train_raw = train_raw[:20]
    val_raw   = val_raw[:5]
    print(f"  [smoke-test] truncated to train={len(train_raw)}, val={len(val_raw)}")

def format_example(example: dict) -> str:
    """Apply the model's chat template to produce a training string."""
    # Strip _meta before sending to model
    msgs = example["messages"]
    text = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text

# Build HF datasets from raw lists
from datasets import Dataset as HFDataset

def build_hf_dataset(raw: list[dict]) -> HFDataset:
    texts = [format_example(ex) for ex in raw]
    return HFDataset.from_dict({"text": texts})

train_ds = build_hf_dataset(train_raw)
val_ds   = build_hf_dataset(val_raw)

# ── training arguments ────────────────────────────────────────────────────────
steps_override = 5 if args.smoke_test else (args.max_steps if args.max_steps > 0 else -1)

sft_config = SFTConfig(
    output_dir                  = str(CKPT_DIR),
    num_train_epochs            = args.epochs if not args.smoke_test else 1,
    max_steps                   = steps_override,
    per_device_train_batch_size = MICRO_BS,
    per_device_eval_batch_size  = MICRO_BS,
    gradient_accumulation_steps = GRAD_ACCUM,
    gradient_checkpointing      = True,
    optim                       = "paged_adamw_8bit" if use_qlora else "adamw_torch",
    learning_rate               = LR,
    weight_decay                = WEIGHT_DECAY,
    warmup_ratio                = WARMUP_RATIO,
    lr_scheduler_type           = "cosine",
    fp16                        = True,
    bf16                        = False,
    logging_steps               = 10,
    eval_strategy               = "steps",
    eval_steps                  = 50,
    save_strategy               = "steps",
    save_steps                  = 100,
    save_total_limit            = 2,
    load_best_model_at_end      = False,
    report_to                   = "none",
    seed                        = SEED,
    max_seq_length              = MAX_SEQ_LEN,
    packing                     = False,
    dataset_text_field          = "text",
)

# ── trainer ───────────────────────────────────────────────────────────────────
print("\nInitialising SFTTrainer ...")
trainer = SFTTrainer(
    model          = model,
    args           = sft_config,
    train_dataset  = train_ds,
    eval_dataset   = val_ds,
    peft_config    = lora_cfg if use_qlora else None,
)

# ── memory snapshot before training ───────────────────────────────────────────
allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
reserved_gb  = torch.cuda.memory_reserved(0)  / 1024**3
print(f"\nVRAM before training: allocated={allocated_gb:.2f} GB  reserved={reserved_gb:.2f} GB")
if allocated_gb > 5.8:
    print("WARN: VRAM very tight. If OOM occurs, close other GPU apps and retry.")

# ── train ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Starting training ...")
print(f"{'='*55}")
t0 = time.time()

try:
    train_result = trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"  train_loss:     {train_result.training_loss:.4f}")
    print(f"  steps:          {train_result.global_step}")
except torch.cuda.OutOfMemoryError:
    print("\n" + "!"*55)
    print("  CUDA OUT OF MEMORY")
    print("  Solutions:")
    print("  1. Close ChatGPT Classic, Edge, and other GPU apps")
    print("  2. Reduce MAX_SEQ_LEN: edit train_lora.py line ~60")
    print("  3. Use QLoRA: remove --no-qlora flag")
    print("  4. Run: py -3.12 training/train_lora.py --smoke-test first")
    print("!"*55)
    sys.exit(1)

# ── save adapter ─────────────────────────────────────────────────────────────
print(f"\nSaving adapter to {OUTPUT} ...")
trainer.model.save_pretrained(str(OUTPUT))
tokenizer.save_pretrained(str(OUTPUT))

# Save training config for reproducibility
config_out = {
    "model_id":      MODEL_ID,
    "mode":          "qlora_4bit" if use_qlora else "lora_fp16",
    "lora_r":        LORA_R,
    "lora_alpha":    LORA_ALPHA,
    "lora_dropout":  LORA_DROPOUT,
    "max_seq_len":   MAX_SEQ_LEN,
    "micro_bs":      MICRO_BS,
    "grad_accum":    GRAD_ACCUM,
    "lr":            LR,
    "epochs":        args.epochs,
    "train_examples": len(train_raw),
    "val_examples":  len(val_raw),
    "train_loss":    round(train_result.training_loss, 4),
    "steps":         train_result.global_step,
    "elapsed_sec":   round(elapsed, 1),
    "gpu":           gpu_name,
    "vram_gb":       round(vram_gb, 1),
    "torch_ver":     torch.__version__,
    "smoke_test":    args.smoke_test,
}
with open(OUTPUT / "training_config.json", "w") as fh:
    json.dump(config_out, fh, indent=2)

print(f"  Adapter saved: {OUTPUT}")
print(f"  Contents: {[f.name for f in OUTPUT.iterdir()]}")

print(f"\n{'='*55}")
print(f"  train_lora.py  COMPLETE")
print(f"  Next step: py -3.12 training/merge_adapter.py")
print(f"{'='*55}")
