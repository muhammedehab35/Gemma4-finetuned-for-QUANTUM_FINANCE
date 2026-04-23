"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Fine-tuning Gemma 4 E4B — Quantitative Finance                           ║
║   Framework : Unsloth + QLoRA + HuggingFace Hub                             ║
║   Target GPU : vast.ai RTX 4090 (24 GB VRAM)                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTRUCTIONS (vast.ai):
  1. Connect via SSH:
       ssh -p PORT root@IP -L 8080:localhost:8080
  2. Upload files from your local machine:
       scp -P PORT .env root@IP:/workspace/
       scp -P PORT dataset/quant_finance_dataset.json root@IP:/workspace/
       scp -P PORT finetune_vastai.py root@IP:/workspace/
  3. Run:
       cd /workspace && pip uninstall torchao -y -q && python finetune_vastai.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Installation
# ══════════════════════════════════════════════════════════════════════════════
import subprocess, sys

def install():
    print("Installing dependencies...")
    cmds = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "-q"],
        # PyTorch cu124 (Ada Lovelace compatible — RTX 40xx)
        [sys.executable, "-m", "pip", "install",
         "torch", "torchvision",
         "--index-url", "https://download.pytorch.org/whl/cu124", "-q"],
        # Core dependencies required by trl/peft
        [sys.executable, "-m", "pip", "install",
         "transformers", "safetensors", "jinja2", "-q"],
        # Unsloth
        [sys.executable, "-m", "pip", "install",
         "unsloth @ git+https://github.com/unslothai/unsloth.git", "-q"],
        # TRL + HuggingFace stack
        [sys.executable, "-m", "pip", "install",
         "trl", "peft", "accelerate", "bitsandbytes",
         "huggingface_hub", "datasets", "python-dotenv", "-q"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=True)
    print("✅ Dependencies installed")

install()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Imports and Configuration
# ══════════════════════════════════════════════════════════════════════════════
import os
import json
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login

# Load .env from /workspace/
env_file = Path("/workspace/.env")
if env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(env_file)
    print("✅ Variables loaded from /workspace/.env")

# HuggingFace credentials
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_USERNAME = os.environ.get("HF_USERNAME", "mo35")

if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN missing.\n"
        "Set it with: export HF_TOKEN='hf_...'\n"
        "Or create /workspace/.env with HF_TOKEN=hf_..."
    )
print(f"✅ HF_TOKEN loaded (user: {HF_USERNAME})")

# HuggingFace repositories
MODEL_REPO   = f"{HF_USERNAME}/gemma4-quantfin-lora"
DATASET_REPO = f"{HF_USERNAME}/quant-finance-dataset"
GGUF_REPO    = f"{HF_USERNAME}/gemma4-quantfin-gguf"

# Paths on vast.ai (/workspace/)
DATA_FILE  = "/workspace/dataset/quant_finance_dataset.json"
if not os.path.exists(DATA_FILE):
    DATA_FILE = "/workspace/quant_finance_dataset.json"
OUTPUT_DIR = "/workspace/outputs"
LORA_DIR   = "/workspace/gemma4-quantfin-lora"
GGUF_DIR   = "/workspace/gemma4-quantfin-gguf"

for d in [OUTPUT_DIR, LORA_DIR, GGUF_DIR]:
    os.makedirs(d, exist_ok=True)

# Model
UNSLOTH_MODEL = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

# LoRA hyperparameters
MAX_SEQ_LEN = 4096
LORA_RANK   = 32
LORA_ALPHA  = 64      # = 2 × rank

# Training hyperparameters — calibrated for 24 examples
# 24 examples, packing=False → 21 train sequences
# BATCH_SIZE=1, GRAD_ACCUM=4 → ~5 steps/epoch
# EPOCHS=10 → ~52 total steps
EPOCHS     = 10
LR         = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 4

print(f"\n{'='*55}")
print("CONFIGURATION")
print(f"{'='*55}")
print(f"  Model           : {UNSLOTH_MODEL}")
print(f"  HF Model repo   : {MODEL_REPO}")
print(f"  HF Dataset repo : {DATASET_REPO}")
print(f"  Dataset file    : {DATA_FILE}")
if torch.cuda.is_available():
    gpu  = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory // 1024**3
    print(f"  GPU             : {gpu}")
    print(f"  VRAM            : {vram} GB")
else:
    print("  GPU             : ❌ CUDA not available")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Load Gemma 4 E4B via Unsloth
# ══════════════════════════════════════════════════════════════════════════════
from unsloth import FastModel

print("\nLoading Gemma 4 E4B via Unsloth (QLoRA 4-bit)...")
model, tokenizer = FastModel.from_pretrained(
    model_name     = UNSLOTH_MODEL,
    max_seq_length = MAX_SEQ_LEN,
    load_in_4bit   = True,
    load_in_16bit  = False,
    token          = HF_TOKEN,
)

# Apply LoRA adapters
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r                          = LORA_RANK,
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = 0,
    bias                       = "none",
    random_state               = 42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"✅ Model loaded")
print(f"   Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Load and format dataset
# ══════════════════════════════════════════════════════════════════════════════
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Upload quant_finance_dataset.json to /workspace/ or /workspace/dataset/"
    )

with open(DATA_FILE, encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"\nDataset loaded: {len(raw_data)} examples")

def format_conversations(item: dict) -> dict:
    """Convert conversations[] format to Gemma 4 chat text."""
    convs = item.get("conversations", [])
    if len(convs) < 2:
        return {"text": "", "skip": True}
    if convs[0].get("role") != "user":
        return {"text": "", "skip": True}
    if convs[1].get("role") not in ("assistant", "model"):
        return {"text": "", "skip": True}

    text = tokenizer.apply_chat_template(
        convs,
        tokenize              = False,
        add_generation_prompt = False,
    )
    eos = tokenizer.eos_token or "<eos>"
    if not text.endswith(eos):
        text = text + eos
    return {"text": text, "skip": False}


formatted_data = [format_conversations(item) for item in raw_data]
valid_data     = [d for d in formatted_data if not d.get("skip", False) and d["text"]]
print(f"Valid examples: {len(valid_data)} / {len(raw_data)}")

if len(valid_data) == 0:
    raise ValueError("No valid examples found in dataset.")

dataset = Dataset.from_list([{"text": d["text"]} for d in valid_data])
split   = dataset.train_test_split(test_size=0.1, seed=42)

print(f"\nSplit:")
print(f"  Train : {len(split['train'])} examples")
print(f"  Test  : {len(split['test'])} examples")

# Token length estimate (Gemma 4 uses multimodal processor — use char/4 estimate)
lengths   = [len(t) // 4 for t in [split["train"][i]["text"] for i in range(min(len(split["train"]), 20))]]
est_steps = (len(split["train"]) * EPOCHS) // (BATCH_SIZE * GRAD_ACCUM)
print(f"\nToken stats (estimated):")
print(f"  Min: {min(lengths)} | Max: {max(lengths)} | Avg: {int(np.mean(lengths))}")
print(f"\nTraining estimate:")
print(f"  Total steps : {est_steps}")
print(f"  ETA         : ~{max(1, est_steps * 25 // 60)} min (RTX 4090)")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Upload dataset to HuggingFace Hub
# ══════════════════════════════════════════════════════════════════════════════
def push_dataset_to_hub() -> bool:
    print("\nUploading dataset to HuggingFace Hub...")
    try:
        login(token=HF_TOKEN)
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=DATASET_REPO, repo_type="dataset",
                        private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj = DATA_FILE,
            path_in_repo    = "quant_finance_dataset.json",
            repo_id         = DATASET_REPO,
            repo_type       = "dataset",
            commit_message  = "Add quantitative finance Q&A dataset",
        )
        DatasetDict({"train": split["train"], "test": split["test"]}
                    ).push_to_hub(repo_id=DATASET_REPO, token=HF_TOKEN)

        readme = f"""---
license: apache-2.0
language:
- en
tags:
- quantitative-finance
- mathematical-finance
- stochastic-calculus
- options-pricing
- fine-tuning
size_categories:
- n<1K
---

# Quantitative Finance Fine-Tuning Dataset

A dataset of {len(raw_data)} high-quality Q&A examples for fine-tuning LLMs on quantitative finance topics.

## Categories

| Category | Topics | Examples |
|---|---|---|
| Volatility models | SABR (corrected), Bergomi, rBergomi, Heston | 5 |
| Derivatives pricing | Dupire, VIX, BS Greeks, CVaR | 5 |
| Rates & credit | HJM, Hull-White, Merton, CDS | 4 |
| Numerical methods | Crank-Nicolson, Monte Carlo, FFT, LSM | 5 |
| Quant strategies | Momentum, Pairs Trading, VaR, Fama-French | 5 |

## Key Corrections vs Base Model

| Base Model Error | Correction |
|---|---|
| SABR attributed to HJM | Hagan, Kumar, Lesniewski & Woodward (2002) |
| SABR vol with mean reversion | Log-normal GBM, no drift |
| Bergomi = Heston/CIR | Forward variance curve ξᵗᵤ |
| rBergomi like standard BM | Fractional BM with H < 0.5 (rough vol) |

## Data Format

```json
{{
  "conversations": [
    {{"role": "user", "content": "Quantitative finance question..."}},
    {{"role": "assistant", "content": "Rigorous answer with LaTeX math..."}}
  ]
}}
```

## Fine-tuned Model

[{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})
"""
        api.upload_file(path_or_fileobj=readme.encode("utf-8"),
                        path_in_repo="README.md", repo_id=DATASET_REPO,
                        repo_type="dataset")
        print(f"✅ Dataset: https://huggingface.co/datasets/{DATASET_REPO}")
        return True
    except Exception as e:
        print(f"⚠️  Dataset upload error: {e}")
        return False


push_dataset_to_hub()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Fine-tuning with SFTTrainer
# ══════════════════════════════════════════════════════════════════════════════
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = split["train"],
    eval_dataset       = split["test"],
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LEN,
    dataset_num_proc   = 2,
    packing            = False,
    args = SFTConfig(
        output_dir                    = OUTPUT_DIR,
        per_device_train_batch_size   = BATCH_SIZE,
        gradient_accumulation_steps   = GRAD_ACCUM,
        warmup_ratio                  = 0.1,
        num_train_epochs              = EPOCHS,
        learning_rate                 = LR,
        bf16                          = torch.cuda.is_bf16_supported(),
        fp16                          = not torch.cuda.is_bf16_supported(),
        optim                         = "adamw_8bit",
        weight_decay                  = 0.01,
        lr_scheduler_type             = "cosine",
        logging_steps                 = 1,
        eval_strategy                 = "epoch",
        save_strategy                 = "epoch",
        save_total_limit              = 2,
        load_best_model_at_end        = True,
        metric_for_best_model         = "eval_loss",
        neftune_noise_alpha           = 5,
        seed                          = 42,
        report_to                     = "none",
        dataloader_pin_memory         = False,
    ),
)

print(f"\n{'='*55}")
print("STARTING TRAINING")
print(f"{'='*55}")
print(f"  Epochs         : {EPOCHS}")
print(f"  Learning rate  : {LR}")
print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  Train samples  : {len(split['train'])}")

trainer_stats = trainer.train()

runtime = trainer_stats.metrics.get("train_runtime", 0)
loss    = trainer_stats.metrics.get("train_loss", 0)
print(f"\n✅ Training complete!")
print(f"  Time : {runtime:.0f}s ({runtime/60:.1f} min)")
print(f"  Loss : {loss:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Post-training evaluation
# ══════════════════════════════════════════════════════════════════════════════
FastModel.for_inference(model)

def generate(question: str, max_new_tokens: int = 600) -> str:
    """Generate a response using the fine-tuned model."""
    messages = [{"role": "user", "content": [{"type": "text", "text": question}]}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs, max_new_tokens=max_new_tokens,
            do_sample=False, repetition_penalty=1.1, use_cache=True,
        )
    return tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)


EVAL_QUESTIONS = [
    ("SABR — attribution + SDEs",
     "Write the SABR model SDEs. Who wrote the original paper? "
     "Does the volatility process have mean reversion?"),
    ("Bergomi — forward variance",
     "What is ξᵗᵤ in the Bergomi model? How is this different from Heston?"),
    ("CDS — par spread",
     "Under constant hazard rate h and recovery R, what is the par CDS spread s?"),
    ("HJM — drift condition",
     "State the HJM no-arbitrage drift condition and explain why drift is "
     "determined by volatility."),
]

print(f"\n{'='*60}")
print("POST-TRAINING EVALUATION")
print(f"{'='*60}")
for i, (name, question) in enumerate(EVAL_QUESTIONS, 1):
    print(f"\n[{i}] {name}")
    print(f"Q: {question}")
    print("-" * 60)
    try:
        print(f"A: {generate(question)}")
    except Exception as e:
        print(f"⚠️  Generation error: {e}")
    print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Save + push to HuggingFace Hub
# ══════════════════════════════════════════════════════════════════════════════

# 8a — Save LoRA adapters locally
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"\n✅ LoRA saved → {LORA_DIR}/")

# 8b — Export GGUF (for Ollama / llama.cpp)
gguf_ok = False
try:
    model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")
    print(f"✅ GGUF exported → {GGUF_DIR}/")
    gguf_ok = True
except Exception as e:
    print(f"⚠️  GGUF export: {e}")

# 8c — Push model to HuggingFace Hub
def push_model_to_hub() -> bool:
    print("\nPushing model to HuggingFace Hub...")
    try:
        login(token=HF_TOKEN)
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=MODEL_REPO, private=False, exist_ok=True)

        model.push_to_hub(MODEL_REPO, token=HF_TOKEN)
        tokenizer.push_to_hub(MODEL_REPO, token=HF_TOKEN)
        print(f"✅ LoRA pushed: https://huggingface.co/{MODEL_REPO}")

        model_card = f"""---
base_model: google/gemma-4-E4B-it
license: gemma
language:
- en
tags:
- quantitative-finance
- lora
- unsloth
- gemma4
pipeline_tag: text-generation
---

# Gemma 4 E4B — Quantitative Finance (LoRA)

Fine-tuned on {len(raw_data)} Q&A examples covering quantitative finance via QLoRA (rank={LORA_RANK}).

## Errors Corrected vs Base Model

| Base Model Error | Correction |
|---|---|
| SABR attributed to HJM | Hagan, Kumar, Lesniewski & Woodward (2002) |
| SABR vol with mean reversion | Log-normal GBM, no drift |
| Bergomi = Heston/CIR | Forward variance curve ξᵗᵤ |
| SABR formula invented | Correct formula with z, χ(z), T corrections |

## Training Details

| Parameter | Value |
|---|---|
| Base model | google/gemma-4-E4B-it |
| GPU | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| Method | QLoRA 4-bit |
| LoRA rank | {LORA_RANK} |
| LoRA alpha | {LORA_ALPHA} |
| Dataset | [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO}) |
| Training examples | 21 |
| Epochs | {EPOCHS} |
| NEFTune noise alpha | 5 |
| Max seq length | {MAX_SEQ_LEN} |
| Final loss | {loss:.4f} |

## Topics Covered

- Stochastic volatility: Heston, SABR, Bergomi, rBergomi (rough vol)
- Local volatility: Dupire, model-free VIX
- Interest rates: Vasicek, Hull-White, HJM
- Credit risk: Merton structural, CDS
- Numerical methods: Crank-Nicolson, Monte Carlo, FFT, Longstaff-Schwartz
- Quant strategies: Momentum, Pairs Trading, Fama-French, VaR/CVaR

## Quick Usage

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    "{MODEL_REPO}", max_seq_length=4096, load_in_4bit=True
)
FastModel.for_inference(model)

messages = [{{"role": "user", "content": [{{"type": "text", "text": "Derive the SABR implied volatility formula."}}]}}]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", add_generation_prompt=True
).to("cuda")
print(tokenizer.decode(
    model.generate(inputs, max_new_tokens=1024)[0][inputs.shape[-1]:],
    skip_special_tokens=True
))
```
"""
        api.upload_file(path_or_fileobj=model_card.encode("utf-8"),
                        path_in_repo="README.md", repo_id=MODEL_REPO,
                        commit_message="Add model card")

        if gguf_ok:
            api.create_repo(repo_id=GGUF_REPO, private=False, exist_ok=True)
            model.push_to_hub_gguf(GGUF_REPO, tokenizer,
                                   quantization_method=["q4_k_m", "q8_0"],
                                   token=HF_TOKEN)
            print(f"✅ GGUF pushed: https://huggingface.co/{GGUF_REPO}")

        print(f"\n{'='*55}")
        print("ALL UPLOADS COMPLETE")
        print(f"{'='*55}")
        print(f"  Model  : https://huggingface.co/{MODEL_REPO}")
        print(f"  Dataset: https://huggingface.co/datasets/{DATASET_REPO}")
        if gguf_ok:
            print(f"  GGUF   : https://huggingface.co/{GGUF_REPO}")
        return True
    except Exception as e:
        print(f"⚠️  Model push error: {e}")
        return False


push_model_to_hub()
