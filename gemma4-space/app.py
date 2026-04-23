"""
HuggingFace Space — Gemma 4 Quantitative Finance Chat
Hardware: Nvidia T4 medium (16 GB VRAM)
"""

import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_USERNAME = os.environ.get("HF_USERNAME", "mo35")
BASE_MODEL  = "google/gemma-4-E4B-it"
LORA_REPO   = f"{HF_USERNAME}/gemma4-quantfin-lora"

# ── Load model at startup ─────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)

print("Loading base model in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_compute_dtype    = torch.bfloat16,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type       = "nf4",
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config = bnb_config,
    device_map          = "auto",
    token               = HF_TOKEN,
)

print(f"Loading LoRA adapter from {LORA_REPO}...")
model = PeftModel.from_pretrained(base_model, LORA_REPO, token=HF_TOKEN)
model.eval()
print("Model ready.")

# ── Inference ─────────────────────────────────────────────────────────────────
def respond(message: str, history: list) -> str:
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    # apply_chat_template returns BatchEncoding in newer transformers
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt = True,
        return_tensors        = "pt",
        return_dict           = True,
    )
    input_ids      = encoded["input_ids"].to(model.device)
    attention_mask = encoded.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask     = attention_mask,
            max_new_tokens     = 1024,
            temperature        = 0.7,
            do_sample          = True,
            repetition_penalty = 1.1,
        )

    return tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens = True,
    )

# ── Gradio UI ─────────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn          = respond,
    type        = "messages",
    title       = "Gemma 4 — Quantitative Finance",
    description = (
        "A specialized AI assistant fine-tuned on quantitative finance: derivatives pricing, "
        "stochastic calculus, risk models, and portfolio theory. "
        "Answers include LaTeX mathematical derivations."
    ),
    examples    = [
        "Derive the Black-Scholes PDE from first principles.",
        "Explain the SABR model and its implied volatility approximation.",
        "What is the difference between risk-neutral and real-world measures?",
        "Derive the Heston model characteristic function.",
        "Explain Value at Risk vs Expected Shortfall.",
    ],
    theme          = gr.themes.Soft(),
    cache_examples = False,
)

demo.launch()
